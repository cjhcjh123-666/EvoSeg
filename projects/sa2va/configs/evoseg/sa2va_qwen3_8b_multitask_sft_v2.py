"""EvoSeg SFT stage 2 (multi-task): Qwen3-VL-4B + Sa2VA-4B warm start.

Data mix (Sa2VA-style concat + repeats):
  RefCOCO(x5) + RefCOCO+(x5) + RefCOCOg(x6) + gRefCOCO(x2) + ReasonSeg(x10)
  + MeViS(x6) + RefYTVOS(x6) + Ryvos(x3) + LLaVA-150K(x0.3 anti-forgetting)
# v2: up-weight RefCOCOg + video to fix 8B's RefCOCOg/video gaps
Recipe: LoRA r128/alpha256, freeze LLM+visual, trainable SAM2 decoder,
        lr 4e-5, wd 0.05, warmup 0.05 cosine, mask BCE 2.0 + Dice 0.5.
"""
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen3VLProcessor

from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from peft import LoraConfig

from projects.sa2va.models import Sa2VAModel, SAM2TrainRunner, DirectResize
from projects.sa2va.datasets import (
    sa2va_collect_fn, Sa2VA01RefSeg, Sa2VAFinetuneDataset,
    Sa2VA03RefVOS, LLaVADataset,
)
from projects.sa2va.datasets.data_utils import ConcatDatasetSa2VA
from projects.sa2va.models.mllm.qwen3vl import Qwen3VL

# ---------------- settings ----------------
path = '/9950backfile/chenjiahui/evo_artifacts/models/Qwen3-VL-8B-Instruct'
pretrained_pth = None  # 8B: train from scratch (no Sa2VA-8B pretrained)
DATA_ROOT = '/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/pixel_llm_data/'
VIDEO_ROOT = '/9950backfile/chenjiahui/evo_artifacts/datasets/'
MEVIS_FRAMES = VIDEO_ROOT + 'mevis_v2/train/JPEGImages'
MEVIS_META = VIDEO_ROOT + 'mevis_v2/train/meta_expressions_v2.json'
MEVIS_MASK = VIDEO_ROOT + 'mevis_v2/train/mask_dict.json'
RYTVOS_FRAMES = VIDEO_ROOT + 'ref_youtube_vos/extracted/train/JPEGImages'
RYTVOS_META = VIDEO_ROOT + 'ref_youtube_vos/extracted/meta_expressions/train/meta_expressions.json'
RYTVOS_MASK = DATA_ROOT + 'ref_youtube_vos/mask_dict.pkl'

template = "qwen_chat"
prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 4096

batch_size = 2
accumulative_counts = 8
dataloader_num_workers = 8
max_epochs = 1
optim_type = AdamW
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1
warmup_ratio = 0.05
save_steps = 1000
save_total_limit = 3

special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

extra_image_processor = dict(type=DirectResize, target_length=1024)

model = dict(
    type=Sa2VAModel,
    training_bs=batch_size,
    special_tokens=special_tokens,
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,
    frozen_sam2_decoder=False,
    arch_type='qwen',
    mllm=dict(
        type=Qwen3VL,
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=128, lora_alpha=256, lora_dropout=0.05, bias='none',
            task_type='CAUSAL_LM',
            modules_to_save=['lm_head', 'embed_tokens'],
            target_modules=None,
        ),
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(type=SAM2TrainRunner),
    loss_mask=dict(type=CrossEntropyLoss, use_sigmoid=True, reduction='mean', loss_weight=2.0),
    loss_dice=dict(type=DiceLoss, use_sigmoid=True, activate=True, reduction='mean',
                   naive_dice=True, eps=1.0, loss_weight=0.5),
)

sa2va_default_dataset_configs = dict(
    tokenizer=tokenizer, special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    prompt_template=prompt_template, max_length=max_length,
    arch_type='qwen',
    preprocessor=dict(type=Qwen3VLProcessor.from_pretrained,
                      pretrained_model_name_or_path=path, trust_remote_code=True),
)
sa2va_qa_dataset_configs = dict(
    tokenizer=tokenizer, special_tokens=special_tokens,
    prompt_template=prompt_template, max_length=max_length,
    arch_type='qwen',
    preprocessor=dict(type=Qwen3VLProcessor.from_pretrained,
                      pretrained_model_name_or_path=path, trust_remote_code=True),
)

RES_ROOT = DATA_ROOT + 'ref_seg/'
train_dataset = dict(
    type=ConcatDatasetSa2VA, datasets=[
        dict(type=Sa2VA01RefSeg, name='RefCOCO',
             data_root=RES_ROOT + 'refcoco',
             data_prefix=dict(img_path='coco2014/train2014/'),
             ann_file='instances.json', split_file='refs(unc).p',
             split='train', num_classes_per_sample=5, repeats=5,
             serialize_data=False, **sa2va_default_dataset_configs),
        dict(type=Sa2VA01RefSeg, name='RefCOCO+',
             data_root=RES_ROOT + 'refcoco+',
             data_prefix=dict(img_path='coco2014/train2014/'),
             ann_file='instances.json', split_file='refs(unc).p',
             split='train', num_classes_per_sample=5, repeats=5,
             serialize_data=False, **sa2va_default_dataset_configs),
        dict(type=Sa2VA01RefSeg, name='RefCOCOg',
             data_root=RES_ROOT + 'refcocog',
             data_prefix=dict(img_path='coco2014/train2014/'),
             ann_file='instances.json', split_file='refs(umd).p',
             split='train', num_classes_per_sample=5, repeats=6,
             serialize_data=False, **sa2va_default_dataset_configs),
        dict(type=Sa2VAFinetuneDataset, name='gRefCOCO',
             data_root=DATA_ROOT + 'grefcoco/',
             data_prefix=dict(img_path='images/train2014/'),
             ann_file='annotations.json', serialize_data=False,
             repeats=2, **sa2va_default_dataset_configs),
        dict(type=Sa2VAFinetuneDataset, name='ReasonSeg',
             data_root=DATA_ROOT + 'reason_seg/',
             data_prefix=dict(img_path='images/'),
             ann_file='annotations.json', serialize_data=False,
             repeats=10, **sa2va_default_dataset_configs),
        dict(type=Sa2VA03RefVOS, name='MeViS',
             image_folder=MEVIS_FRAMES, expression_file=MEVIS_META,
             mask_file=MEVIS_MASK, dataset_type='default',
             select_number=1, sampled_frames=5, repeats=6,
             **sa2va_default_dataset_configs),
        dict(type=Sa2VA03RefVOS, name='RefYTVOS',
             image_folder=RYTVOS_FRAMES, expression_file=RYTVOS_META,
             mask_file=RYTVOS_MASK, dataset_type='refytvos',
             select_number=1, sampled_frames=5, repeats=6,
             **sa2va_default_dataset_configs),
        dict(type=Sa2VA03RefVOS, name='RyvOS',
             image_folder=RYTVOS_FRAMES,
             expression_file=DATA_ROOT + 'ryvos/meta_expressions.json',
             mask_file=DATA_ROOT + 'ryvos/mask_dict.json',
             dataset_type='default', select_number=1, sampled_frames=5, repeats=3,
             **sa2va_default_dataset_configs),
        dict(type=LLaVADataset, name='LLaVA150K',
             data_path=DATA_ROOT + 'llava/llava_instruct_150k.json',
             image_folder=DATA_ROOT + 'llava/llava_images',
             skip_pure_text=True, repeats=0.3,
             **sa2va_qa_dataset_configs),
    ],
)
train_dataloader = dict(
    batch_size=batch_size, num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=LengthGroupedSampler, length_property='modality_length',
                 per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=sa2va_collect_fn),
)

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic', dtype='bfloat16',
)

param_scheduler = [
    dict(type=LinearLR, start_factor=1e-5, by_epoch=True, begin=0,
         end=warmup_ratio * max_epochs, convert_to_iter_based=True),
    dict(type=CosineAnnealingLR, eta_min=0.0, by_epoch=True,
         begin=warmup_ratio * max_epochs, end=max_epochs, convert_to_iter_based=True),
]

train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, save_optimizer=False, by_epoch=False,
                    interval=save_steps, max_keep_ckpts=save_total_limit),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(cudnn_benchmark=False,
               mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
               dist_cfg=dict(backend='nccl'))
visualizer = None
log_level = 'INFO'
load_from = None
resume = False

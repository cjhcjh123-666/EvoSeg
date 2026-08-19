"""EvoSeg no-target (abstain) SFT: teach the model to say "no".

Diagnosis: the multitask SFT (sa2va_qwen3_4b_multitask_sft) never included
no-target samples (build_pixel_llm_grefcoco.py skips them), so the model
hallucinates 100% on empty-target queries and RL alone cannot explore the
abstain path. This stage LoRA-fine-tunes the 4B-MultiTask on a faithfulness-
balanced mix: gRefCOCO train no-target abstain data (majority, x4) + a small
present mix (RefCOCO / gRefCOCO present, x1) to preserve segmentation. The
no-target samples carry a zero mask and no [SEG] token; Sa2VA.forward attaches
a zero embedding for them (empty-mask supervision) while the LLM learns the
refusal text. This requires the mixed-batch fix in Sa2VA.forward (an additive
branch for 0-[SEG] samples; present-only batches behave identically).

Continue from the multitask SFT checkpoint (LoRA unmerged, mmengine format):
  evo_artifacts/checkpoints/4b_multitask_iter50768_weights.pth
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
    Sa2VA07NoTargetDataset, Sa2VA08VideoFaithfulnessDataset,
)
from projects.sa2va.datasets.data_utils import ConcatDatasetSa2VA
from projects.sa2va.models.mllm.qwen3vl import Qwen3VL

# ---------------- settings ----------------
path = '/9950backfile/chenjiahui/evo_artifacts/models/Qwen3-VL-4B-Instruct'
pretrained_pth = '/9950backfile/chenjiahui/evo_artifacts/models/Sa2VA-Qwen3-VL-4B.pth'
continue_pth = '/9950backfile/chenjiahui/evo_artifacts/checkpoints/4b_videofaithful_iter6376_weights.pth'
DATA_ROOT = '/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/pixel_llm_data/'
COCO_IMG = '/9950backfile/chenjiahui/evo_artifacts/datasets/coco2014/train2014/'
RES_ROOT = DATA_ROOT + 'ref_seg/'

template = "qwen_chat"
prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 4096

batch_size = 2
accumulative_counts = 8
dataloader_num_workers = 8
max_epochs = 1
optim_type = AdamW
lr = 2e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1
warmup_ratio = 0.05
save_steps = 500
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
    frozen_sam2_decoder=True,          # preserve the trained seg head
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

train_dataset = dict(
    type=ConcatDatasetSa2VA, datasets=[
        dict(type=Sa2VA07NoTargetDataset, name='NoTarget',
             data_root=DATA_ROOT + 'notarget/',
             data_prefix=dict(img_path=COCO_IMG),
             ann_file='annotations.json', serialize_data=False,
             repeats=4, **sa2va_default_dataset_configs),
        dict(type=Sa2VA01RefSeg, name='RefCOCO',
             data_root=RES_ROOT + 'refcoco',
             data_prefix=dict(img_path='coco2014/train2014/'),
             ann_file='instances.json', split_file='refs(unc).p',
             split='train', num_classes_per_sample=5, repeats=1,
             serialize_data=False, **sa2va_default_dataset_configs),
        dict(type=Sa2VAFinetuneDataset, name='gRefCOCO',
             data_root=DATA_ROOT + 'grefcoco/',
             data_prefix=dict(img_path='images/train2014/'),
             ann_file='annotations.json', serialize_data=False,
             repeats=1, **sa2va_default_dataset_configs),
        dict(type=Sa2VA08VideoFaithfulnessDataset, name='VideoFaithfulness',
             manifest_file='/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/faithfulness_train.json',
             image_folder='/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/train/JPEGImages',
             ann_folder='/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/train/Annotations',
             sampled_frames=8, repeats=2, **sa2va_default_dataset_configs),    ],
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
load_from = continue_pth
resume = False

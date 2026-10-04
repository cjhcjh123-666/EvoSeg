"""FTG temporal pilot on public MeViS-v2 and Long-RVOS training data.

This deliberately excludes every historical faithfulness/no-target dataset and
loss.  It warm-starts the public Qwen3-VL-4B + SAM3 Sa2VA checkpoint, freezes the
native SAM3 weights, and trains LoRA, the existing text projection, and FTG's
identity-conditioned state cross-attention.
"""

from mmengine.config import read_base

with read_base():
    from ..sa2va_qwenvl3.sa2va_qwen3_4b_sam3 import *


ARTIFACT_ROOT = '/9950backfile/chenjiahui/evo_artifacts/'
path = ARTIFACT_ROOT + 'models/Qwen3-VL-4B-Instruct'
pretrained_pth = ARTIFACT_ROOT + 'models/Sa2VA-Qwen3-VL-4B-SAM3-training.pth'
VIDEO_ROOT = ARTIFACT_ROOT + 'datasets/'

batch_size = 1
accumulative_counts = 4  # effective batch 32 on eight GPUs
dataloader_num_workers = 4
max_epochs = 1
save_steps = 100
save_total_limit = 3

tokenizer['pretrained_model_name_or_path'] = path
extra_image_processor['target_length'] = 1008

model['training_bs'] = batch_size
model['grounding_img_size'] = 1008
model['pretrained_pth'] = pretrained_pth
model['frozen_sam2_decoder'] = True
model['grounding_variant'] = 'ftg'
model['use_existence_head'] = False
model['mllm']['model_path'] = path
model['grounding_encoder']['load_checkpoint'] = False

public_video_dataset = dict(
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    prompt_template=prompt_template,
    max_length=4096,
    arch_type='qwen',
    preprocessor=dict(
        type=Qwen3VLProcessor.from_pretrained,
        pretrained_model_name_or_path=path,
        trust_remote_code=True,
    ),
)

train_dataset = dict(
    type=ConcatDatasetSa2VA,
    datasets=[
        dict(
            type=Sa2VA03RefVOS,
            name='MeViS-v2',
            image_folder=VIDEO_ROOT + 'mevis_v2/train/JPEGImages',
            expression_file=VIDEO_ROOT + 'mevis_v2/train/meta_expressions_v2.json',
            mask_file=VIDEO_ROOT + 'mevis_v2/train/mask_dict.json',
            dataset_type='default',
            select_number=1,
            sampled_frames=5,
            repeats=1,
            **public_video_dataset,
        ),
        dict(
            type=Sa2VA03RefVOS,
            name='Long-RVOS',
            image_folder=VIDEO_ROOT + 'long_rvos/train/JPEGImages',
            annotation_folder=VIDEO_ROOT + 'long_rvos/train/Annotations',
            expression_file=VIDEO_ROOT + 'long_rvos/train/meta_expressions.json',
            mask_file=None,
            dataset_type='long_rvos',
            select_number=1,
            sampled_frames=5,
            repeats=1,
            **public_video_dataset,
        ),
    ],
)

train_dataloader['batch_size'] = batch_size
train_dataloader['num_workers'] = dataloader_num_workers
train_dataloader['dataset'] = train_dataset
train_dataloader['sampler']['per_device_batch_size'] = batch_size * accumulative_counts
optim_wrapper['accumulative_counts'] = accumulative_counts
train_cfg['max_epochs'] = max_epochs
default_hooks['checkpoint']['interval'] = save_steps
default_hooks['checkpoint']['max_keep_ckpts'] = save_total_limit
randomness = dict(seed=42, deterministic=False)

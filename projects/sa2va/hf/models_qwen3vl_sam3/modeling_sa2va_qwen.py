import torch
from torch import nn
from transformers import (AutoModel, GenerationConfig, Qwen3VLForConditionalGeneration,
                          Qwen2ForCausalLM)
from transformers.modeling_utils import PreTrainedModel

from .configuration_sa2va_chat import Sa2VAChatConfigQwen

from .sam3 import SAM3
from .ftg_interface import FactorizedPromptTokens


def select_temporal_indices(num_frames, budget=5, strategy='first'):
    """Choose the frames that supply VLM evidence and segmentation prompts."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if budget <= 0:
        raise ValueError("temporal budget must be positive")
    count = min(num_frames, budget)
    if strategy == 'first':
        return list(range(count))
    if strategy == 'uniform':
        return np.linspace(0, num_frames - 1, count, dtype=int).tolist()
    raise ValueError(f"unknown temporal sampling strategy: {strategy}")


def resize_mask_logits_to_bool_cpu(
    mask_logits,
    size,
    chunk_size=8,
):
    """Resize independent video masks with bounded accelerator memory.

    Long-RVOS contains very long, high-resolution clips. Resizing the complete
    ``[T, 1, h, w]`` tensor at once can require tens of GiB even though each
    frame is independent. Threshold and move every chunk to CPU immediately so
    peak CUDA memory is constant in ``T``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunks = []
    for start in range(0, len(mask_logits), chunk_size):
        resized = F.interpolate(
            mask_logits[start:start + chunk_size],
            size=size,
            mode='bilinear',
            align_corners=False,
        )
        chunks.append((resized[:, 0].sigmoid() > 0.5).cpu())
    if not chunks:
        return np.zeros((0, *size), dtype=bool)
    return torch.cat(chunks, dim=0).numpy()

import numpy as np
from torchvision.transforms.functional import to_pil_image

import torch.nn.functional as F

from qwen_vl_utils import process_vision_info



class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

class Sa2VAChatModelQwen(PreTrainedModel):
    config_class = Sa2VAChatConfigQwen
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['Qwen3VisionTransformerPretrainedModel', 'Qwen3VLDecoderLayer', 'SAM3']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True



    def __init__(self, config: Sa2VAChatConfigQwen, model=None, use_flash_attn=True):
        super().__init__(config)
        self.extra_image_processor = DirectResize(target_length=1008, )

        self.min_pixels = 512 * 28 * 28
        self.max_pixels = 2048 * 28 * 28

        self.torch_dtype = torch.bfloat16

        if model is not None:
            self.model=model
        else:
            self.model = Qwen3VLForConditionalGeneration(config)

        llm_hidden_size = config.text_config.hidden_size

        self.grounding_encoder = SAM3()
        out_dim = self.grounding_encoder.hidden_dim
        in_dim = llm_hidden_size
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )
        self.grounding_variant = getattr(
            config, 'grounding_variant', 'identity_memory'
        )
        self.temporal_sampling = getattr(config, 'temporal_sampling', 'first')
        self.temporal_budget = getattr(config, 'temporal_budget', 5)
        self.identity_temporal_sampling = (
            getattr(config, 'identity_temporal_sampling', None)
            or self.temporal_sampling
        )
        self.state_temporal_sampling = (
            getattr(config, 'state_temporal_sampling', None)
            or self.temporal_sampling
        )
        self.factorized_grounding = (
            None if self.grounding_variant == 'identity_memory'
            else FactorizedPromptTokens(
                out_dim,
                max_residual_ratio=getattr(
                    config, 'grounding_residual_ratio', 0.02),
            )
        )

    @property
    def lm_head(self):
        return self.model.lm_head

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def predict_forward(
            self,
            image=None,
            video=None,
            text=None,
            past_text='',
            mask_prompts=None,
            tokenizer=None,
            processor=None,
    ):
        assert processor is not None
        self.processor = processor
        
        self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids('[SEG]')

        text = text.replace('<image>', "")
        self.last_grounding_gates = []

        if image is None and video is None and '<image>' not in past_text:
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": past_text + text},
                    ],
                }
            ]

            # Preparation for inference
            processsed_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            mm_inputs = self.processor(
                text=[processsed_text],
                images=None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            mm_inputs = mm_inputs.to(self.device)

        else:
            input_dict = {}
            if video is not None:
                pixel_values = []
                extra_pixel_values = []
                images = []
                content = []
                ori_image_size = video[0].size
                identity_frame_indices = select_temporal_indices(
                    len(video),
                    self.temporal_budget,
                    self.identity_temporal_sampling,
                )
                # A monolithic identity prompt keeps the legacy behavior. FTG
                # can instead anchor identity on stable frames while deriving
                # frame-dependent state prompts from a different temporal view.
                if self.factorized_grounding is None:
                    conditioning_frame_indices = identity_frame_indices
                else:
                    conditioning_frame_indices = select_temporal_indices(
                        len(video),
                        self.temporal_budget,
                        self.state_temporal_sampling,
                    )
                identity_frame_set = set(identity_frame_indices)
                for frame_idx, frame_image in enumerate(video):
                    # assert ori_image_size == frame_image.size
                    g_image = np.array(frame_image)  # for grounding
                    g_image = self.extra_image_processor.apply_image(g_image)
                    g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    extra_pixel_values.append(g_image)
                    if frame_idx in identity_frame_set:
                        content.append({"type": "image", "image": frame_image},)


                content.append({"type": "text", "text": text})
                messages = [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]

                # Preparation for inference
                processsed_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                image_inputs, video_inputs = process_vision_info(messages)
                mm_inputs = self.processor(
                    text=[processsed_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels
                )
                mm_inputs = mm_inputs.to(self.device)

                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                ]).to(self.torch_dtype)

                num_frames = len(conditioning_frame_indices)

            else:
                ori_image_size = image.size
                
                # prepare grounding images
                g_image = np.array(image)  # for grounding
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(self.torch_dtype)
                extra_pixel_values = [g_pixel_values]
                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                ]).to(self.torch_dtype)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {"type": "text", "text": text},
                        ],
                    }
                ]

                # Preparation for inference
                processsed_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                image_inputs, video_inputs = process_vision_info(messages)
                mm_inputs = self.processor(
                    text=[processsed_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels
                )
                mm_inputs = mm_inputs.to(self.device)

                num_frames = 1
                conditioning_frame_indices = [0]
                identity_frame_indices = [0]
            
            input_dict['g_pixel_values'] = g_pixel_values
            ret_masks = []

        generate_output = self.model.generate(
            **mm_inputs,
            max_new_tokens=2048,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        generate_output_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(mm_inputs.input_ids, generate_output.sequences)
        ]

        predict = self.processor.batch_decode(generate_output_trimmed, skip_special_tokens=False)[0].strip()

        if image is None and video is None and '<image>' not in past_text:
            return {'prediction': predict, 'prediction_masks': ret_masks, }

        # if have seg result, find the seg hidden states
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=self.seg_token_idx
        )
        all_seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

        for seg_hidden_states in all_seg_hidden_states:
            seg_hidden_states = seg_hidden_states.unsqueeze(0)
            g_pixel_values = input_dict['g_pixel_values']
            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            if self.factorized_grounding is None:
                frame_prompts = [seg_hidden_states] * num_frames
            else:
                with torch.no_grad(), torch.autocast(
                        device_type='cuda', dtype=torch.bfloat16):
                    spatial_features = []
                    for frame_idx in conditioning_frame_indices:
                        features = self.grounding_encoder.sam2_model._get_image_feature(
                            sam_states, frame_idx, batch_size=1
                        )
                        spatial_features.append(
                            features[2][-1].permute(1, 0, 2))
                    spatial_features = torch.cat(spatial_features, dim=0)
                    frame_identities = seg_hidden_states.repeat(num_frames, 1)
                    tokens, grounding_gate = self.factorized_grounding(
                        frame_identities,
                        spatial_features,
                        variant=self.grounding_variant,
                        group_shape=(num_frames, 1),
                    )
                    self.last_grounding_gates.append(
                        grounding_gate.detach().float().cpu()
                    )
                    frame_prompts = list(tokens.split(1, dim=0))
            pred_masks = self.grounding_encoder.language_embd_inference(
                sam_states,
                frame_prompts,
                frame_indices=conditioning_frame_indices,
            )
            w, h = ori_image_size
            masks = resize_mask_logits_to_bool_cpu(pred_masks, size=(h, w))
            ret_masks.append(masks)

        return {'prediction': predict, 'prediction_masks': ret_masks,}

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]

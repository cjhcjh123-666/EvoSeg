import torch
from torch import nn
from transformers import (AutoModel, GenerationConfig, Qwen3VLForConditionalGeneration,
                          Qwen2ForCausalLM)
from transformers.modeling_utils import PreTrainedModel

from .configuration_sa2va_chat import Sa2VAChatConfigQwen

from .sam2 import SAM2

class ETHead(nn.Module):
    """B+ fidelity evaluator (e_t): bidirectional GRU over per-frame
    [VLM hidden state (the model's own perception of this frame),
     mask-region feat, mask geometry, frame-0 proto, lang] -> e_t."""
    def __init__(self, vlm_dim=2560, feat_dim=256, mask_dim=256, lang_dim=256,
                 hidden=512, n_layers=1):
        super().__init__()
        self.vlm_proj = nn.Linear(vlm_dim, 256)
        self.fc_in = nn.Sequential(
            nn.Linear(256 + mask_dim * 2 + 3 + lang_dim, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=n_layers,
                          batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, vlm, mask_cond, geom, anchor, lang):
        B, T, C = vlm.shape
        v = self.vlm_proj(vlm)                                   # [B,T,256]
        x = torch.cat([v, mask_cond, geom, anchor.expand(B, T, -1),
                       lang.expand(B, T, -1)], dim=-1)          # [B,T,256+256+3+256+256]
        x = self.fc_in(x)                                       # [B,T,H]
        out, _ = self.gru(x)                                    # [B,T,H]
        return self.head(out).squeeze(-1)                       # [B,T]


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
    _no_split_modules = ['Qwen3VisionTransformerPretrainedModel', 'Qwen3VLDecoderLayer', 'SAM2']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True



    def __init__(self, config: Sa2VAChatConfigQwen, model=None, use_flash_attn=True):
        super().__init__(config)
        self.extra_image_processor = DirectResize(target_length=1024, )

        self.min_pixels = 512 * 28 * 28
        self.max_pixels = 2048 * 28 * 28

        self.torch_dtype = torch.bfloat16

        if model is not None:
            self.model=model
        else:
            self.model = Qwen3VLForConditionalGeneration(config)

        llm_hidden_size = config.text_config.hidden_size

        self.grounding_encoder = SAM2()
        out_dim = self.grounding_encoder.hidden_dim
        in_dim = llm_hidden_size
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )
        # [TEG] temporal existence gate head (mirrors the mmengine Sa2VAModel)
        self.existence_head = nn.Sequential(
            nn.Linear(out_dim + out_dim, out_dim), nn.ReLU(inplace=True),
            nn.Linear(out_dim, 1)
        )
        # [TEG-v2] lightweight temporal existence head (GRU over frames);
        # loaded from temporal_existence_head.pt next to the model files.
        self.temporal_existence_head = None
        try:
            import os as _os
            # NOTE: under trust_remote_code transformers re-executes this file
            # from ~/.cache, so __file__ points at the cache dir. The real model
            # dir is config._name_or_path -- look there first.
            _model_dir = getattr(config, '_name_or_path', None)
            _head_cands = []
            if _model_dir:
                _head_cands.append(_os.path.join(_model_dir, 'temporal_existence_head.pt'))
            _head_cands.append(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                             'temporal_existence_head.pt'))
            _head_path = next((c for c in _head_cands if _os.path.exists(c)), None)
            if _head_path is not None:
                _ckpt = torch.load(_head_path, map_location='cpu')
                _cfg = _ckpt.get('config', {})
                _h = ETHead(
                    vlm_dim=_cfg.get('vlm_dim', 2560),
                    feat_dim=_cfg.get('feat_dim', out_dim),
                    mask_dim=_cfg.get('mask_dim', out_dim),
                    lang_dim=_cfg.get('lang_dim', out_dim),
                    hidden=_cfg.get('hidden', 512),
                    n_layers=_cfg.get('n_layers', 1),
                )
                _h.load_state_dict(_ckpt['state_dict'])
                # keep the head in float32 (it was trained in float32);
                # inference casts inputs to .float() so dtypes must match.
                self.temporal_existence_head = _h.float()
                print(f'[Sa2VA] loaded temporal existence head ({_head_path})',
                      flush=True)
        except Exception as _e:
            print(f'[Sa2VA] temporal existence head load skipped: {_e}', flush=True)
            self.temporal_existence_head = None

    def _apply(self, *args, **kwargs):
        # keep the temporal existence head in float32: GRU in bf16 is
        # numerically unstable (NaN logits -> gate collapses to all-zero).
        ret = super()._apply(*args, **kwargs)
        if getattr(self, 'temporal_existence_head', None) is not None:
            self.temporal_existence_head.float()
        return ret

    def load_temporal_head(self):
        """(Re)load the GRU temporal existence head from
        temporal_existence_head.pt next to the model dir. Call AFTER
        AutoModel.from_pretrained, which may re-initialize modules that are
        not in the safetensors checkpoint."""
        import os as _os
        _model_dir = getattr(self.config, '_name_or_path', None) or _os.path.dirname(
            _os.path.abspath(__file__))
        _p = _os.path.join(_model_dir, 'temporal_existence_head.pt')
        if not _os.path.exists(_p):
            print(f'[Sa2VA] no temporal_existence_head.pt at {_p}', flush=True)
            return self
        try:
            _ck = torch.load(_p, map_location='cpu')
            _cfg = _ck.get('config', {})
            _h = ETHead(vlm_dim=_cfg.get('vlm_dim', 2560),
                        feat_dim=_cfg.get('feat_dim', self.grounding_encoder.hidden_dim),
                        mask_dim=_cfg.get('mask_dim', self.grounding_encoder.hidden_dim),
                        lang_dim=_cfg.get('lang_dim', self.grounding_encoder.hidden_dim),
                        hidden=_cfg.get('hidden', 512), n_layers=_cfg.get('n_layers', 1))
            _h.load_state_dict(_ck['state_dict'])
            _h = _h.float().to(self.device)
            self.temporal_existence_head = _h
            self.temporal_gate_thr = float(_cfg.get('thr', 0.5))
            print(f'[Sa2VA] load_temporal_head -> {_p} (float32, '
                  f'{sum(p.numel() for p in _h.parameters())} params, '
                  f'thr={self.temporal_gate_thr})', flush=True)
        except Exception as _e:
            print(f'[Sa2VA] load_temporal_head skipped: {_e}', flush=True)
            self.temporal_existence_head = None
        return self

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
            vlm_all_frames=False,
    ):
        self._vlm_feat = None  # set in B+ mode (per-frame VLM hidden states)
        assert processor is not None
        self.processor = processor
        
        self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids('[SEG]')

        text = text.replace('<image>', "")

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

            ret_masks = []
        else:
            input_dict = {}
            if video is not None:
                pixel_values = []
                extra_pixel_values = []
                images = []
                content = []
                ori_image_size = video[0].size
                for frame_idx, frame_image in enumerate(video):
                    # assert ori_image_size == frame_image.size
                    g_image = np.array(frame_image)  # for grounding
                    g_image = self.extra_image_processor.apply_image(g_image)
                    g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    extra_pixel_values.append(g_image)
                    if (vlm_all_frames) or (frame_idx < 5):
                        content.append({"type": "image", "image": frame_image})


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

                num_frames = min(5, len(video))

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
        # [B+] per-frame VLM hidden states (the model perceives every frame)
        if vlm_all_frames and video is not None:
            try:
                _vs = self.processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
                _ve = self.processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
                _ids = mm_inputs.input_ids[0].tolist()
                _ranges = []
                _i = 0
                while _i < len(_ids):
                    if _ids[_i] == _vs:
                        _j = _i + 1
                        while _j < len(_ids) and _ids[_j] != _ve:
                            _j += 1
                        _ranges.append((_i + 1, _j))
                        _i = _j + 1
                    else:
                        _i += 1
                _hs0 = hidden_states[0][-1][0]          # [seq, C]
                _pf = [_hs0[s:e].mean(dim=0) for s, e in _ranges]
                if len(_pf) == len(video):
                    self._vlm_feat = torch.stack(_pf).float()   # [T, 2560]
            except Exception as _e:
                print(f'[Sa2VA] B+ per-frame feat extraction skipped: {_e}', flush=True)
        all_seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

        for seg_hidden_states in all_seg_hidden_states:
            seg_hidden_states = seg_hidden_states.unsqueeze(0)
            g_pixel_values = input_dict['g_pixel_values']
            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            pred_masks = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * num_frames)
            w, h = ori_image_size
            masks = F.interpolate(pred_masks, size=(h, w), mode='bilinear', align_corners=False)
            masks = masks[:, 0]
            masks = masks.sigmoid() > 0.5
            # [TEG] gate by the temporal existence head e_t (per frame).
            # Single SAM2 propagation pass (fast, no re-propagation / no OOM);
            # e_t only decides whether to keep or zero each frame's mask.
            if video is not None and getattr(self, 'temporal_gate_enabled', True) and (
                    getattr(self, 'temporal_existence_head', None) is not None
                    or hasattr(self, 'existence_head')):
                feats = self.grounding_encoder.sam2_model.forward_image(
                    g_pixel_values.to(self.device))
                _, vision_feats, _, _ = self.grounding_encoder.sam2_model._prepare_backbone_features(feats)
                vis_feat = vision_feats[-1]                                # [HW, T, C]
                feat_pool = vis_feat.mean(dim=0)                        # [T, C]
                lang = seg_hidden_states.squeeze(0)                       # [C]
                if getattr(self, 'temporal_existence_head', None) is not None:
                    with torch.no_grad():
                        _vlm = getattr(self, '_vlm_feat', None)
                        # mask-region features: pool SAM2 features under the
                        # propagated mask (fidelity evidence for e_t)
                        H = int(round((vis_feat.shape[0]) ** 0.5))
                        C = vis_feat.shape[2]
                        T_f = vis_feat.shape[1]
                        vf_2d = vis_feat.permute(1, 2, 0).view(
                            T_f, C, H, H).float()                        # [T,C,H,H]
                        mp = pred_masks.sigmoid()                        # [T,1,Hl,Wl]
                        mpr = F.interpolate(mp, size=(H, H), mode='bilinear',
                                            align_corners=False).squeeze(1)  # [T,H,H]
                        _num = (vf_2d * mpr.unsqueeze(1)).sum(dim=(-2, -1))
                        _den = mpr.sum(dim=(-2, -1)).clamp(min=1e-5)
                        mask_cond = _num / _den.unsqueeze(-1)            # [T,C]
                        # AutoModel.from_pretrained(torch_dtype=...) casts the
                        # whole model (incl. the head) to that dtype, so cast
                        # the inputs to the head's current dtype.
                        # mask geometry (area / centroid) -- 'SAM2 lost the
                        # track' signature: area explodes when target leaves
                        _mp = pred_masks.sigmoid()               # [T,1,H,W]
                        _B, _H, _W = _mp.shape[0], _mp.shape[2], _mp.shape[3]
                        _mpf = _mp.reshape(_B, _H * _W)
                        _mass = _mpf.sum(dim=1)
                        _yy, _xx = torch.meshgrid(
                            torch.arange(_H).float().to(_mp.device),
                            torch.arange(_W).float().to(_mp.device), indexing='ij')
                        _xx = _xx.reshape(-1); _yy = _yy.reshape(-1)
                        _s = _mass.clamp(min=1e-5)
                        _gx = (_mpf * _xx.unsqueeze(0)).sum(dim=1) / _s
                        _gy = (_mpf * _yy.unsqueeze(0)).sum(dim=1) / _s
                        geom = torch.stack(
                            [_mass / (_H * _W), _gx / _W, _gy / _H], dim=1)  # [T,3]
                        _hdtype = next(
                            self.temporal_existence_head.parameters()).dtype
                        if _vlm is not None and _vlm.shape[0] == masks.shape[0]:
                            e_logit = self.temporal_existence_head(
                                _vlm.unsqueeze(0).to(_hdtype),
                                mask_cond.unsqueeze(0).to(_hdtype),
                                geom.unsqueeze(0).to(_hdtype),
                                mask_cond[0:1].unsqueeze(0).to(_hdtype),
                                lang.unsqueeze(0).unsqueeze(0).to(_hdtype))  # [1,T]
                        else:
                            # fallback: zero VLM features
                            e_logit = self.temporal_existence_head(
                                torch.zeros(1, masks.shape[0], 2560,
                                            device=feat_pool.device).to(_hdtype),
                                mask_cond.unsqueeze(0).to(_hdtype),
                                geom.unsqueeze(0).to(_hdtype),
                                mask_cond[0:1].unsqueeze(0).to(_hdtype),
                                lang.unsqueeze(0).unsqueeze(0).to(_hdtype))  # [1,T]
                    _thr = getattr(self, 'temporal_gate_thr', 0.5)
                    e = (e_logit.sigmoid() > _thr).squeeze(0)        # [T]
                else:
                    feat_pool = feat_pool.to(lang.dtype)
                    e_logit = self.existence_head(torch.cat(
                        [feat_pool, lang.unsqueeze(0).expand(
                            feat_pool.shape[0], -1)], dim=-1))
                    e = (e_logit.sigmoid() > 0.5).squeeze(-1)            # [T]
                e = e[:masks.shape[0]]
                masks = masks * e.unsqueeze(-1).unsqueeze(-1)
            masks = masks.cpu().numpy()
            ret_masks.append(masks)

        return {'prediction': predict, 'prediction_masks': ret_masks,}

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]

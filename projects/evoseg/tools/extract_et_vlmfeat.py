"""B+ feature pass: per-frame VLM hidden states (the model perceives every frame).

For each [SEG] case, feed ALL T frames to the VLM (not just 5), generate the
[SEG] token, and pool the LLM hidden states over each frame's vision tokens:
  vlm_feat [T, 2560]  (per-frame VLM perception)
  lang     [1, 256]   (text_hidden_fcs([SEG]))
Saves to vlmfeat_rank{r}.pt as list of {vid, query[:40], vlm_feat, lang}.

Usage (4 GPUs):
  torchrun --nproc_per_node=4 ... extract_et_vlmfeat.py
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL = '/9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-TEG'
JPEGROOT = ('/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/'
            'extracted/train/JPEGImages')
MAN = ('/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/'
       'faithfulness_train.json')
FEATDIR = '/9950backfile/chenjiahui/evo_artifacts/data/et_features'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat-dir', default=FEATDIR)
    ap.add_argument('--outdir', default=FEATDIR)
    ap.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = ap.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(rank)
    os.makedirs(args.outdir, exist_ok=True)

    cases = json.load(open(MAN))['cases']
    by_vid = {}
    for c in cases:
        by_vid.setdefault(c['video_id'], c)

    files = sorted([f for f in os.listdir(args.feat_dir) if f.startswith('et_features_rank')])
    my_files = [f for i, f in enumerate(files) if i % world == rank]

    model = AutoModel.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    vs_id = tok.convert_tokens_to_ids('<|vision_start|>')
    ve_id = tok.convert_tokens_to_ids('<|vision_end|>')
    seg_id = tok.convert_tokens_to_ids('[SEG]')

    out_path = os.path.join(args.outdir, f'vlmfeat_rank{rank}.pt')
    res = []
    if os.path.exists(out_path):
        res = torch.load(out_path, map_location='cpu')
        print(f'[rank{rank}] resume {len(res)}', flush=True)
    done = len(res)
    t0 = time.time(); n_total = 0
    for fi, f in enumerate(my_files):
        data = torch.load(os.path.join(args.feat_dir, f), map_location='cpu')
        for ri, r in enumerate(data):
            n_total += 1
            if n_total - 1 < done:
                continue
            if not r.get('seg'):
                continue
            vid = r['vid']; query = r['query']
            c = by_vid.get(vid)
            if c is None:
                continue
            frames = [Image.open(os.path.join(JPEGROOT, vid, fr + '.jpg')).convert('RGB')
                      for fr in c['frames']]
            T = len(frames)
            content = [{'type': 'image', 'image': fr} for fr in frames]
            content.append({'type': 'text', 'text': query})
            messages = [{'role': 'user', 'content': content}]
            ps = proc.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
            ii, _ = process_vision_info(messages)
            mm = proc(text=[ps], images=ii, videos=None, padding=True,
                      return_tensors='pt', min_pixels=model.min_pixels,
                      max_pixels=model.max_pixels).to(model.device)
            ids = mm.input_ids[0].tolist()
            # vision ranges per frame
            ranges = []
            i = 0
            while i < len(ids):
                if ids[i] == vs_id:
                    j = i + 1
                    while j < len(ids) and ids[j] != ve_id:
                        j += 1
                    ranges.append((i + 1, j))
                    i = j + 1
                else:
                    i += 1
            with torch.no_grad():
                out = model.model.generate(**mm, max_new_tokens=32, do_sample=False,
                                           output_hidden_states=True,
                                           return_dict_in_generate=True)
            hs0 = out.hidden_states[0][-1][0]          # [seq, C]
            seq = out.sequences[0][:-1]
            seg_hs_list = []
            sm = seq == seg_id
            all_hs = torch.cat([h[-1][0] for h in out.hidden_states], dim=0)
            shs = all_hs[-len(sm):][sm]
            if shs.shape[0] == 0:
                continue
            lang = model.text_hidden_fcs(shs)[-1].unsqueeze(0)   # [1,256]
            # per-frame pooled VLM features
            feats = []
            for s, e in ranges:
                seg_hs = hs0[s:e]
                feats.append(seg_hs.mean(dim=0))        # [C]
            vlm_feat = torch.stack(feats)               # [T, C]
            res.append({'vid': vid, 'query': query[:40],
                        'vlm_feat': vlm_feat.float().cpu(),
                        'lang': lang.float().cpu()})
            if n_total % 25 == 0:
                el = time.time() - t0
                print(f'[rank{rank}] {n_total} {el/n_total:.2f}s/case', flush=True)
            if n_total % 200 == 0:
                torch.save(res, out_path)
        torch.save(res, out_path)
    print(f'[rank{rank}] DONE {len(res)} -> {out_path}', flush=True)


if __name__ == '__main__':
    main()

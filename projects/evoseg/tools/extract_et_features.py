"""Extract per-frame SAM2 pooled features + [SEG] lang vectors for the
EvoSeg temporal existence head (e_t).

For every video faithfulness training case where the model emits a [SEG]
token (i.e. a mask is actually propagated and therefore needs gating), we
dump:

  feat      : [T, 256]  mean-pooled SAM2 backbone features per frame
  lang      : [1, 256]  text_hidden_fcs([SEG] hidden state)  (last [SEG])
  presence  : [T] bool  expected mask presence per frame (from GT masks)
  cat / vid / query     metadata

Cases where the model refuses (no [SEG]) are skipped: no mask is propagated
so there is nothing to gate.

Speedups:
  * LOCAL_RANK env (torchrun >=2.4 does not pass --local-rank)
  * prefetch next case's frames while the GPU works on the current one
  * per-video per-frame SAM2 feature cache (manifest is grouped by video)
  * single preprocessing pass; chunked SAM2 image-encoder forward

Usage (4 GPUs):
  torchrun --nproc_per_node=4 --master_port=29601 \\
      projects/evoseg/tools/extract_et_features.py \\
      --outdir /9950backfile/chenjiahui/evo_artifacts/data/et_features
"""
import argparse
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
CH = 4  # SAM2 image-encoder chunk size


def load_frames(c):
    vid = c['video_id']
    return [Image.open(os.path.join(JPEGROOT, vid, f + '.jpg')).convert('RGB')
            for f in c['frames']]


def sam2_feat_cache(model, cache, pre):
    """pre: {(vid, frame_idx): resized uint8 tensor}; returns [T, 256]."""
    out, todo = [], []
    for k, t in pre.items():
        if k in cache:
            out.append(cache[k])
        else:
            todo.append((k, t))
    if todo:
        gs = torch.stack([model.grounding_encoder.preprocess_image(t)
                          for _, t in todo]).to(torch.bfloat16)
        with torch.no_grad():
            for s in range(0, len(todo), CH):
                feats = model.grounding_encoder.sam2_model.forward_image(
                    gs[s:s + CH].cuda())
                _, vf, _, _ = model.grounding_encoder.sam2_model._prepare_backbone_features(feats)
                pool = vf[-1].mean(dim=0).float().cpu()   # [chunk, 256]
                for j, (k, _t) in enumerate(todo[s:s + CH]):
                    cache[k] = pool[j]
                    out.append(cache[k])
    return torch.stack(out)


def extract_one(model, tok, proc, c, cache, frames):
    vid = c['video_id']
    T = len(frames)
    # single preprocessing pass
    pre = {}
    for fi, fr in zip(c['frames'], frames):
        g = model.extra_image_processor.apply_image(np.array(fr))
        pre[(vid, fi)] = torch.from_numpy(g).permute(2, 0, 1).contiguous()

    content = [{'type': 'image', 'image': fr} for fr in frames[:5]]
    content.append({'type': 'text', 'text': c['query']})
    messages = [{'role': 'user', 'content': content}]
    ps = proc.apply_chat_template(messages, tokenize=False,
                                  add_generation_prompt=True)
    ii, _vi = process_vision_info(messages)
    mm = proc(text=[ps], images=ii, videos=None, padding=True,
              return_tensors='pt', min_pixels=model.min_pixels,
              max_pixels=model.max_pixels).to(model.device)
    with torch.no_grad():
        out = model.model.generate(**mm, max_new_tokens=32, do_sample=False,
                                   output_hidden_states=True,
                                   return_dict_in_generate=True)
    trimmed = out.sequences[0][len(mm.input_ids[0]):]
    text = proc.batch_decode(trimmed.unsqueeze(0),
                             skip_special_tokens=False)[0].strip()
    hs = out.hidden_states
    lhs = torch.cat([h[-1][0] for h in hs], dim=0)
    seq = out.sequences[0][:-1]
    seg_id = tok.convert_tokens_to_ids('[SEG]')
    sm = seq == seg_id
    seg_hs = lhs[-len(sm):][sm]
    if seg_hs.shape[0] == 0:
        return {'seg': False, 'text': text}
    lang = model.text_hidden_fcs(seg_hs)[-1].unsqueeze(0)   # [1, 256]
    feat_pool = sam2_feat_cache(model, cache, pre)          # [T, 256]
    torch.cuda.empty_cache()
    return {'seg': True, 'text': text,
            'lang': lang.float().cpu(), 'feat': feat_pool.float(),
            'presence': [bool(x) for x in c['presence']],
            'cat': c['category'], 'vid': vid, 'query': c['query']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='/tmp/et_features')
    ap.add_argument('--max-cases', type=int, default=None)
    ap.add_argument('--ckpt-every', type=int, default=200)
    ap.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = ap.parse_args()

    rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(rank)
    os.makedirs(args.outdir, exist_ok=True)

    cases = json.load(open(MAN))['cases']
    if args.max_cases:
        cases = cases[:args.max_cases]
    my_cases = [c for i, c in enumerate(cases) if i % world == rank]

    model = AutoModel.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    out_path = os.path.join(args.outdir, f'et_features_rank{rank}.pt')
    done = 0
    res = []
    if os.path.exists(out_path):
        res = torch.load(out_path, map_location='cpu')
        done = len(res)
        print(f'[rank{rank}] resume from {done} saved cases', flush=True)

    cache = {}  # (vid, frame_idx) -> pooled SAM2 feature
    t0 = time.time()
    n_seg = 0
    next_frames = None
    for i, c in enumerate(my_cases):
        if i < done:
            n_seg += res[i]['seg']
            continue
        if next_frames is None:
            next_frames = load_frames(c)
        frames = next_frames
        if i + 1 < len(my_cases):
            next_frames = load_frames(my_cases[i + 1])
        else:
            next_frames = None
        r = extract_one(model, tok, proc, c, cache, frames)
        res.append(r)
        n_seg += r['seg']
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f'[rank{rank}] {i+1}/{len(my_cases)} seg={n_seg} '
                  f'{el/(i+1):.2f}s/case eta={(len(my_cases)-i-1)*el/(i+1)/60:.0f}min',
                  flush=True)
        if (i + 1) % args.ckpt_every == 0:
            torch.save(res, out_path)
    torch.save(res, out_path)
    print(f'[rank{rank}] DONE {len(res)} cases ({n_seg} seg) -> {out_path}',
          flush=True)


if __name__ == '__main__':
    main()

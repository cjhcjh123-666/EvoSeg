"""Video faithfulness eval for EvoSeg faithful referring segmentation.

Runs a Sa2VA model on the video faithfulness benchmark manifest produced by
tools/build_video_faithfulness_manifest.py and measures per-frame mask-presence
agreement with the expected `presence` signal. This directly tests the
faithfulness dimensions general RVOS ignores:

  temporal_absence    : masks must stop after the target disappears / start
                        only when it appears (SAM2 memory hallucination).
  global_absence      : no mask on any frame (query refers to an absent object).
  counterfactual_swap : no mask (query was adversarially modified).
  identity_swap       : only the referenced instance is segmented; the mask
                        must not jump to a lookalike (presence still from GT).

Metrics (per category and overall):
  absent_halluc : fraction of expected-absent frames with a non-empty pred mask
  present_miss  : fraction of expected-present frames with an empty pred mask
  frame_acc     : per-frame mask-presence agreement
  n_queries     : number of test queries

Usage (8 GPUs):
  torchrun --nproc_per_node=8 projects/evoseg/eval/eval_video_faithfulness.py \
      <model_path> --save <out.json> \
      [--manifest <faithfulness_valid.json>] [--max-cases 200]
"""
import argparse
import datetime
import json
import os

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MANIFEST = ('/9950backfile/chenjiahui/evo_artifacts/datasets/'
            'ref_youtube_vos/faithfulness_valid.json')
JPEGROOT = ('/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/'
            'extracted/valid/JPEGImages')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('model_path')
    p.add_argument('--save', default=None)
    p.add_argument('--manifest', default=MANIFEST)
    p.add_argument('--max-cases', type=int, default=None)
    p.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    import torch.distributed as dist
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=120))
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)

    manifest = json.load(open(args.manifest))
    cases = manifest['cases']
    if args.max_cases:
        cases = cases[:args.max_cases]
    my_cases = [c for i, c in enumerate(cases) if i % world == rank]
    if rank == 0:
        print(f'cases={len(cases)} per_rank={len(my_cases)}', flush=True)

    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    local = []
    for c in my_cases:
        vid = c['video_id']
        frames = [Image.open(os.path.join(JPEGROOT, vid, f + '.jpg')).convert('RGB')
                  for f in c['frames']]
        text = f'<image>\n Please segment {c["query"]} in this video.'
        with torch.no_grad():
            out = model.predict_forward(
                video=frames, text=text, tokenizer=tokenizer, processor=processor)
        pred_masks = out['prediction_masks']
        pred = pred_masks[0] if len(pred_masks) > 0 else None
        n_frames = len(frames)
        pred_presence = [False] * n_frames
        if pred is not None:
            for t in range(min(n_frames, len(pred))):
                pred_presence[t] = bool((pred[t] > 0).sum())
        expected = c['presence'][:n_frames]
        local.append({
            'category': c['category'],
            'video_id': vid,
            'query': c['query'][:120],
            'n_frames': n_frames,
            'n_pred_frames': int(sum(pred_presence)),
            'n_expected_present': int(sum(expected)),
            'pred_presence': pred_presence,
            'expected_presence': expected,
            'pred_text': out['prediction'][:200],
        })

    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    all_res = []
    for g in gathered:
        all_res.extend(g)

    if rank == 0:
        cats = ['temporal_absence', 'global_absence', 'counterfactual_swap', 'identity_swap']
        summary = {}
        for cat in cats + ['overall']:
            res = [r for r in all_res if cat == 'overall' or r['category'] == cat]
            if not res:
                continue
            absent_den = sum(r['n_frames'] - r['n_expected_present'] for r in res)
            present_den = sum(r['n_expected_present'] for r in res)
            absent_h = sum(
                1 for r in res
                for t in range(r['n_frames'])
                if not r['expected_presence'][t] and r['pred_presence'][t])
            present_m = sum(
                1 for r in res
                for t in range(r['n_frames'])
                if r['expected_presence'][t] and not r['pred_presence'][t])
            n_correct = sum(
                1 for r in res
                for t in range(r['n_frames'])
                if r['pred_presence'][t] == r['expected_presence'][t])
            n_total = sum(r['n_frames'] for r in res)
            summary[cat] = {
                'n_queries': len(res),
                'n_frames': n_total,
                'absent_halluc_rate': round(absent_h / absent_den, 4) if absent_den else None,
                'present_miss_rate': round(present_m / present_den, 4) if present_den else None,
                'frame_acc': round(n_correct / n_total, 4) if n_total else None,
            }
        print('=' * 60)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if args.save:
            json.dump({'summary': summary, 'results': all_res},
                      open(args.save, 'w'), ensure_ascii=False)
            print('saved to', args.save)


if __name__ == '__main__':
    main()

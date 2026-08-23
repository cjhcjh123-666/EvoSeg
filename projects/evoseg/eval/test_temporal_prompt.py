"""Test: does adding temporal context to the per-frame image prompt help the
model refuse temporal-absence frames? (zero-training experiment)
"""
import argparse, json, os, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL = '/9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-TEG'
MANIFEST = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/faithfulness_valid.json'
JPEG = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/valid/JPEGImages'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-cases', type=int, default=50)
    ap.add_argument('--prompt', type=str, default='ctx', choices=['plain', 'ctx'])
    ap.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = ap.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(rank)

    manifest = json.load(open(MANIFEST))['cases']
    ta = [c for c in manifest if c['category'] == 'temporal_absence'][:args.max_cases]
    my = [c for i, c in enumerate(ta) if i % world == rank]

    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    absent_h = present_m = 0
    n_abs = n_pre = 0
    for c in my:
        vid = c['video_id']
        frames = [Image.open(os.path.join(JPEG, vid, f + '.jpg')).convert('RGB') for f in c['frames']]
        n = len(frames)
        for t in range(n):
            if args.prompt == 'plain':
                text = f'<image>\nPlease segment {c["query"]} in this image.'
            else:
                # temporal context: how many previous frames were present
                prev = sum(c['presence'][:t])
                text = (f'<image>\nIn the previous {t} frames of this video, the target '
                        f'"{c["query"]}" was visible in {prev} of them. '
                        f'Please segment {c["query"]} in this frame if it is present; '
                        f'otherwise say you do not see it.')
            with torch.no_grad():
                out = model.predict_forward(image=frames[t], text=text,
                                            tokenizer=tok, processor=proc)
            has_mask = bool(out['prediction_masks'] and out['prediction_masks'][0].any())
            if not c['presence'][t]:
                n_abs += 1; absent_h += has_mask
            else:
                n_pre += 1; present_m += (not has_mask)
    print(f'[rank{rank}] prompt={args.prompt} cases={len(my)} '
          f'absent_halluc={absent_h/max(n_abs,1):.4f} ({absent_h}/{n_abs}) '
          f'present_miss={present_m/max(n_pre,1):.4f} ({present_m}/{n_pre})', flush=True)


if __name__ == '__main__':
    main()

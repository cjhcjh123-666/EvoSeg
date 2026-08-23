"""Run EvoSeg B+ v5 on Ref-YT-VOS valid and save RLE predictions for J&F."""
import argparse, json, os
import torch
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL = '/9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-TEG'
META = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/valid/meta_expressions_challenge.json'
JPEG = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/valid/JPEGImages'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/ryvos_b5.json')
    ap.add_argument('--max-videos', type=int, default=None)
    ap.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = ap.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(rank)

    meta = json.load(open(META))['videos']
    vids = list(meta.keys())
    if args.max_videos:
        vids = vids[:args.max_videos]
    my_vids = [v for i, v in enumerate(vids) if i % world == rank]

    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).eval().cuda()
    model.load_temporal_head()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    res = {}
    for vi, vid in enumerate(my_vids):
        exprs = meta[vid]['expressions']
        frame_files = sorted(os.listdir(os.path.join(JPEG, vid)))
        frame_files = [f for f in frame_files if f.endswith('.jpg')][::5]  # every 5
        frames = [Image.open(os.path.join(JPEG, vid, f)).convert('RGB') for f in frame_files]
        vid_res = {}
        for exp_id, e in exprs.items():
            text = f'<image>\nPlease segment {e["exp"]} in this video.'
            with torch.no_grad():
                out = model.predict_forward(video=frames, text=text,
                                            tokenizer=tok, processor=proc,
                                            vlm_all_frames=True)
            pm = out['prediction_masks']
            masks_rle = []
            if pm:
                pred = pm[0]
                for t in range(len(frame_files)):
                    m = pred[t] if t < len(pred) else None
                    if m is not None and m.any():
                        _rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
                        _rle['counts'] = _rle['counts'].decode('utf-8')
                        masks_rle.append(_rle)
                    else:
                        masks_rle.append({'size': list(m.shape) if m is not None else [0, 0],
                                          'counts': ''} if False else None)
            vid_res[e['obj_id']] = {
                'index': 0, 'video_id': vid, 'exp_id': exp_id,
                'text_prediction': out['prediction'][:200],
                'frames': [f[:-4] for f in frame_files],
                'exp': text,
                'prediction_masks': masks_rle,
            }
        res[vid] = vid_res
        if (vi + 1) % 10 == 0:
            print(f'[rank{rank}] {vi+1}/{len(my_vids)} vids', flush=True)
    out_path = args.out.replace('.json', f'_rank{rank}.json')
    json.dump(res, open(out_path, 'w'))
    print(f'[rank{rank}] DONE -> {out_path}', flush=True)


if __name__ == '__main__':
    main()

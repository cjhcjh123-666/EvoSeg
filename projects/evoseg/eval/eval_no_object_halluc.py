"""No-object (empty-target) hallucination eval for Sa2VA-based EvoSeg models.

Feeds gRefCOCO val no-target referring expressions and measures how often the
model emits a [SEG]/mask (hallucination) vs abstains (0 masks).
"""
import argparse
import datetime
import json
import os
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor

GREFS = '/9950backfile/chenjiahui/evo_artifacts/datasets/grefcoco/grefs_unc.json'
IMG_ROOT = '/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/pixel_llm_data/glamm_data/images/coco2014/train2014'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument('--save', type=str, default=None, help='output json path')
    parser.add_argument('--max-refs', type=int, default=None)
    parser.add_argument('--max-sents', type=int, default=None)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    import torch.distributed as dist
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=60))
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)

    refs = [r for r in json.load(open(GREFS))
            if r['split'] == 'val' and (r['no_target'] or r['ann_id'] == [-1])]
    if args.max_refs:
        refs = refs[:args.max_refs]
    # shard over ranks
    my_refs = [r for i, r in enumerate(refs) if i % world == rank]

    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    local = []
    for ref in my_refs:
        img_path = os.path.join(IMG_ROOT, ref['file_name'])
        image = Image.open(img_path).convert('RGB')
        sents = ref['sentences']
        if args.max_sents:
            sents = sents[:args.max_sents]
        for sent in sents:
            text = f"<image>\n Please segment {sent['sent']} in this image."
            with torch.no_grad():
                out = model.predict_forward(
                    image=image, text=text, tokenizer=tokenizer, processor=processor)
            pred_text = out['prediction']
            masks = out['prediction_masks']
            n_masks = len(masks)
            area = 0.0
            if n_masks > 0:
                area = float(masks[0].astype(float).mean())
            local.append({
                'ref_id': ref['ref_id'],
                'sent': sent['sent'],
                'file': ref['file_name'],
                'n_masks': n_masks,
                'mask_area': area,
                'has_seg': '[SEG]' in pred_text,
                'pred_text': pred_text[:300],
            })

    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    all_res = []
    for g in gathered:
        all_res.extend(g)

    if rank == 0:
        n = len(all_res)
        hall = [r for r in all_res if r['n_masks'] > 0]
        n_hall = len(hall)
        seg_no_mask = [r for r in all_res if r['has_seg'] and r['n_masks'] == 0]
        mean_area = float(sum(r['mask_area'] for r in hall) / n_hall) if n_hall else 0.0
        summary = {
            'n_queries': n,
            'n_hallucinated': n_hall,
            'hallucination_rate': round(n_hall / n, 4) if n else None,
            'abstain_rate': round((n - n_hall) / n, 4) if n else None,
            'mean_mask_area_among_hallucinated': round(mean_area, 5),
            'n_seg_but_no_mask': len(seg_no_mask),
        }
        print('=' * 50)
        print(json.dumps(summary, indent=2))
        print('sample hallucinated:', json.dumps(hall[:3], ensure_ascii=False, indent=2))
        print('sample abstained:', json.dumps([r for r in all_res if r['n_masks'] == 0][:3], ensure_ascii=False, indent=2))
        if args.save:
            json.dump({'summary': summary, 'results': all_res},
                      open(args.save, 'w'), ensure_ascii=False)
            print('saved to', args.save)


if __name__ == '__main__':
    main()

"""EvoSeg on the external HalluSegBench counterfactual benchmark.

HalluSegBench (PLAN-Lab/HalluSegBench, test split) contains factual/
counterfactual image pairs: the factual image contains the referent (query =
class_name), the counterfactual image has it edited away. A faithful model must
segment the factual image and REFUSE (empty mask) on the counterfactual image.

Metrics:
  factual_mask_rate   : fraction of factual queries with a non-empty pred mask
  counterfactual_hall : fraction of counterfactual queries with a non-empty
                        pred mask (this is the pixel-grounding hallucination)
  abstain_rate        : 1 - counterfactual_hall
  mean_mask_area_cf   : mask mass on counterfactual (leakage)

Usage (8 GPUs):
  torchrun --nproc_per_node=8 projects/evoseg/eval/hallusegbench_eval.py \
      <model_path> --save <out.json> [--max-pairs 100]
"""
import argparse, datetime, json, os
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

PARQUET = '/tmp/hsb_try/viewer/test.parquet'
IMG_ROOT = ('/9950backfile/chenjiahui/evo_artifacts/datasets/hallusegbench/test/'
            'refer_seg')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('model_path')
    p.add_argument('--save', default=None)
    p.add_argument('--max-pairs', type=int, default=None)
    p.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return p.parse_args()


def main():
    import pandas as pd
    import torch.distributed as dist
    args = parse_args()
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=120))
    rank = dist.get_rank(); world = dist.get_world_size()
    torch.cuda.set_device(rank)

    df = pd.read_parquet(PARQUET)
    df = df[df['split'] == 'test']
    pairs = sorted(df['pair_id'].unique())
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    my_pairs = [p for i, p in enumerate(pairs) if i % world == rank]

    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    local = []
    for pair in my_pairs:
        sub = df[df['pair_id'] == pair]
        fact = sub[sub['is_counterfactual'] == False]
        cf = sub[sub['is_counterfactual'] == True]
        if len(fact) == 0 or len(cf) == 0:
            continue
        cls = fact.iloc[0]['class_name']
        fimg = fact.iloc[0]['image_relpath']
        cimg = cf.iloc[0]['image_relpath']
        fp = os.path.join(IMG_ROOT, os.path.basename(fimg))
        cp = os.path.join(IMG_ROOT, os.path.basename(cimg))
        # parquet relpath is test/refer_seg/factual_images/<name>; counterfactual
        # is counterfactual_images/<name>
        fdir = 'factual_images' if 'factual' in fimg else 'counterfactual_images'
        cdir = 'counterfactual_images' if 'counterfactual' in cimg else 'factual_images'
        fp = os.path.join(IMG_ROOT, fdir, os.path.basename(fimg))
        cp = os.path.join(IMG_ROOT, cdir, os.path.basename(cimg))
        if not (os.path.isfile(fp) and os.path.isfile(cp)):
            print(f'rank{rank} missing images {pair}: {fp} {cp}', flush=True)
            continue
        query = f'the {cls}'
        text = f"<image>\n Please segment {query} in this image."
        rec = {'pair': pair, 'class_name': cls, 'query': query}
        for tag, ip in (('factual', fp), ('counterfactual', cp)):
            image = Image.open(ip).convert('RGB')
            with torch.no_grad():
                out = model.predict_forward(
                    image=image, text=text, tokenizer=tokenizer, processor=processor)
            masks = out['prediction_masks']
            n_masks = len(masks)
            area = float(masks[0].astype(float).mean()) if n_masks > 0 else 0.0
            rec[f'{tag}_n_masks'] = n_masks
            rec[f'{tag}_mask_area'] = area
        local.append(rec)

    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    all_res = [r for g in gathered for r in g]

    if rank == 0:
        n = len(all_res)
        f_hit = sum(1 for r in all_res if r['factual_n_masks'] > 0)
        c_hall = sum(1 for r in all_res if r['counterfactual_n_masks'] > 0)
        c_area = sum(r['counterfactual_mask_area'] for r in all_res)
        summary = {
            'n_pairs': n,
            'factual_mask_rate': round(f_hit / n, 4) if n else None,
            'counterfactual_hallucination_rate': round(c_hall / n, 4) if n else None,
            'counterfactual_abstain_rate': round((n - c_hall) / n, 4) if n else None,
            'mean_counterfactual_mask_area': round(c_area / n, 5) if n else None,
        }
        print('=' * 60)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if args.save:
            json.dump({'summary': summary, 'results': all_res},
                      open(args.save, 'w'), ensure_ascii=False)


if __name__ == '__main__':
    main()

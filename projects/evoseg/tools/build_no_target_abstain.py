#!/usr/bin/env python
"""Build no-target (abstain) SFT data from gRefCOCO train.

Diagnosis: the 4B/8B multitask SFT mix contains NO no-target examples
(build_pixel_llm_grefcoco.py skips them), so the model never learned to say
"the target does not exist" and hallucinates 100% on empty-target queries. RL
cannot fix this on its own because sampling never explores the abstain path.

This tool converts gRefCOCO train no-target referring expressions into
Pixel-LLM finetune entries with an explicit abstain response (no [SEG] token):

  {"image": <coco file_name>, "text": [query], "response": "<abstain text>"}

The companion dataset class Sa2VA07NoTargetDataset reads this manifest, emits
question+abstain-answer conversations WITHOUT masks (the model's pseudo-zero
mask path then reinforces empty masks), so the LLM learns to refuse.

Usage:
  python projects/evoseg/tools/build_no_target_abstain.py \
      --out /9950backfile/chenjiahui/evo_artifacts/datasets/s4b/pixel_llm_data/notarget/annotations.json \
      [--max-refs 2000]
"""
import argparse
import json
import os
import random

GREFS = '/9950backfile/chenjiahui/evo_artifacts/datasets/grefcoco/grefs_unc.json'
IMG_ROOT = ('/9950backfile/chenjiahui/evo_artifacts/datasets/coco2014/train2014')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--max-refs', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def strip_article(phrase):
    low = phrase.lower()
    for art in ('a ', 'an ', 'the '):
        if low.startswith(art):
            return phrase[len(art):]
    return phrase


def abstain_templates(phrase):
    """Diverse abstain answers, all without [SEG]."""
    noun = strip_article(phrase)
    return [
        f'There is no {noun} in the image.',
        f'I cannot find {phrase} in this image.',
        f'No {noun} exists in this image.',
        'The described object is not present in this image.',
        f"I don't see {phrase} in this image.",
    ]


def main():
    args = parse_args()
    grefs = json.load(open(GREFS))
    rng = random.Random(args.seed)
    out = []
    n_skip = 0
    for r in grefs:
        if r['split'] != 'train' or not (r['no_target'] or r['ann_id'] == [-1]):
            continue
        fn = os.path.join(IMG_ROOT, r['file_name'])
        if not os.path.exists(fn):
            n_skip += 1
            continue
        q = r['sentences'][0]['sent']
        if q.endswith('.'):
            q = q[:-1]
        resp = rng.choice(abstain_templates(q))
        out.append({'image': r['file_name'], 'text': [q], 'response': resp})
        if args.max_refs and len(out) >= args.max_refs:
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, 'w'), ensure_ascii=False)
    print(f'wrote {len(out)} no-target abstain samples (skipped {n_skip}) -> {args.out}')
    for e in out[:3]:
        print('  q:', e['text'][0], '| a:', e['response'])


if __name__ == '__main__':
    main()

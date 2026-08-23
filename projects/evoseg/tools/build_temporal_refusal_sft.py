"""Build temporal-aware refusal SFT data.

For temporal_absence / identity_swap / global_absence train cases, construct
per-frame examples:
  input : last K frames (images) + query + instruction
  target: "[SEG]" if the referent is present in the LAST frame, else a
          refusal text ("I don't see ... in this frame.")
The model learns to use real visual temporal context to decide frame-wise
existence. Output: jsonl list of {conversations: [{role,content}], frames, ...}
"""
import argparse
import json
import os
import random

K = 4          # temporal context window (last K frames incl. current)
JPEG = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/train/JPEGImages'
MAN = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/faithfulness_train.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/pixel_llm_data/temporal_refusal/annotations.json')
    ap.add_argument('--max-temporal', type=int, default=2194)
    ap.add_argument('--max-identity', type=int, default=2500)
    ap.add_argument('--max-global', type=int, default=1200)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cases = json.load(open(MAN))['cases']
    by_cat = {}
    for c in cases:
        by_cat.setdefault(c['category'], []).append(c)
    sel = []
    sel += random.sample(by_cat['temporal_absence'], min(args.max_temporal, len(by_cat['temporal_absence'])))
    sel += random.sample(by_cat['identity_swap'], min(args.max_identity, len(by_cat['identity_swap'])))
    sel += random.sample(by_cat['global_absence'], min(args.max_global, len(by_cat['global_absence'])))
    t = by_cat['temporal_absence']; i = by_cat['identity_swap']; g = by_cat['global_absence']
    print(f'selected cases: temporal={len(t)} identity={len(i)} global={len(g)}')
    examples = []
    for c in sel:
        vid = c['video_id']
        frames = c['frames']
        pres = c['presence']
        T = len(frames)
        for t in range(T):
            lo = max(0, t - K + 1)
            win_frames = frames[lo:t + 1]          # last K frames incl current
            # presence summary of the window's earlier frames (context)
            prev_present = sum(pres[lo:t])
            label = pres[t]
            content = [{'type': 'image', 'image': os.path.join(JPEG, vid, f + '.jpg')} for f in win_frames]
            inst = (f'In this video clip, the target "{c["query"]}" was visible in '
                    f'{prev_present} of the previous {t - lo} frames. Decide whether it '
                    f'is present in the LAST frame. If present, output [SEG]. '
                    f'If not present, say you do not see it.')
            content.append({'type': 'text', 'text': inst})
            answer = f'It is present. [SEG].' if label else \
                     f'I do not see {c["query"]} in the last frame.'
            examples.append({
                'conversations': [{'role': 'user', 'content': content},
                                  {'role': 'assistant', 'content': answer}],
                'frames': win_frames, 'vid': vid, 'query': c['query'],
                'presence': label, 't': t, 'category': c['category'],
            })
    json.dump({'n_examples': len(examples), 'examples': examples},
              open(args.out, 'w'))
    print(f'wrote {len(examples)} examples -> {args.out}')


if __name__ == '__main__':
    main()

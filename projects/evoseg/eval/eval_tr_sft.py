"""Eval temporal-refusal SFT: per-frame detection with temporal window."""
import argparse, json, os, torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL = '/9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-TEG'
LORA = '/9950backfile/chenjiahui/evo_artifacts/checkpoints/tr_sft3/lora'
MANIFEST = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/faithfulness_valid.json'
JPEG = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/valid/JPEGImages'
K = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-cases', type=int, default=None)
    ap.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = ap.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(rank)

    manifest = json.load(open(MANIFEST))['cases']
    cases = [c for c in manifest if c['category'] == 'temporal_absence']
    if args.max_cases:
        cases = cases[:args.max_cases]
    my = [c for i, c in enumerate(cases) if i % world == rank]

    base = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    base.model = PeftModel.from_pretrained(base.model, LORA)
    base.model.eval()
    model = base.cuda()

    absent_h = present_m = 0
    n_abs = n_pre = 0
    for c in my:
        vid = c['video_id']
        all_frames = [Image.open(os.path.join(JPEG, vid, f + '.jpg')).convert('RGB') for f in c['frames']]
        T = len(all_frames)
        for t in range(T):
            lo = max(0, t - K + 1)
            win = all_frames[lo:t + 1]
            prev = sum(c['presence'][lo:t])
            inst = (f'Please segment {c["query"]} in the LAST frame of this '
                    f'video clip if it is present; otherwise output None.')
            content = [{'type': 'image', 'image': fr} for fr in win]
            content.append({'type': 'text', 'text': inst})
            messages = [{'role': 'user', 'content': content}]
            ps = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            mm = proc(text=[ps], images=[win], videos=None, padding=True,
                      return_tensors='pt', min_pixels=model.min_pixels,
                      max_pixels=768 * 28 * 28).to(model.device)
            with torch.no_grad():
                out = model.model.generate(**mm, max_new_tokens=32, do_sample=False)
            txt = proc.batch_decode(out[:, len(mm.input_ids[0]):], skip_special_tokens=False)[0].strip().lower()
            has_seg = '[seg]' in txt.lower() or 'present' in txt
            if not c['presence'][t]:
                n_abs += 1; absent_h += has_seg
            else:
                n_pre += 1; present_m += (not has_seg)
    print(f'[rank{rank}] cases={len(my)} absent_halluc={absent_h/max(n_abs,1):.4f} ({absent_h}/{n_abs}) '
          f'present_miss={present_m/max(n_pre,1):.4f} ({present_m}/{n_pre})', flush=True)


if __name__ == '__main__':
    main()

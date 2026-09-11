"""M5.2 — build the human prefix-identifiability annotation bundle.

For each selected case we sample *checkpoints* (coarse fractions of the
annotated frames plus event-driven ones derived from GT visibility/proximity —
used only to place checkpoints, never shown as the answer) and emit a
self-contained local HTML tool:

  * the annotator sees the query plus the video prefix up to the checkpoint
    (a small filmstrip of raw frames);
  * an optional instance palette (GT masks + obj ids) exists ONLY to name the
    object when the label is UNIQUE;
  * labels: AMBIGUOUS / UNIQUE / INVALID / UNCERTAIN (+ obj id, confidence
    1-5, evidence type, note);
  * state is kept in localStorage per annotator (>=3 supported) and can be
    exported to JSON and reloaded.

Images are written as ordinary JPEG files under `<out-dir>/frames/` and
referenced relatively, so the HTML tool stays small enough for a browser
(embedding everything as base64 produced a ~96 MB page).

No model/LLM label is ever produced here: the bundle is annotation-ready only.

The heavy HTML lives under the artifact root (never committed); this script and
the README/index are what get committed.

Usage
-----
python build_annotation_bundle.py \
  --pool <artifact_root>/results/premature_commitment/candidate_pool.json \
  --out-dir <artifact_root>/results/premature_commitment/annotation_bundle \
  --top-k 80 --max-checkpoints 12
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
from PIL import Image

DEFAULT_ANN = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/valid/Annotations'
DEFAULT_JPEG = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/valid/JPEGImages'
DEFAULT_META = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/valid/meta_expressions_challenge.json'


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pool', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--ann-root', default=DEFAULT_ANN)
    ap.add_argument('--jpeg-root', default=DEFAULT_JPEG)
    ap.add_argument('--meta', default=DEFAULT_META)
    ap.add_argument('--top-k', type=int, default=80)
    ap.add_argument('--max-checkpoints', type=int, default=12)
    ap.add_argument('--thumb-width', type=int, default=360)
    ap.add_argument('--fractions', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    return ap.parse_args()


def safe_name(s: str) -> str:
    return re.sub(r'[^0-9A-Za-z_.-]+', '_', str(s))


def write_jpeg(img: Image.Image, width: int, path: str) -> None:
    im = img.convert('RGB')
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    im.save(path, 'JPEG', quality=82, optimize=True)


def mask_overlay(frame: Image.Image, mask: np.ndarray, color=(0, 200, 0), alpha=0.45) -> Image.Image:
    a = np.array(frame.convert('RGB')).copy()
    m = np.asarray(mask).astype(bool)
    if m.shape != a.shape[:2]:
        m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(
            (a.shape[1], a.shape[0]), Image.NEAREST)) > 0
    a[m] = (a[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return Image.fromarray(a)


def main():
    args = parse_args()
    pool = json.load(open(args.pool))
    cases = pool['cases'][:args.top_k] if isinstance(pool, dict) else pool[:args.top_k]
    meta = json.load(open(args.meta))['videos']
    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    bundle = {'schema_version': 1, 'n_cases': len(cases), 'annotators_supported': 3,
              'states': ['AMBIGUOUS', 'UNIQUE', 'INVALID', 'UNCERTAIN'],
              'evidence_types': ['appearance', 'action', 'temporal_order', 'relation',
                                 'reappearance', 'other'],
              'image_root': 'frames',
              'cases': []}
    for c in cases:
        vid = c['video_id']
        cid = safe_name(f"{vid}_{c['exp_id']}")
        frames = c['annotated_frames']
        T = len(frames)
        obj_ids = [c['target_obj_id']] + [o for o in c.get('candidate_obj_ids', [])]
        # checkpoints: coarse fractions (+ midpoint-of-longest-target-gap if any)
        idxs = sorted({max(0, min(T - 1, int(round(f * T)) - 1)) for f in args.fractions})
        sig = c.get('mining_signals', {})
        if sig.get('target_gap_frames', 0) > 0:
            idxs.append(max(0, min(T - 1, T // 2)))
        idxs = sorted(set(idxs))[:args.max_checkpoints]

        # instance palette thumbnails (GT mask overlay) – naming aid only
        palette = []
        for o in obj_ids:
            eid = None
            for eid_, e in meta[vid]['expressions'].items():
                if str(e['obj_id']) == str(o):
                    eid = eid_; break
            if eid is None:
                continue
            mid = frames[min(len(frames) - 1, len(frames) // 2)]
            p = os.path.join(args.ann_root, vid, str(eid), mid + '.png')
            if not os.path.exists(p):
                continue
            mask = np.array(Image.open(p).convert('L')) > 0
            fr = Image.open(os.path.join(args.jpeg_root, vid, mid + '.jpg')).convert('RGB')
            rel = f"frames/{cid}_pal_obj{o}.jpg"
            write_jpeg(mask_overlay(fr, mask), 200, os.path.join(args.out_dir, rel))
            palette.append({'obj_id': str(o),
                            'category': meta[vid].get('objects', {}).get(str(o), {}).get('category', ''),
                            'thumb': rel})

        checkpoints = []
        for ci_, t in enumerate(idxs):
            # filmstrip of the prefix (up to 6 evenly spaced raw frames)
            k = min(6, t + 1)
            sel = sorted({int(i * t / max(1, k - 1)) for i in range(k)}) if k > 1 else [0]
            thumbs = []
            for i in sel:
                fr = Image.open(os.path.join(args.jpeg_root, vid, frames[i] + '.jpg')).convert('RGB')
                rel = f"frames/{cid}_cp{ci_}_i{i}.jpg"
                write_jpeg(fr, args.thumb_width, os.path.join(args.out_dir, rel))
                thumbs.append({'frame_idx': i, 'frame': frames[i], 'img': rel})
            checkpoints.append({'checkpoint_idx': len(checkpoints), 'prefix_last_frame_idx': t,
                                'prefix_last_frame': frames[t], 'thumbs': thumbs})
        bundle['cases'].append({
            'case_id': f"{vid}:{c['exp_id']}",
            'video_id': vid, 'exp_id': c['exp_id'], 'query': c['query'],
            'target_obj_id': c['target_obj_id'], 'candidate_obj_ids': c.get('candidate_obj_ids', []),
            'n_frames': T, 'mining_signals': sig,
            'checkpoints': checkpoints, 'palette': palette,
        })

    bundle_path = os.path.join(args.out_dir, 'annotation_bundle.json')
    json.dump(bundle, open(bundle_path, 'w'), ensure_ascii=False)
    html = render_html(bundle)
    html_path = os.path.join(args.out_dir, 'annotate.html')
    open(html_path, 'w').write(html)
    json.dump({'bundle': bundle_path, 'html': html_path, 'n_cases': len(bundle['cases']),
               'n_checkpoints': sum(len(c['checkpoints']) for c in bundle['cases']),
               'regenerate': 'python projects/evoseg/premature_commitment/build_annotation_bundle.py '
                             f'--pool {args.pool} --out-dir {args.out_dir} --top-k {args.top_k}'},
              open(os.path.join(args.out_dir, 'index.json'), 'w'), indent=1)
    print(f'[annot] {len(bundle["cases"])} cases, '
          f'{sum(len(c["checkpoints"]) for c in bundle["cases"])} checkpoints -> {html_path}', flush=True)


def render_html(bundle: dict) -> str:
    data = json.dumps(bundle, ensure_ascii=False)
    return """<!DOCTYPE html><html><head><meta charset="utf-8"><title>M5 prefix-identifiability annotation</title>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#fafafa;color:#111}
#top{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;z-index:5;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.card{background:#fff;border:1px solid #e3e3e3;border-radius:10px;padding:14px;margin:12px 16px}
.q{font-size:19px;font-weight:700}
.meta{font-size:12px;color:#555;margin:4px 0 10px}
.strip{display:flex;gap:4px;flex-wrap:wrap}
.strip img{height:110px;border:1px solid #ccc;border-radius:4px}
.pal{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
.pal div{font-size:11px;text-align:center}
.pal img{height:64px;display:block;border:1px solid #ccc;border-radius:4px}
.btns button{font-size:14px;padding:7px 14px;margin-right:6px;border:none;border-radius:7px;cursor:pointer}
.b_amb{background:#ffe08a}.b_uni{background:#bfe8bf}.b_inv{background:#f7c2c2}.b_unc{background:#dcdcdc}
.sel{outline:3px solid #06c}
#prog{height:8px;background:#eee;border-radius:4px;overflow:hidden;flex:1;min-width:160px}
#prog>div{height:100%;background:#4c4}
label{font-size:13px}
select,input{font-size:13px;padding:3px}
#done{display:none;background:#eef8ee;border:1px solid #8c8;border-radius:10px;padding:16px;margin:12px 16px}
</style></head><body>
<div id="top">
  <label>annotator: <select id="who"><option>A</option><option>B</option><option>C</option></select></label>
  <div id="prog"><div id="bar"></div></div><span id="cnt"></span>
  <button onclick="exportAll()">export JSON</button>
  <label><input type="checkbox" id="showPal"> show instance palette (naming aid)</label>
  <button onclick="clearAll()">clear</button>
</div>
<div id="caseroot"></div>
<div id="done"><b>All checkpoints labelled.</b><br>Click “export JSON” and send the file for adjudication /
kappa computation.</div>
<script>
const BUNDLE = __DATA__;
const KEY = () => 'm5_annot_' + document.getElementById('who').value;
let labels = JSON.parse(localStorage.getItem(KEY()) || '{}');
let ci = 0, ki = 0;
const total = () => BUNDLE.cases.reduce((a,c)=>a+c.checkpoints.length,0);
function doneCount(){ return Object.keys(labels).length; }
function save(){ localStorage.setItem(KEY(), JSON.stringify(labels)); }
function render(){
  const c = BUNDLE.cases[ci];
  document.getElementById('cnt').textContent = doneCount() + '/' + total();
  document.getElementById('bar').style.width = Math.round(100*doneCount()/total()) + '%';
  document.getElementById('done').style.display = (doneCount()>=total()) ? 'block' : 'none';
  if (!c) { document.getElementById('caseroot').innerHTML=''; return; }
  const cp = c.checkpoints[ki];
  const key = c.case_id + '#' + cp.checkpoint_idx;
  const cur = labels[key];
  const strip = cp.thumbs.map(t=>`<img src="${t.img}" title="frame ${t.frame}">`).join('');
  const pal = c.palette.map(p=>`<div><img src="${p.thumb}"><span>obj ${p.obj_id} (${p.category||'?'})</span></div>`).join('');
  const opts = c.palette.map(p=>`<option value="${p.obj_id}" ${cur&&cur.selected_obj_id===p.obj_id?'selected':''}>obj ${p.obj_id} — ${p.category||'?'}</option>`).join('');
  const sel = (v)=> cur && cur.state===v ? 'sel' : '';
  document.getElementById('caseroot').innerHTML = `<div class="card">
    <div class="q">${c.query}</div>
    <div class="meta">case ${ci+1}/${BUNDLE.cases.length} · checkpoint ${ki+1}/${c.checkpoints.length}
      · prefix ends at annotated frame #${cp.prefix_last_frame_idx} (of ${c.n_frames})
      · video ${c.video_id} · exp ${c.exp_id} · target obj ${c.target_obj_id}</div>
    <div class="strip">${strip}</div>
    <div id="palwrap" style="display:none"><div class="pal">${pal}</div></div>
    <div style="margin:10px 0"><b>Using only the video evidence up to this checkpoint, can the expression uniquely identify one object?</b></div>
    <div class="btns">
      <button class="b_amb ${sel('AMBIGUOUS')}" onclick="mark('AMBIGUOUS')">AMBIGUOUS</button>
      <button class="b_uni ${sel('UNIQUE')}" onclick="mark('UNIQUE')">UNIQUE</button>
      <button class="b_inv ${sel('INVALID')}" onclick="mark('INVALID')">INVALID</button>
      <button class="b_unc ${sel('UNCERTAIN')}" onclick="mark('UNCERTAIN')">UNCERTAIN</button>
    </div>
    <div style="margin-top:8px">
      <label>object (if UNIQUE): <select id="objsel">${opts}</select></label>
      <label>confidence 1-5: <select id="conf"><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
      <label>evidence: <select id="ev"><option>appearance</option><option>action</option><option>temporal_order</option><option>relation</option><option>reappearance</option><option>other</option></select></label>
      <input id="note" placeholder="optional note" style="width:220px" value="${cur&&cur.note?cur.note:''}">
    </div>
    <div style="margin-top:8px;font-size:12px;color:#666">keys: 1 AMBIGUOUS · 2 UNIQUE · 3 INVALID · 4 UNCERTAIN · ←/→ move</div>
  </div>`;
  if (cur) { document.getElementById('conf').value = cur.confidence||3;
             document.getElementById('ev').value = cur.evidence_type||'appearance'; }
  document.getElementById('palwrap').style.display = document.getElementById('showPal').checked ? 'block':'none';
}
function mark(state){
  const c = BUNDLE.cases[ci], cp = c.checkpoints[ki];
  const key = c.case_id + '#' + cp.checkpoint_idx;
  labels[key] = {case_id:c.case_id, annotator_id:document.getElementById('who').value,
    checkpoint_idx:cp.checkpoint_idx, prefix_last_frame_idx:cp.prefix_last_frame_idx,
    state:state,
    selected_obj_id: (state==='UNIQUE') ? document.getElementById('objsel').value : null,
    confidence: parseInt(document.getElementById('conf').value),
    evidence_type: document.getElementById('ev').value,
    note: document.getElementById('note').value};
  save(); next();
}
function next(){ if (ki+1 < BUNDLE.cases[ci].checkpoints.length) ki++; else { ki=0; ci++; } render(); }
function prev(){ if (ki>0) ki--; else if (ci>0){ci--; ki=BUNDLE.cases[ci].checkpoints.length-1;} render(); }
function exportAll(){
  const rows = [];
  for (const c of BUNDLE.cases) for (const cp of c.checkpoints){
    const k = c.case_id+'#'+cp.checkpoint_idx;
    const l = labels[k];
    rows.push({case_id:c.case_id, video_id:c.video_id, exp_id:c.exp_id, query:c.query,
      checkpoint_idx:cp.checkpoint_idx, prefix_last_frame_idx:cp.prefix_last_frame_idx,
      state: l?l.state:'UNLABELLED', selected_obj_id: l?l.selected_obj_id:null,
      confidence: l?l.confidence:null, evidence_type: l?l.evidence_type:null, note: l?l.note:''});
  }
  const payload = {annotator_id:document.getElementById('who').value, schema_version:1, labels:rows};
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,1)],{type:'application/json'}));
  a.download='m5_annot_'+document.getElementById('who').value+'.json'; a.click();
}
function clearAll(){ if(confirm('clear this annotator?')){ labels={}; save(); ci=0; ki=0; render(); } }
document.getElementById('who').onchange = ()=>{ labels=JSON.parse(localStorage.getItem(KEY())||'{}'); render(); };
document.getElementById('showPal').onchange = render;
document.addEventListener('keydown', e=>{
  if(e.key==='1') mark('AMBIGUOUS'); else if(e.key==='2') mark('UNIQUE');
  else if(e.key==='3') mark('INVALID'); else if(e.key==='4') mark('UNCERTAIN');
  else if(e.key==='ArrowRight') next(); else if(e.key==='ArrowLeft') prev();
});
render();
</script></body></html>""".replace('__DATA__', data)


if __name__ == '__main__':
    main()

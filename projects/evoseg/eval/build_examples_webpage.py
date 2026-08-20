"""Build the multi-model showcase webpage for EvoSeg temporal faithfulness.

Reads <dir>/<case>/meta.json + <dir>/<case>/<model>.json/.mp4 and writes a
self-contained <dir>/index.html (inline JSON, relative videos) that opens via
file://.

Usage:
  python build_examples_webpage.py --dir <examples_dir>
"""
import argparse
import glob
import json
import os

PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EvoSeg · 时序忠实指代分割（模型对比）</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3;
          --muted:#8b949e; --green:#2ea043; --red:#f85149; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font:14px/1.5 -apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif; padding:24px; }
  h1 { font-size:22px; margin-bottom:4px; }
  .sub { color:var(--muted); margin-bottom:16px; }
  .note { background:#1f6feb14; border:1px solid #1f6feb44; color:#79c0ff; border-radius:8px; padding:8px 12px; margin-bottom:16px; font-size:13px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:12px; margin-bottom:20px; overflow:hidden; }
  .card-head { padding:12px 16px; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .badge { font-size:11px; padding:2px 10px; border-radius:20px; font-weight:600; }
  .b-temporal_absence{background:#1f6feb33;color:#58a6ff;border:1px solid #1f6feb66}
  .b-identity_swap{background:#d2992233;color:#e3b341;border:1px solid #d2992266}
  .b-global_absence{background:#2ea04333;color:#3fb950;border:1px solid #2ea04366}
  .b-counterfactual_swap{background:#f8514933;color:#ff7b72;border:1px solid #f8514966}
  .query { color:var(--muted); font-size:13px; margin-left:auto; font-style:italic; }
  .body { padding:12px 16px; }
  .vid { width:100%; border-radius:8px; background:#000; }
  .row { display:flex; align-items:center; gap:10px; margin:4px 0; }
  .lbl { width:150px; flex:none; font-size:12px; color:var(--text); }
  .strip { display:flex; gap:2px; flex-wrap:wrap; }
  .blk { width:13px; height:18px; border-radius:2px; font-size:8px; color:#000; display:flex; align-items:center; justify-content:center; }
  .on { background:var(--green); } .off { background:var(--red); }
  .legend { font-size:12px; color:var(--muted); margin:8px 0 4px; }
  .sel { margin:10px 0 6px; font-size:13px; }
  select { background:#0d1117; color:var(--text); border:1px solid var(--border); border-radius:6px; padding:4px 8px; }
  .meta { font-size:12px; color:var(--muted); margin-top:8px; }
  .sum { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px; }
  .sum .s { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:8px 14px; }
  .sum .num { font-size:18px; font-weight:700; }
  .sum .lbl { color:var(--muted); font-size:11px; }
</style>
</head>
<body>
  <h1>🎯 EvoSeg · 时序忠实指代分割（模型对比）</h1>
  <div class="sub">目标存在性是<em>逐帧谓词</em> e<sub>t</sub>。每个 case 用同一批 query 跑所有模型版本；绿块 = 该帧应有 mask，红块 = 该帧不应有 mask。</div>
  <div class="note">GRU 时序头训练中；训练完成后自动追加 <b>TEG-4B+GRU</b> 行并更新本页。</div>
  <div class="sum" id="sum"></div>
  <div id="cards"></div>

<script id="data" type="application/json">__DATA__</script>
<script>
const CASES = JSON.parse(document.getElementById('data').textContent);
function strip(arr){return arr.map(v=>v?'<span class="blk on">1</span>':'<span class="blk off">0</span>').join('');}
function render(){
  const per = {};
  for (const c of CASES) for (const m of c.models) {
    if (!per[m]) per[m] = {hard:0, stop:0, ok:0};
    for (let t=0;t<c.expected.length;t++){
      const post = c.data[m].presence[t] || false;
      per[m].ok += (post===c.expected[t])?1:0;
      if (!c.expected[t]){ per[m].hard++; if(!post) per[m].stop++; }
    }
  }
  document.getElementById('sum').innerHTML = Object.keys(per).map(m=>{
    const s = per[m];
    const acc = (100*s.ok/(s.hard + s.ok)).toFixed(1);
    const stop = s.hard ? (100*s.stop/s.hard).toFixed(0) : '—';
    return `<div class="s"><div class="num">${acc}% / ${stop}%</div><div class="lbl">${m} · 帧准确率 / absent 停止率</div></div>`;
  }).join('');
  document.getElementById('cards').innerHTML = CASES.map((c,ci)=>{
    const rows = c.models.map(m=>`<div class="row"><div class="lbl">${m}</div><div class="strip">${strip(c.data[m].presence)}</div></div>`).join('');
    const opts = c.models.map((m,i)=>`<option value="${i}" ${i===0?'selected':''}>${m}</option>`).join('');
    return `<div class="card">
      <div class="card-head"><span class="badge b-${c.category}">${c.category.replace('_',' ')}</span><b>${c.title}</b><span class="query">"${c.query}"</span></div>
      <div class="body">
        <div class="sel">播放模型：<select data-vid="${ci}" onchange="document.getElementById('vid${ci}').src=this.value">${opts}</select></div>
        <video id="vid${ci}" class="vid" src="${c.videos[0]}" autoplay loop muted controls playsinline></video>
        <div class="legend">目标存在性时序（绿=该帧应有 mask，红=不应有；GT 在最上行）</div>
        <div class="row"><div class="lbl"><b>GT</b></div><div class="strip">${strip(c.expected)}</div></div>
        ${rows}
        <div class="meta">video_id=${c.video_id} · 帧数=${c.n_frames} · 每格 1 帧</div>
      </div></div>`;
  }).join('');
  // wire select -> video src
  for (const sel of document.querySelectorAll('select[data-vid]')) {
    sel.addEventListener('change', e => {
      const c = CASES[+sel.dataset.vid];
      document.getElementById('vid'+sel.dataset.vid).src = c.videos[+sel.value];
      document.getElementById('vid'+sel.dataset.vid).load();
    });
  }
}
render();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/9950backfile/chenjiahui/evo_artifacts/data/examples')
    args = ap.parse_args()

    cases = []
    for meta_p in sorted(glob.glob(os.path.join(args.dir, '*/meta.json'))):
        meta = json.load(open(meta_p))
        data = {}
        videos = []
        for m in meta['models']:
            jp = os.path.join(args.dir, meta['title'], f'{m}.json')
            vp = os.path.join(args.dir, meta['title'], f'{m}.mp4')
            if os.path.exists(jp):
                data[m] = json.load(open(jp))
                videos.append(f"{meta['title']}/{m}.mp4")
        cases.append({**meta, 'data': data, 'videos': videos})

    html = PAGE.replace('__DATA__', json.dumps(cases, ensure_ascii=False))
    out = os.path.join(args.dir, 'index.html')
    with open(out, 'w') as f:
        f.write(html)
    print(f'wrote {out} with {len(cases)} cases')
    for c in cases:
        print(f"  {c['title']}: models={c['models']} videos={len(c['videos'])}")


if __name__ == '__main__':
    main()

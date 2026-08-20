"""Build a self-contained showcase webpage for EvoSeg temporal faithfulness
before/after video examples.

Reads <out>/*.json (per-case data) + <out>/*.mp4 (side-by-side videos) and
writes <out>/index.html that works when opened directly via file:// (data is
inlined; videos are referenced relatively).

Usage:
  python build_examples_webpage.py --dir /9950backfile/chenjiahui/evo_artifacts/data/examples
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
<title>EvoSeg · 时序忠实指代分割（Temporal Faithfulness）</title>
<style>
  :root {
    --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3;
    --muted:#8b949e; --green:#2ea043; --red:#f85149; --amber:#d29922;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font:14px/1.5 -apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif; padding:24px; }
  h1 { font-size:22px; margin-bottom:4px; }
  .sub { color:var(--muted); margin-bottom:20px; }
  .stats { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:24px; }
  .stat { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:10px 18px; }
  .stat .num { font-size:20px; font-weight:700; }
  .stat .lbl { color:var(--muted); font-size:12px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:12px; margin-bottom:20px; overflow:hidden; }
  .card-head { padding:12px 16px; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .badge { font-size:11px; padding:2px 10px; border-radius:20px; font-weight:600; }
  .b-temporal { background:#1f6feb33; color:#58a6ff; border:1px solid #1f6feb66; }
  .b-identity { background:#d2992233; color:#e3b341; border:1px solid #d2992266; }
  .b-global { background:#2ea04333; color:#3fb950; border:1px solid #2ea04366; }
  .b-counter { background:#f8514933; color:#ff7b72; border:1px solid #f8514966; }
  .query { color:var(--muted); font-size:13px; margin-left:auto; font-style:italic; }
  .body { padding:12px 16px; }
  .vid { width:100%; border-radius:8px; background:#000; }
  .legend { font-size:12px; color:var(--muted); margin:8px 0 4px; }
  .strip { display:flex; gap:2px; margin:4px 0 10px; }
  .strip .blk { width:14px; height:18px; border-radius:2px; display:flex; align-items:center; justify-content:center; font-size:8px; color:#000; }
  .on { background:var(--green); }
  .off { background:var(--red); }
  .strip-label { font-size:11px; color:var(--muted); width:72px; flex:none; line-height:18px; }
  .strip-row { display:flex; align-items:center; gap:8px; }
  .meta { font-size:12px; color:var(--muted); margin-top:4px; }
  .note { background:#1f6feb14; border:1px solid #1f6feb44; color:#79c0ff; border-radius:8px; padding:8px 12px; margin-bottom:16px; font-size:13px; }
</style>
</head>
<body>
  <h1>🎯 EvoSeg · 时序忠实指代分割</h1>
  <div class="sub">目标存在性是<em>逐帧谓词</em>：<code>e<sub>t</sub></code>。BEFORE = 无存在性门控的 SAM2 传播（目标消失后仍继续出 mask）；AFTER = 时序存在性门控（目标消失即干净停住）。</div>
  <div class="note">当前 AFTER 一列使用逐帧 MLP 门控（GRU 时序头正在训练，训完自动更新本页）。绿色块 = 该帧应有 mask，红色块 = 该帧不应有 mask。上方视频绿叠影 = 模型实际分割区域。</div>
  <div class="stats" id="stats"></div>
  <div id="cards"></div>

<script id="data" type="application/json">__DATA__</script>
<script>
const CASES = JSON.parse(document.getElementById('data').textContent);

function strip(label, arr) {
  const on = (v) => v ? '<span class="blk on">1</span>' : '<span class="blk off">0</span>';
  const html = arr.map(on).join('');
  return `<div class="strip-row"><div class="strip-label">${label}</div><div class="strip">${html}</div></div>`;
}
function catBadge(c) {
  const m = {'temporal_absence':'temporal','identity_swap':'identity','global_absence':'global','counterfactual_swap':'counterfactual'};
  const k = m[c] || c;
  return `<span class="badge b-${k}">${c.replace('_',' ')}</span>`;
}
function render() {
  // summary stats
  let nStop = 0, nHard = 0;
  for (const c of CASES) {
    const exp = c.expected, pre = c.pred_before, post = c.pred_after;
    for (let t=0;t<exp.length;t++){
      if (!exp[t]) { nHard++; if (!post[t]) nStop++; }
    }
  }
  const stats = [
    ['展示案例', CASES.length + ' 段视频', ''],
    ['absent 帧停止率(AFTER)', (100*nStop/Math.max(nHard,1)).toFixed(0)+'%', 'absent 帧中 AFTER 正确不出 mask'],
  ];
  document.getElementById('stats').innerHTML = stats.map(s =>
    `<div class="stat"><div class="num">${s[1]}</div><div class="lbl">${s[0]}</div><div class="lbl">${s[2]}</div></div>`).join('');

  document.getElementById('cards').innerHTML = CASES.map(c => `
    <div class="card">
      <div class="card-head">${catBadge(c)}<b>${c.title}</b>
        <span class="query">"${c.query}"</span></div>
      <div class="body">
        <video class="vid" src="${c.video}" autoplay loop muted controls playsinline></video>
        <div class="legend">目标存在性时序（GT / BEFORE 预测 / AFTER 预测，每帧一格）</div>
        ${strip('GT', c.expected)}
        ${strip('BEFORE', c.pred_before)}
        ${strip('AFTER', c.pred_after)}
        <div class="meta">video_id=${c.video_id} · 帧数=${c.n_frames} · 每格 = 1 帧（红色=该帧应无 mask）</div>
      </div>
    </div>`).join('');
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
    jsons = sorted(glob.glob(os.path.join(args.dir, '*.json')))
    data = []
    for j in jsons:
        d = json.load(open(j))
        data.append(d)
    html = PAGE.replace('__DATA__', json.dumps(data, ensure_ascii=False))
    out = os.path.join(args.dir, 'index.html')
    with open(out, 'w') as f:
        f.write(html)
    print(f'wrote {out} with {len(data)} cases')
    print('  videos:')
    for d in data:
        vp = os.path.join(args.dir, d['video'])
        print(f"    {d['video']} ({'OK' if os.path.exists(vp) else 'MISSING'}) size={os.path.getsize(vp)//1024}KB" if os.path.exists(vp) else f"    {d['video']} MISSING")


if __name__ == '__main__':
    main()

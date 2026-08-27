"""Rebuild EvoSeg demo webpage: clean single-purpose showcase.

Per case: GT (Ref-YT-VOS annotation) vs Ours (Faithful + e_t head) vs Sa2VA
baseline. Header shows full-set metrics (Ref-YT-VOS J&F, faithfulness).
"""
import argparse, glob, json, os

PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EvoSeg · 时序忠实指代分割</title>
<style>
  :root { --bg:#0b0f14; --panel:#121821; --border:#1f2a36; --text:#e8eef5;
          --muted:#8b98a9; --accent:#4da3ff; --green:#3fb950; --red:#f85149;
          --blue:#58a6ff; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font:14px/1.55 -apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif; padding:28px 20px 60px; }
  .wrap { max-width:1080px; margin:0 auto; }
  h1 { font-size:24px; letter-spacing:.2px; }
  h1 .dot { color:var(--accent); }
  .sub { color:var(--muted); margin:6px 0 20px; font-size:13.5px; }
  .sub em { color:var(--text); font-style:normal; }
  .metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin-bottom:26px; }
  .metric { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:14px 16px; }
  .metric .k { color:var(--muted); font-size:12px; margin-bottom:6px; }
  .metric .v { font-size:20px; font-weight:700; }
  .metric .v small { color:var(--muted); font-weight:400; font-size:12px; }
  .metric .vs { color:var(--muted); font-size:12px; margin-top:4px; }
  .metric.best .v { color:var(--green); }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:14px; margin-bottom:18px; overflow:hidden; }
  .card-head { padding:12px 16px; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .badge { font-size:11px; padding:2px 10px; border-radius:20px; font-weight:600; }
  .b-temporal_absence{background:#1f6feb2a;color:#58a6ff;border:1px solid #1f6feb55}
  .b-identity_swap{background:#d2992226;color:#e3b341;border:1px solid #d2992255}
  .b-global_absence{background:#2ea04326;color:#3fb950;border:1px solid #2ea04355}
  .b-counterfactual_swap{background:#f8514926;color:#ff7b72;border:1px solid #f8514955}
  .card-head b { font-size:15px; }
  .query { color:var(--muted); font-size:12.5px; margin-left:auto; font-style:italic; }
  .body { padding:14px 16px; }
  video { width:100%; border-radius:10px; background:#000; margin-bottom:12px; }
  .row { display:flex; align-items:center; gap:10px; margin:5px 0; }
  .lbl { width:150px; flex:none; font-size:12px; color:var(--text); }
  .lbl.ours { color:var(--accent); font-weight:600; }
  .strip { display:flex; gap:2px; flex-wrap:wrap; }
  .blk { width:13px; height:18px; border-radius:2px; }
  .on { background:var(--green); } .off { background:var(--red); opacity:.85; }
  .legend { font-size:12px; color:var(--muted); margin:8px 0 2px; }
  .meta { font-size:11.5px; color:var(--muted); margin-top:10px; }
  .note { background:#1f6feb12; border:1px solid #1f6feb33; color:#79c0ff; border-radius:10px; padding:10px 14px; margin-bottom:20px; font-size:13px; }
  .note b { color:#a5d6ff; }
</style>
</head>
<body><div class="wrap">
  <h1>EvoSeg<span class="dot">·</span> 时序忠实指代分割</h1>
  <div class="sub">目标存在性是<em>逐帧谓词</em> e<sub>t</sub>。Ours = 图像级拒答（Faithful SFT，分割无损）+ 外部轻量时序存在性头（4.5M 参数）。Sa2VA 基线 = 无条件传播（目标消失后继续画）。GT = Ref-YT-VOS 官方标注。</div>
  <div class="note">✅ 全部数字为<b>全集</b>：Ref-YT-VOS valid 202 视频（J&F）+ faithfulness 基准 1986 查询（幻觉率）。</div>

  <div class="metrics" id="metrics"></div>
  <div id="cards"></div>

<script id="data" type="application/json">__DATA__</script>
<script>
const CASES = JSON.parse(document.getElementById('data').textContent);
const METRICS = [
  {k:'分割能力 (Ref-YT-VOS J&F)', v:'0.523', vs:'Sa2VA 基线 0.509', best:true, note:'分割近乎无损甚至更好'},
  {k:'整体幻觉率 (faithfulness)', v:'9.7%', vs:'Sa2VA 基线 100%', best:true, note:'目标不存在时知道不画'},
  {k:'global 幻觉', v:'1.5%', vs:'图像级拒答生效', best:true},
  {k:'counterfactual 幻觉', v:'14.9%', vs:'对抗改写拒答'},
  {k:'temporal 幻觉', v:'49.2%', vs:'hardest 类别（消失边界）'},
  {k:'identity 幻觉', v:'55.3%', vs:'换人/相似目标'},
];
function strip(a){return a.map(v=>'<span class="blk '+(v?'on':'off')+'"></span>').join('');}
document.getElementById('metrics').innerHTML = METRICS.map(m=>
  `<div class="metric${m.best?' best':''}"><div class="k">${m.k}</div><div class="v">${m.v}${m.note?' <small>· '+m.note+'</small>':''}</div>${m.vs?`<div class="vs">${m.vs}</div>`:''}</div>`).join('');
document.getElementById('cards').innerHTML = CASES.map(c=>{
  const rows = ['GT (标注)','Faithful + e_t (Ours)','Sa2VA-4B (基线)'].map(m=>{
    const d = c.data[m]; if(!d) return '';
    const ours = m.includes('Ours');
    return `<div class="row"><div class="lbl${ours?' ours':''}">${ours?'★ ':''}${m}</div><div class="strip">${strip(d.presence)}</div></div>`;
  }).join('');
  const videos = c.models.map(m=>`<source src="${c.title}/${encodeURIComponent(m)}.mp4" type="video/mp4">`).join('');
  return `<div class="card">
    <div class="card-head"><span class="badge b-${c.category}">${c.category.replace('_',' ')}</span><b>${c.title}</b><span class="query">"${c.query}"</span></div>
    <div class="body">
      <video controls muted playsinline autoplay loop>${videos}你的浏览器不支持 video。</video>
      <div class="legend">逐帧目标存在性（绿 = 应有 mask，红 = 不应有）</div>
      ${rows}
      <div class="meta">video=${c.video_id} · ${c.n_frames} 帧（每格 1 帧）· GT 来自 Ref-YT-VOS 官方标注</div>
    </div></div>`;
}).join('');
</script>
</div></body></html>
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/9950backfile/chenjiahui/evo_artifacts/data/examples')
    args = ap.parse_args()
    cases = []
    for meta_p in sorted(glob.glob(os.path.join(args.dir, '*/meta.json'))):
        meta = json.load(open(meta_p))
        data = {}
        for m in meta['models']:
            jp = os.path.join(args.dir, meta['title'], f'{m}.json')
            if os.path.exists(jp):
                data[m] = json.load(open(jp))
        cases.append({**meta, 'data': data})
    html = PAGE.replace('__DATA__', json.dumps(cases, ensure_ascii=False))
    out = os.path.join(args.dir, 'index.html')
    with open(out, 'w') as f:
        f.write(html)
    print(f'wrote {out} with {len(cases)} cases')
    for c in cases:
        print(f"  {c['title']}: models={c['models']}")

if __name__ == '__main__':
    main()

"""Live training dashboard for a campaign.

Serves a small self-contained page on localhost that polls the run
directories and draws the learning curves as they are written. train.py
writes history.json and status.json after every epoch, so a run in
progress shows up within one poll interval.

Usage:  python dashboard.py [--runs runs/linear/seed0 ...] [--port 8770]
then open http://127.0.0.1:8770/

With --runs omitted it watches every directory under runs/ (and runs-quick/)
that holds a campaign.json, i.e. every array and seed of the campaign.

The plan panel lists every run the campaign intends to make, queued ones
included, so the page answers "what is left" and not only "what is drawn".
Planned-but-not-started runs are known from the campaign.json that
run_campaign.py writes before its first run.

The focused run is drawn as the one identity-coloured series; the other
runs of the same array and seed are neutral context lines behind it,
direct-labelled. Click any row of the plan to focus it. Nothing external is
fetched, so it works offline.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Only used for --arms' default, for directories without a campaign.json.
DEFAULT_ARMS = ["raw", "vmb", "vonmises_D2", "even_D2", "odd_D2", "dual_D2", "reim",
                "vonmises-w_D2", "even-w_D2", "odd-w_D2", "dual-w_D2"]

# A run is "running" only while its files keep moving. status.json says
# running:true right up until the process dies, so a crash, a kill or a
# power cut leaves that flag set forever -- trusting it means the page
# reports a training job that stopped days ago as live. Treat the heartbeat
# as stale after a few epochs' worth of silence, floored for the case where
# epochs are quick and ceilinged for the long first epoch of a big rung
# (before secs_per_epoch exists, all we know is the run has started).
STALE_EPOCHS = 3.0
STALE_FLOOR = 120.0
STALE_UNKNOWN = 900.0

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SSL training monitor</title>
<style>
  :root{
    color-scheme:light;
    --surface-0:#f4f4f1; --surface-1:#fcfcfb; --border:#e1e0d9;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#898781;
    --series-1:#2a78d6; --context:#b8b7b0; --good:#1baf7a;
    /* status palette: fixed, never themed, never reused as a series colour */
    --st-good:#0ca30c; --st-warn:#fab219; --st-crit:#d03b3b;
    --row-hover:#efeee9;
  }
  @media (prefers-color-scheme:dark){ :root:not([data-theme=light]){
    color-scheme:dark;
    --surface-0:#121211; --surface-1:#1a1a19; --border:#33322e;
    --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#8d8b83;
    --series-1:#3987e5; --context:#5c5a53; --good:#199e70;
    --row-hover:#232322;
  }}
  *{box-sizing:border-box}
  body{margin:0;background:var(--surface-0);color:var(--text-primary);
    font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
  .wrap{max-width:1180px;margin:0 auto;padding:28px 20px 60px}
  h1{font-size:19px;margin:0 0 2px;letter-spacing:-.01em}
  .sub{color:var(--text-secondary);font-size:13px;margin-bottom:20px}
  .live{display:inline-flex;align-items:center;gap:6px;font-size:12px;
    font-weight:600;color:var(--good)}
  .dot{width:7px;height:7px;border-radius:50%;background:var(--good);
    animation:p 1.6s ease-in-out infinite}
  @keyframes p{0%,100%{opacity:1}50%{opacity:.25}}
  /* Status colour rides the glyph and the bar, never the words: amber on the
     light surface is 1.79:1, which is a fine mark and an unreadable label. */
  .halted{display:inline-flex;align-items:center;gap:5px;font-size:12px;
    font-weight:600;color:var(--text-primary)}
  .halted .g{color:var(--st-warn)}
  .tiles{display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));
    margin-bottom:18px}
  .tile{background:var(--surface-1);border:1px solid var(--border);
    border-radius:10px;padding:11px 13px}
  .tile .k{font-size:11px;color:var(--text-muted);text-transform:uppercase;
    letter-spacing:.05em;margin-bottom:3px}
  .tile .v{font-size:21px;font-weight:600;letter-spacing:-.02em;
    font-variant-numeric:tabular-nums}
  .tile .v small{font-size:12px;font-weight:400;color:var(--text-secondary)}
  .card{background:var(--surface-1);border:1px solid var(--border);
    border-radius:12px;padding:16px 18px 10px;margin-bottom:14px}
  .card h2{font-size:14px;margin:0 0 1px;font-weight:600}
  .card .cap{font-size:12px;color:var(--text-secondary);margin-bottom:8px}
  .legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;
    color:var(--text-secondary);margin:4px 0 0 4px}
  .legend span{display:inline-flex;align-items:center;gap:6px}
  .sw{width:14px;height:3px;border-radius:2px;display:inline-block}
  svg{display:block;width:100%;height:auto;overflow:visible}
  .bar{height:5px;background:var(--border);border-radius:3px;overflow:hidden;
    margin-top:8px}
  .bar>i{display:block;height:100%;background:var(--series-1);border-radius:3px}
  details{margin-top:8px}
  summary{cursor:pointer;font-size:12px;color:var(--text-secondary)}
  table{border-collapse:collapse;font-size:12px;margin-top:10px;width:100%;
    font-variant-numeric:tabular-nums}
  th,td{text-align:right;padding:3px 9px;border-bottom:1px solid var(--border)}
  th:first-child,td:first-child{text-align:left}
  th{color:var(--text-muted);font-weight:500}
  .empty{color:var(--text-secondary);padding:26px 0;text-align:center}
  /* plan panel */
  .plan{width:100%;margin-top:4px}
  .plan td,.plan th{padding:5px 9px;vertical-align:middle}
  .plan tr.run{cursor:pointer}
  .plan tr.run:hover td{background:var(--row-hover)}
  .plan tr.sel td{background:var(--row-hover)}
  .plan tr.sel td:first-child{box-shadow:inset 3px 0 0 var(--series-1)}
  .plan .rung td{padding-top:13px;border-bottom:1px solid var(--border);
    color:var(--text-muted);font-size:11px;text-transform:uppercase;
    letter-spacing:.05em}
  .plan .arm{font-weight:500}
  .plan .mini{height:4px;background:var(--border);border-radius:2px;
    overflow:hidden;width:110px;display:inline-block;vertical-align:middle}
  .plan .mini>i{display:block;height:100%;border-radius:2px}
  .chip{display:inline-flex;align-items:center;gap:5px;font-size:11px;
    font-weight:600;white-space:nowrap;color:var(--text-secondary)}
  .chip .g{font-size:10px;line-height:1}
  .foot{display:flex;flex-wrap:wrap;gap:16px;align-items:center;
    font-size:12px;color:var(--text-secondary);margin:10px 0 4px}
  .btn{font:inherit;font-size:12px;color:var(--text-secondary);cursor:pointer;
    background:var(--surface-0);border:1px solid var(--border);
    border-radius:999px;padding:2px 11px}
  .btn:hover{color:var(--text-primary)}
</style></head><body><div class="wrap">
  <h1>SSL training monitor <span id="live"></span></h1>
  <div class="sub" id="sub">connecting…</div>
  <div class="card">
    <h2>Campaign plan</h2>
    <div class="cap" id="capPlan">every run this campaign intends to make</div>
    <div id="plan"></div>
    <div class="foot" id="planFoot"></div>
    <div class="bar"><i id="planBar" style="width:0%"></i></div>
  </div>
  <div class="tiles" id="tiles"></div>
  <div class="card">
    <h2>Validation accuracy</h2>
    <div class="cap" id="cap1">by epoch</div>
    <div id="c1"></div><div class="legend" id="lg1"></div>
    <details><summary>Table view</summary><div id="t1"></div></details>
  </div>
  <div class="card">
    <h2>Training loss</h2>
    <div class="cap">cross-entropy, mean over batches</div>
    <div id="c2"></div>
  </div>
</div><script>
const $=id=>document.getElementById(id);
const cvar=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmtT=s=>{s=Math.round(s);const h=~~(s/3600),m=~~(s%3600/60);
  return h?`${h}h ${m}m`:(m?`${m}m ${s%60}s`:`${s}s`);};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let STATE=null, HOVER=null, PICK=null;   // PICK: user-chosen run key, or null = follow live

// Status colours are reserved and always ship with a glyph AND a word, so
// state is never carried by colour alone.
const SMETA={
  done:       {v:'--st-good',   g:'✓', t:'done'},
  running:    {v:'--series-1',  g:'●', t:'running'},
  interrupted:{v:'--st-warn',   g:'⚠', t:'interrupted'},
  queued:     {v:'--text-muted',g:'○', t:'queued'},
};

// Round tick values -- readers anchor on 0/5/10, never on 4.74.
function niceTicks(lo,hi,target,intOnly){
  const span=hi-lo; if(!(span>0))return [lo];
  const raw=span/target, mag=Math.pow(10,Math.floor(Math.log10(raw)));
  let step=[1,2,2.5,5,10].map(m=>m*mag).find(s=>s>=raw-1e-12)||10*mag;
  if(intOnly)step=Math.max(1,Math.round(step));
  const out=[];
  for(let v=Math.ceil(lo/step)*step;v<=hi+1e-9;v+=step)
    out.push(Math.abs(v)<1e-12?0:+v.toFixed(10));
  return out;
}

function line(el,series,opts){
  const W=1000,H=opts.h||300,P={t:14,r:74,b:34,l:52};
  const pts=series.flatMap(s=>s.data);
  if(!pts.length){el.innerHTML='<div class="empty">waiting for the first epoch…</div>';return;}
  const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);
  const x0=1,x1=Math.max(opts.xMax||0,...xs,2);
  let y0=opts.y0!==undefined?opts.y0:Math.min(...ys),y1=Math.max(...ys);
  if(y1-y0<1e-9)y1=y0+1; const pad=(y1-y0)*0.08; y1+=pad; if(opts.y0===undefined)y0-=pad;
  const X=v=>P.l+(v-x0)/(x1-x0)*(W-P.l-P.r), Y=v=>H-P.b-(v-y0)/(y1-y0)*(H-P.t-P.b);
  const ticksY=niceTicks(y0,y1,4);
  const ticksX=niceTicks(x0,x1,7,true).filter(v=>v>=x0);
  const g=[],grid=cvar('--border'),mut=cvar('--text-muted'),sec=cvar('--text-secondary');
  ticksY.forEach(t=>{g.push(`<line x1="${P.l}" x2="${W-P.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="${grid}" stroke-width="1"/>`);
    g.push(`<text x="${P.l-9}" y="${Y(t)+4}" text-anchor="end" font-size="11" fill="${mut}">${opts.fmt(t)}</text>`);});
  ticksX.forEach(t=>g.push(`<text x="${X(t)}" y="${H-P.b+17}" text-anchor="middle" font-size="11" fill="${mut}">${t}</text>`));
  g.push(`<text x="${(P.l+W-P.r)/2}" y="${H-2}" text-anchor="middle" font-size="11" fill="${sec}">epoch</text>`);
  // context series first, identity series on top
  const labels=[];
  for(const s of series){
    if(s.data.length<1)continue;
    const d=s.data.map((p,i)=>(i?'L':'M')+X(p[0])+' '+Y(p[1])).join(' ');
    g.push(`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.w||2}"
      stroke-linejoin="round" stroke-linecap="round" opacity="${s.op||1}"/>`);
    const last=s.data[s.data.length-1];
    if(s.label)labels.push({y:Y(last[1]),x:X(last[0])+7,t:s.label,
                            c:s.dim?mut:sec});
    if(s.mark){ // best-so-far marker: 2px surface ring keeps it legible on the line
      g.push(`<circle cx="${X(s.mark[0])}" cy="${Y(s.mark[1])}" r="5"
        fill="${s.color}" stroke="${cvar('--surface-1')}" stroke-width="2"/>`);}
  }
  // push overlapping end-labels apart, else dense ladders stack into mush
  labels.sort((a,b)=>a.y-b.y);
  for(let i=1;i<labels.length;i++)
    if(labels[i].y-labels[i-1].y<12)labels[i].y=labels[i-1].y+12;
  const over=labels.length?labels[labels.length-1].y-(H-P.b):0;
  if(over>0)labels.forEach(l=>l.y-=over);
  labels.forEach(l=>g.push(`<text x="${l.x}" y="${l.y+4}" font-size="11"
    fill="${l.c}">${esc(l.t)}</text>`));
  if(HOVER&&HOVER.el===el.id){
    const e=Math.max(x0,Math.min(x1,Math.round(HOVER.x)));
    const main=series.find(s=>s.main);
    const p=main&&main.data.find(q=>q[0]===e);
    if(p){g.push(`<line x1="${X(e)}" x2="${X(e)}" y1="${P.t}" y2="${H-P.b}" stroke="${mut}" stroke-width="1" stroke-dasharray="3 3"/>`);
      g.push(`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="4.5" fill="${main.color}" stroke="${cvar('--surface-1')}" stroke-width="2"/>`);
      const tx=X(p[0])>W*0.62?X(p[0])-140:X(p[0])+12;
      g.push(`<rect x="${tx}" y="${Y(p[1])-32}" width="128" height="34" rx="6"
        fill="${cvar('--surface-1')}" stroke="${grid}"/>`);
      g.push(`<text x="${tx+9}" y="${Y(p[1])-17}" font-size="11" fill="${sec}">epoch ${p[0]}</text>`);
      g.push(`<text x="${tx+9}" y="${Y(p[1])-3}" font-size="12" font-weight="600" fill="${cvar('--text-primary')}">${opts.fmt(p[1])}</text>`);}
  }
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.aria||'')}">${g.join('')}</svg>`;
  const svg=el.firstChild;
  svg.onmousemove=ev=>{const r=svg.getBoundingClientRect();
    const vx=(ev.clientX-r.left)/r.width*W;
    HOVER={el:el.id,x:x0+(vx-P.l)/(W-P.l-P.r)*(x1-x0)};draw();};
  svg.onmouseleave=()=>{HOVER=null;draw();};
}

function chip(state){
  const m=SMETA[state]||SMETA.queued;
  return `<span class="chip"><span class="g" aria-hidden="true"
    style="color:${cvar(m.v)}">${m.g}</span>${m.t}</span>`;
}

function drawPlan(){
  const s=STATE, rows=[];
  let planned=0,doneN=0,epDone=0,epTotal=0,eta=0,etaKnown=true;
  for(const c of s.campaigns){
    // basename only: --runs takes absolute paths and a full one wraps the
    // header onto two shouting uppercase lines. The whole path is the title.
    const dirShort=c.dir.split(/[\\/]/).filter(Boolean).pop()||c.dir;
    rows.push(`<tr class="rung"><td colspan="6" title="${esc(c.dir)}">${esc(c.label)}
      &middot; ${esc(dirShort)}${c.inferred?' &middot; no plan file — showing what is on disk':''}</td></tr>`);
    for(const r of c.plan){
      planned++; if(r.state==='done')doneN++;
      // Epoch checkpoints preserve progress for resumable campaigns.
      const gone=r.state==='interrupted' && !r.resumable;
      epTotal+=r.epochs||0; epDone+=gone?0:(r.epoch||0);
      if(r.state!=='done'){
        const rem=gone?(r.epochs||0):(r.epochs||0)-(r.epoch||0);
        if(c.secs_per_epoch)eta+=rem*c.secs_per_epoch; else if(rem>0)etaKnown=false;
      }
      const m=SMETA[r.state]||SMETA.queued, col=cvar(m.v);
      const pct=r.epochs?100*(r.epoch||0)/r.epochs:0;
      const sel=(r.key===focusKey())?' sel':'';
      rows.push(`<tr class="run${sel}" data-key="${esc(r.key)}" title="${esc(r.dir)}/${esc(r.arm)}">
        <td class="arm">${esc(r.arm)}</td>
        <td style="text-align:left">${chip(r.state)}</td>
        <td style="text-align:left"><span class="mini">
          <i style="width:${pct}%;background:${col}"></i></span></td>
        <td>${r.epoch||0}<span style="color:var(--text-muted)"> / ${r.epochs||'?'}</span></td>
        <td>${r.best_val_acc!=null?(100*r.best_val_acc).toFixed(2)+'%':'—'}</td>
        <td>${r.state==='running'&&r.eta_secs?fmtT(r.eta_secs)
             :r.state==='interrupted'?'stopped '+fmtT(r.age)+' ago':'—'}</td></tr>`);
    }
  }
  $('plan').innerHTML=`<table class="plan"><tr><th>arm</th><th style="text-align:left">state</th>
    <th style="text-align:left">progress</th><th>epochs</th><th>best val</th><th>eta</th></tr>
    ${rows.join('')}</table>`;
  $('plan').querySelectorAll('tr.run').forEach(tr=>tr.onclick=()=>{
    PICK=(PICK===tr.dataset.key)?null:tr.dataset.key;draw();});

  const pct=epTotal?100*epDone/epTotal:0;
  $('planBar').style.width=pct.toFixed(1)+'%';
  const bits=[`<b>${doneN} / ${planned}</b> runs complete`,
    `${epDone.toLocaleString()} / ${epTotal.toLocaleString()} epochs (${pct.toFixed(1)}%)`];
  if(eta>0)bits.push(`~${fmtT(eta)} of training left${etaKnown?'':' (+ runs with no timing yet)'}`);
  const nInt=s.campaigns.reduce((a,c)=>a+c.plan.filter(r=>r.state==='interrupted').length,0);
  $('capPlan').textContent=nInt
    ? 'interrupted runs retain progress when an epoch checkpoint is available'
    : 'every run this campaign intends to make';
  if(PICK)bits.push('<button class="btn" id="unpick">follow the live run</button>');
  $('planFoot').innerHTML=bits.map(b=>`<span>${b}</span>`).join('');
  const u=$('unpick'); if(u)u.onclick=()=>{PICK=null;draw();};
}

function focusKey(){
  const s=STATE; if(!s||!s.runs.length)return null;
  if(PICK&&s.runs.some(r=>r.key===PICK))return PICK;
  const live=s.runs.find(r=>r.state==='running');
  if(live)return live.key;
  const seen=s.runs.filter(r=>r.history.length).sort((a,b)=>b.mtime-a.mtime)[0];
  return seen?seen.key:null;
}

function draw(){
  const s=STATE; if(!s)return;
  if(s.error){$('sub').textContent='dashboard error: '+s.error;return;}
  const live=s.runs.find(r=>r.state==='running');
  const halted=!live&&s.runs.find(r=>r.state==='interrupted');
  $('live').innerHTML=live?'<span class="live"><span class="dot"></span>LIVE</span>'
    :halted?'<span class="halted"><span class="g">⚠</span>nothing training</span>':'';
  drawPlan();

  const focus=s.runs.find(r=>r.key===focusKey());
  if(!focus){
    $('sub').innerHTML=s.watching.length
      ? `watching ${s.watching.map(esc).join(', ')} — no runs have started yet`
      : 'no run directories found — pass --runs <dir> [<dir> …]';
    $('tiles').innerHTML='';
    $('c1').innerHTML='<div class="empty">no runs yet</div>';$('c2').innerHTML='';return;}

  const h=focus.history, last=h[h.length-1]||{};
  // derive best/epoch time from the history if status.json lacks them
  const st=Object.assign({},focus.status||{});
  if(st.best_val_acc===undefined&&h.length){
    const b=h.reduce((m,e)=>e.val_acc>m.val_acc?e:m,h[0]);
    st.best_val_acc=b.val_acc; st.best_epoch=b.epoch;
  }
  if(st.secs_per_epoch===undefined&&h.length)
    st.secs_per_epoch=h.slice(-5).reduce((a,e)=>a+e.secs,0)/Math.min(5,h.length);
  const note=focus.state==='running'?'<span class="live"><span class="dot"></span>live</span>'
    :focus.state==='interrupted'
      ? `<span class="halted"><span class="g">⚠</span>interrupted at epoch `
        +`${last.epoch||0}, silent for ${fmtT(focus.age)}</span>`
      : focus.state==='done'?'complete':'queued';
  $('sub').innerHTML=`<b>${esc(focus.label)} / ${esc(focus.arm||focus.name)}</b> &middot; `+
    `${(focus.n_train||0).toLocaleString()} train samples `+
    `&middot; ${note}`;
  const pct=focus.epochs?100*(last.epoch||0)/focus.epochs:0;
  const tiles=[
    ['epoch',`${last.epoch||0}<small> / ${focus.epochs||'?'}</small>`],
    ['best val',`${(100*(st.best_val_acc??0)).toFixed(2)}<small>% @${st.best_epoch??'-'}</small>`],
    ['current val',`${(100*(last.val_acc??0)).toFixed(2)}<small>%</small>`],
    ['train loss',(last.loss??0).toFixed(3)],
    // a sub-second epoch (smoke runs) must not round to a bare "0"
    ['sec / epoch',(v=>v>=10?v.toFixed(0):v.toFixed(1))(st.secs_per_epoch??last.secs??0)],
    ['eta',focus.state==='running'?fmtT(st.eta_secs||0):'—'],
  ];
  $('tiles').innerHTML=tiles.map(t=>`<div class="tile"><div class="k">${t[0]}</div>
    <div class="v">${t[1]}</div></div>`).join('')+
    `<div class="tile" style="grid-column:1/-1;padding-bottom:14px">
      <div class="k">progress</div><div class="v" style="font-size:13px">
      ${pct.toFixed(1)}% &middot; lr ${(last.lr??0).toExponential(2)}</div>
      <div class="bar"><i style="width:${pct}%"></i></div></div>`;

  const c1=cvar('--series-1'),ctx=cvar('--context');
  // context = the other runs of the same array and seed
  const ctxSeries=s.runs.filter(r=>r.dir===focus.dir&&r!==focus&&r.history.length)
    .map(r=>({data:r.history.map(e=>[e.epoch,100*e.val_acc]),color:ctx,w:1.5,
              dim:1,label:r.arm||r.name}));
  const main={data:h.map(e=>[e.epoch,100*e.val_acc]),color:c1,w:2.5,main:1,
    mark:st.best_epoch?[st.best_epoch,100*st.best_val_acc]:null};
  $('cap1').textContent=`${focus.label}: ${focus.arm||focus.name} in colour, `+
    `its sibling arms in grey; dot = best epoch`;
  line($('c1'),[...ctxSeries,main],{fmt:v=>v.toFixed(0)+'%',y0:0,
    xMax:focus.epochs,aria:'validation accuracy by epoch'});
  // one series needs no legend box -- the subtitle already names the run
  $('lg1').innerHTML=ctxSeries.length
    ? `<span><i class="sw" style="background:${c1}"></i>${esc(focus.arm||focus.name)}`+
      `${focus.state==='running'?' (live)':''}</span>`+
      `<span><i class="sw" style="background:${ctx}"></i>other arms, ${esc(focus.label)}</span>` : '';
  line($('c2'),[{data:h.map(e=>[e.epoch,e.loss]),color:c1,w:2.5,main:1}],
    {fmt:v=>v.toFixed(2),h:230,xMax:focus.epochs,aria:'training loss by epoch'});

  const rows=h.slice(-12).reverse();
  $('t1').innerHTML='<table><tr><th>epoch</th><th>val %</th><th>train loss</th>'+
    '<th>lr</th><th>sec</th></tr>'+rows.map(e=>`<tr><td>${e.epoch}</td>
    <td>${(100*e.val_acc).toFixed(2)}</td><td>${e.loss.toFixed(4)}</td>
    <td>${(e.lr??0).toExponential(1)}</td><td>${e.secs}</td></tr>`).join('')+'</table>';
}

async function tick(){
  try{const r=await fetch('/api/state',{cache:'no-store'});STATE=await r.json();draw();}
  catch(e){$('sub').textContent='dashboard server not reachable';}
}
tick();setInterval(tick,5000);
matchMedia('(prefers-color-scheme:dark)').addEventListener('change',draw);
</script></body></html>"""


def read_json(path: Path, default=None):
    """Read a JSON file, tolerating absence and mid-write garbage.

    The trainer rewrites these files every epoch, so a poll can land between
    open and flush; utf-8-sig because PowerShell writes a BOM if a runner
    script ever produces one of these by hand.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, ValueError):
        return default


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def read_run(d: Path, now: float) -> dict | None:
    """One arm directory -> its record, or None if it has not started."""
    hist_f, stat_f, res_f = (d / "history.json", d / "status.json",
                             d / "results.json")
    hist = read_json(hist_f, []) or []
    stat = read_json(stat_f, {}) or {}
    res = read_json(res_f, {}) or {}
    if not hist and not stat and not res:
        return None
    beat = max(_mtime(hist_f), _mtime(stat_f))
    age = now - beat if beat else 0.0
    # How long may the files sit still before the run is presumed dead? Once
    # epochs are landing, a few of them. Before the first one has, there is
    # no epoch time to reason from -- and that first epoch carries the CUDA
    # warm-up on top of a full pass -- so allow a lot. Getting this wrong in
    # the impatient direction is the worse failure: it declares a healthy run
    # dead in the exact minutes after launch when someone is watching.
    spe = stat.get("secs_per_epoch")
    limit = (max(STALE_FLOOR, STALE_EPOCHS * spe) if spe and hist
             else STALE_UNKNOWN)
    if res:
        state = "done"
    elif stat.get("running") and age < limit:
        state = "running"
    else:
        # Either status.json still claims to be running but nothing has
        # touched the files in far longer than an epoch takes (killed
        # process, crash, power cut), or the trainer stopped without
        # writing results.json. Both mean: started, not finished, not
        # advancing. run_campaign.py resumes such a run from last.pt.
        state = "interrupted"
    return {
        "name": d.name, "history": hist, "status": stat, "state": state,
        "age": round(age, 1),
        "arm": stat.get("arm") or res.get("arm") or d.name,
        "n_train": stat.get("n_train") or res.get("n_train"),
        "n_classes": stat.get("n_classes") or res.get("n_classes", 360),
        "epochs": stat.get("epochs") or res.get("epochs") or len(hist),
        "weight_decay": stat.get("weight_decay", res.get("weight_decay")),
        "best_val_acc": (res.get("best_val_acc")
                         if res else stat.get("best_val_acc")),
        "eta_secs": stat.get("eta_secs"),
        "mtime": beat,
    }


def plan_for(d: Path, arms_cli: list[str] | None) -> dict:
    """The runs a directory is *meant* to hold, started or not.

    campaign.json is written by the runner script before the first arm
    starts, so the queue is visible from minute one. Without it there is no
    honest way to know what was planned -- fall back to --arms if given,
    otherwise report only what exists and say so.
    """
    camp = read_json(d / "campaign.json", {}) or {}
    arms = camp.get("arms") or arms_cli
    if isinstance(arms, str):
        arms = [arms]          # PS 5.1 serialises a one-element list as a scalar
    on_disk = [p.name for p in sorted(d.iterdir()) if p.is_dir()
               and any((p / f).is_file() for f in
                       ("history.json", "status.json", "results.json", "best.pt", "last.pt"))] \
        if d.is_dir() else []
    inferred = not camp.get("arms")
    if not arms:
        arms = on_disk
    else:                      # keep planned order, append surprises
        arms = list(arms) + [a for a in on_disk if a not in arms]
    return {
        "dir": str(d), "label": camp.get("label") or d.name, "arms": arms,
        "n_train": camp.get("n_train"), "epochs": camp.get("epochs"),
        "weight_decay": camp.get("weight_decay"), "inferred": inferred,
    }


def gather(run_dirs, arms_cli=None) -> dict:
    now = time.time()
    campaigns, runs = [], []
    for d in run_dirs:
        c = plan_for(d, arms_cli)
        entries, spe = [], []
        for arm in c["arms"]:
            r = read_run(d / arm, now)
            if r is None:                       # planned, not started yet
                entries.append({
                    "arm": arm, "dir": str(d), "key": f"{d}/{arm}",
                    "state": "queued", "epoch": 0,
                    "epochs": c["epochs"], "best_val_acc": None,
                    "eta_secs": None, "age": 0})
                continue
            r |= {"dir": str(d), "label": c["label"], "key": f"{d}/{arm}"}
            if r["n_train"] is None:
                r["n_train"] = c["n_train"]
            runs.append(r)
            hist = r["history"]
            if hist:
                spe.append(sum(e["secs"] for e in hist[-5:])
                           / min(5, len(hist)))
            entries.append({
                "arm": arm, "dir": str(d), "key": r["key"],
                "state": r["state"], "epoch": hist[-1]["epoch"] if hist else 0,
                "epochs": r["epochs"] or c["epochs"],
                "best_val_acc": r["best_val_acc"], "eta_secs": r["eta_secs"],
                "age": r["age"], "resumable": (d / arm / "last.pt").is_file()})
        # per-rung epoch cost, for the campaign-level estimate: arms differ,
        # but within a rung they differ far less than rungs differ from each
        # other, so the rung mean is the honest unit to extrapolate on
        c["plan"] = entries
        c["secs_per_epoch"] = round(sum(spe) / len(spe), 2) if spe else None
        # a rung with no plan file and nothing on disk is not worth a header
        if entries:
            campaigns.append(c)
    return {"campaigns": campaigns, "runs": runs,
            "watching": [str(d) for d in run_dirs], "now": now}


def discover(root: Path) -> list[Path]:
    """Run directories to watch when --runs is not given: every campaign.json
    under runs*/ (one per array and seed)."""
    return sorted({p.parent for top in root.glob("runs*") if top.is_dir()
                   for p in top.rglob("campaign.json")})


def serve(runs, arms_cli, port: int):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api/state"):
                try:
                    state = gather(runs, arms_cli)
                except Exception as exc:        # never 500 the poller
                    state = {"error": f"{type(exc).__name__}: {exc}",
                             "campaigns": [], "runs": [], "watching": []}
                body, ctype = json.dumps(state).encode(), "application/json"
            else:
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass                      # keep the console clean for the trainer

    srv = HTTPServer(("127.0.0.1", port), H)
    # Say what is being watched and what was found: a dashboard aimed at the
    # wrong directory looks exactly like a dashboard aimed at an idle one.
    state = gather(runs, arms_cli)
    print(f"dashboard on http://127.0.0.1:{port}/", flush=True)
    for c in state["campaigns"]:
        n = sum(1 for e in c["plan"] if e["state"] != "queued")
        print(f"  {c['dir']}: {len(c['plan'])} planned, {n} started"
              + ("  (no campaign.json -- showing what is on disk)"
                 if c["inferred"] else ""), flush=True)
    if not state["campaigns"]:
        print(f"  !! nothing found in {', '.join(state['watching']) or '.'}"
              f" -- check the path, or pass --runs <dir> [<dir> ...]",
              flush=True)
    srv.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", default=None,
                    help="run directories to watch (runs/<array>/seed<seed>). "
                         "Default: every campaign under runs*/.")
    ap.add_argument("--arms", nargs="+", default=None,
                    help=f"planned arm names, for directories with no "
                         f"campaign.json (the campaign is {' '.join(DEFAULT_ARMS)})")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    runs = args.runs if args.runs is not None else discover(Path("."))
    serve(runs, args.arms, args.port)


if __name__ == "__main__":
    main()

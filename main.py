"""
Chrona API — Production FastAPI Server
Endpoints: /forecast  /simulate  /anomaly  /embed  /forecast/stream
Dashboard: GET / — full interactive UI with charts, multivariate, event injection
"""

import os, time, asyncio, numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
import torch

from chrona.inference.predict import ChronaPredictor
from chrona.models.hybrid_model import ModelConfig

app = FastAPI(title="Chrona API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=500)

_predictor: Optional[ChronaPredictor] = None

def get_predictor() -> ChronaPredictor:
    global _predictor
    if _predictor is None:
        ckpt = os.environ.get("CHRONA_CHECKPOINT", "")
        _predictor = (ChronaPredictor.from_pretrained(ckpt)
                      if ckpt and Path(ckpt).exists()
                      else ChronaPredictor.from_scratch())
    return _predictor


# ── Schemas ──────────────────────────────────────────────────────────────────

class EventInput(BaseModel):
    name: str
    magnitude: float = 1.0

class ForecastRequest(BaseModel):
    series: List[List[float]]
    timestamps: Optional[List[str]] = None
    covariates: Optional[Dict[str, List[float]]] = None
    events: Optional[List[str]] = None
    horizon: int = Field(48, ge=1, le=720)
    quantiles: List[float] = Field([0.1, 0.5, 0.9])

class SimulateRequest(BaseModel):
    base_series: List[float]
    interventions: List[Dict[str, Any]]
    horizon: int = Field(72, ge=1, le=720)
    events: Optional[List[str]] = None

class AnomalyRequest(BaseModel):
    series: List[float]
    sensitivity: float = Field(0.95, ge=0.5, le=0.999)

class EmbedRequest(BaseModel):
    text: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "chrona-1.0",
            "device": "cuda" if torch.cuda.is_available() else "cpu"}

@app.post("/forecast")
async def forecast(req: ForecastRequest, p: ChronaPredictor = Depends(get_predictor)):
    t0 = time.perf_counter()
    try:
        arr    = np.array(req.series, dtype=np.float32)
        result = p.predict(arr, horizon=req.horizon, events=req.events)
        return {
            "forecast": result.to_api_dict(),
            "metadata": {"horizon": req.horizon,
                         "latency_ms": round((time.perf_counter()-t0)*1000, 2),
                         "num_series": result.num_series,
                         "events_applied": result.events_applied}
        }
    except Exception as e:
        raise HTTPException(422, str(e))

@app.post("/simulate")
async def simulate(req: SimulateRequest, p: ChronaPredictor = Depends(get_predictor)):
    try:
        r = p.simulate(req.base_series, req.interventions, req.horizon, req.events)
        return {"base": r["base"].to_api_dict(),
                "scenario": r["scenario"].to_api_dict(),
                "delta_mean": r["delta_mean"].round(4).tolist()}
    except Exception as e:
        raise HTTPException(422, str(e))

@app.post("/anomaly")
async def anomaly(req: AnomalyRequest, p: ChronaPredictor = Depends(get_predictor)):
    try:
        df = p.detect_anomalies(req.series, req.sensitivity)
        return {"anomalies": df[df["anomaly"]].to_dict("records") if "anomaly" in df else [],
                "total_checked": len(df)}
    except Exception as e:
        raise HTTPException(422, str(e))

@app.post("/embed")
async def embed(req: EmbedRequest):
    rng = np.random.default_rng(sum(ord(c) for c in req.text) % 2**32)
    vec = rng.standard_normal(256).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-8
    return {"embedding": vec.tolist(), "dim": 256}

@app.get("/forecast/stream")
async def forecast_stream(series: str, horizon: int = 10,
                          events: str = "",
                          p: ChronaPredictor = Depends(get_predictor)):
    arr = [float(v) for v in series.split(",")]
    ev  = [e.strip() for e in events.split(",") if e.strip()] or None

    async def gen():
        for i, r in enumerate(p.stream_predict(arr, horizon=horizon, events=ev)):
            yield (f'data: {{"step":{i},"mean":{r.mean[0]:.4f},'
                   f'"p10":{r.p10()[0]:.4f},"p90":{r.p90()[0]:.4f}}}\n\n')
            await asyncio.sleep(0.04)
    return StreamingResponse(gen(), media_type="text/event-stream")


# ── Dashboard ─────────────────────────────────────────────────────────────────

DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Chrona · Forecast Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Syne:wght@600;700;800&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg:#05080f;--surface:#0c1018;--border:#1a2332;
  --accent:#00ffa3;--accent2:#00c8ff;--warn:#ff6b6b;
  --text:#e8edf5;--muted:#4a5568;--subtle:#8899aa;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:"IBM Plex Mono",monospace;overflow-x:hidden}
/* scrollbar */
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

/* LAYOUT */
.app{display:grid;grid-template-columns:300px 1fr;grid-template-rows:56px 1fr;height:100vh}

/* TOP BAR */
.topbar{grid-column:1/-1;display:flex;align-items:center;gap:16px;
  padding:0 24px;border-bottom:1px solid var(--border);
  background:rgba(5,8,15,0.9);backdrop-filter:blur(12px);z-index:10}
.logo{display:flex;align-items:center;gap:8px}
.logo-mark{width:24px;height:24px;border-radius:5px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex;align-items:center;justify-content:center;font-size:12px}
.logo-text{font-family:Syne;font-weight:800;font-size:17px;letter-spacing:-.5px}
.topbar-right{margin-left:auto;display:flex;align-items:center;gap:16px}
.badge{padding:3px 10px;border-radius:20px;font-size:10px;
  border:1px solid var(--accent);color:var(--accent);letter-spacing:.5px}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--accent);
  animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 #00ffa350}
  50%{opacity:.6;box-shadow:0 0 0 5px transparent}}

/* SIDEBAR */
.sidebar{grid-row:2;border-right:1px solid var(--border);
  overflow-y:auto;padding:20px 16px;display:flex;flex-direction:column;gap:16px;
  background:var(--surface)}
.section-label{font-size:9px;letter-spacing:2px;color:var(--muted);
  text-transform:uppercase;margin-bottom:8px;padding-bottom:6px;
  border-bottom:1px solid var(--border)}
label{font-size:11px;color:var(--subtle);display:block;margin-bottom:5px}
textarea,input,select{
  width:100%;background:var(--bg);border:1px solid var(--border);
  color:var(--text);font-family:"IBM Plex Mono",monospace;font-size:11px;
  padding:8px 10px;border-radius:5px;outline:none;
  transition:border-color .15s}
textarea:focus,input:focus,select:focus{border-color:var(--accent)}
textarea{resize:vertical;min-height:80px}
.horizon-row{display:grid;grid-template-columns:1fr 1fr;gap:8px}

/* Events */
.events-grid{display:flex;flex-wrap:wrap;gap:6px}
.event-chip{padding:5px 10px;border-radius:4px;font-size:10px;cursor:pointer;
  border:1px solid var(--border);color:var(--subtle);
  transition:all .15s;user-select:none}
.event-chip.active{border-color:var(--accent);color:var(--accent);
  background:rgba(0,255,163,.08)}
.event-chip:hover:not(.active){border-color:var(--subtle);color:var(--text)}

/* Run button */
.run-btn{
  width:100%;padding:12px;border-radius:6px;border:none;cursor:pointer;
  font-family:Syne;font-weight:700;font-size:13px;letter-spacing:.3px;
  background:var(--accent);color:#000;transition:all .15s;position:relative;overflow:hidden}
.run-btn:hover{transform:translateY(-1px);box-shadow:0 6px 24px #00ffa340}
.run-btn:active{transform:none}
.run-btn.loading{background:var(--border);color:var(--subtle);cursor:wait}
.run-btn.loading::after{content:"";position:absolute;left:-100%;top:0;
  width:100%;height:100%;
  background:linear-gradient(90deg,transparent,rgba(0,255,163,.15),transparent);
  animation:shimmer 1s infinite}
@keyframes shimmer{to{left:100%}}

/* Separator */
.sep{border:none;border-top:1px solid var(--border)}

/* MAIN */
.main{grid-row:2;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:20px}

/* Chart card */
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.card-header{padding:14px 18px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between}
.card-title{font-family:Syne;font-size:13px;font-weight:700;letter-spacing:-.3px}
.card-meta{font-size:10px;color:var(--muted)}
.chart-wrap{padding:18px;position:relative;height:280px}

/* Stats row */
.stats-row{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;
  background:var(--border)}
.stat{background:var(--surface);padding:14px 18px}
.stat-label{font-size:9px;letter-spacing:1.5px;color:var(--muted);text-transform:uppercase;margin-bottom:4px}
.stat-value{font-family:Syne;font-size:20px;font-weight:800}
.stat-value.green{color:var(--accent)}
.stat-value.blue{color:var(--accent2)}
.stat-value.red{color:var(--warn)}

/* Table */
.table-wrap{overflow-x:auto;padding:0 18px 18px}
table{width:100%;border-collapse:collapse;font-size:11px}
th{padding:8px 12px;text-align:left;font-size:9px;letter-spacing:1.5px;
  color:var(--muted);text-transform:uppercase;border-bottom:1px solid var(--border)}
td{padding:8px 12px;border-bottom:1px solid rgba(26,35,50,.5);color:var(--subtle)}
td.accent{color:var(--accent);font-weight:600}
td.up{color:#4ade80}td.down{color:var(--warn)}
tr:last-child td{border-bottom:none}
tr:hover td{background:rgba(0,255,163,.03)}

/* Empty state */
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;
  height:220px;gap:12px;color:var(--muted)}
.empty-icon{font-size:36px;opacity:.3}
.empty-text{font-size:12px;text-align:center;line-height:1.8}

/* Toast */
.toast{position:fixed;bottom:24px;right:24px;padding:12px 18px;border-radius:6px;
  font-size:12px;z-index:999;transition:all .3s;opacity:0;transform:translateY(8px);
  pointer-events:none}
.toast.show{opacity:1;transform:none}
.toast.ok{background:#00ffa320;border:1px solid var(--accent);color:var(--accent)}
.toast.err{background:#ff6b6b20;border:1px solid var(--warn);color:var(--warn)}

/* Latency */
.latency{font-size:10px;color:var(--muted);font-family:"IBM Plex Mono"}
</style>
</head>
<body>
<div class="app">

<!-- TOP BAR -->
<header class="topbar">
  <div class="logo">
    <div class="logo-mark">⏱</div>
    <span class="logo-text">chrona</span>
  </div>
  <span class="badge">v1.0</span>
  <span style="font-size:10px;color:var(--muted)">Forecast Dashboard</span>
  <div class="topbar-right">
    <div class="status-dot"></div>
    <span id="latency" class="latency">ready</span>
  </div>
</header>

<!-- SIDEBAR -->
<aside class="sidebar">

  <div>
    <div class="section-label">Time Series Input</div>
    <label>Paste values (comma or newline separated)</label>
    <textarea id="series" placeholder="10, 12, 14, 13, 16, 18, 20, 19, 22, 24...">8,9,11,10,13,14,16,15,18,19,21,20,23,25,24,27,26,29,30,32,31,34,33,36,38,37,40,41,43,42,45</textarea>
  </div>

  <div>
    <div class="section-label">Forecast Settings</div>
    <div class="horizon-row">
      <div>
        <label>Horizon (steps)</label>
        <input type="number" id="horizon" value="24" min="1" max="720"/>
      </div>
      <div>
        <label>Mode</label>
        <select id="mode">
          <option value="forecast">Forecast</option>
          <option value="simulate">Simulate</option>
          <option value="anomaly">Anomaly</option>
        </select>
      </div>
    </div>
  </div>

  <div>
    <div class="section-label">Event Conditioning</div>
    <div class="events-grid" id="events-grid">
      <div class="event-chip" data-event="black_friday">Black Friday</div>
      <div class="event-chip" data-event="rate_hike">Rate Hike</div>
      <div class="event-chip" data-event="rate_cut">Rate Cut</div>
      <div class="event-chip" data-event="holiday">Holiday</div>
      <div class="event-chip" data-event="storm">Storm</div>
      <div class="event-chip" data-event="promotion">Promo</div>
      <div class="event-chip" data-event="supply_shock">Supply Shock</div>
      <div class="event-chip" data-event="demand_spike">Demand Spike</div>
    </div>
  </div>

  <hr class="sep"/>

  <button class="run-btn" id="run-btn" onclick="runForecast()">
    Run Forecast →
  </button>

  <div>
    <div class="section-label">Quick Load</div>
    <div style="display:flex;flex-direction:column;gap:6px">
      <div class="event-chip" style="cursor:pointer" onclick="loadPreset('trend')">↗ Upward trend</div>
      <div class="event-chip" style="cursor:pointer" onclick="loadPreset('seasonal')">〜 Seasonal wave</div>
      <div class="event-chip" style="cursor:pointer" onclick="loadPreset('noisy')">⚡ Noisy signal</div>
      <div class="event-chip" style="cursor:pointer" onclick="loadPreset('multi')">⊞ Multivariate</div>
    </div>
  </div>

</aside>

<!-- MAIN CONTENT -->
<main class="main">

  <!-- Stats -->
  <div class="card" id="stats-card" style="display:none">
    <div class="stats-row">
      <div class="stat"><div class="stat-label">Forecast Mean</div><div class="stat-value green" id="s-mean">—</div></div>
      <div class="stat"><div class="stat-label">P90 Peak</div><div class="stat-value blue" id="s-p90">—</div></div>
      <div class="stat"><div class="stat-label">P10 Floor</div><div class="stat-value" id="s-p10">—</div></div>
      <div class="stat"><div class="stat-label">Avg Uncertainty</div><div class="stat-value" id="s-std">—</div></div>
    </div>
  </div>

  <!-- Chart -->
  <div class="card">
    <div class="card-header">
      <span class="card-title">Probabilistic Forecast</span>
      <span class="card-meta" id="chart-meta">awaiting input</span>
    </div>
    <div class="chart-wrap">
      <div class="empty" id="chart-empty">
        <div class="empty-icon">⏱</div>
        <div class="empty-text">Configure your series in the sidebar<br/>and click <strong>Run Forecast</strong></div>
      </div>
      <canvas id="forecast-chart" style="display:none"></canvas>
    </div>
  </div>

  <!-- Table -->
  <div class="card" id="table-card" style="display:none">
    <div class="card-header">
      <span class="card-title">Forecast Table</span>
      <span class="card-meta" id="table-meta"></span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>#</th><th>Timestamp</th><th>P10</th><th>P50</th><th>P90</th><th>Mean</th><th>Δ vs prev</th></tr>
        </thead>
        <tbody id="forecast-tbody"></tbody>
      </table>
    </div>
  </div>

</main>
</div>

<!-- Toast -->
<div class="toast" id="toast"></div>

<script>
// ── State ─────────────────────────────────────────────────────────────────
let chartInstance = null;
let activeEvents = new Set();
let lastResult = null;

// ── Event chips ──────────────────────────────────────────────────────────
document.querySelectorAll('.event-chip[data-event]').forEach(chip => {
  chip.addEventListener('click', () => {
    const ev = chip.dataset.event;
    if (activeEvents.has(ev)) { activeEvents.delete(ev); chip.classList.remove('active'); }
    else                       { activeEvents.add(ev);    chip.classList.add('active'); }
  });
});

// ── Presets ───────────────────────────────────────────────────────────────
function loadPreset(type) {
  const ta = document.getElementById('series');
  const presets = {
    trend:    Array.from({length:40}, (_,i) => +(5 + i*0.8 + (Math.random()-.5)*1.5).toFixed(2)),
    seasonal: Array.from({length:48}, (_,i) => +(20 + 8*Math.sin(i*Math.PI/6) + (Math.random()-.5)*2).toFixed(2)),
    noisy:    Array.from({length:35}, (_,i) => +(15 + (Math.random()-.5)*12).toFixed(2)),
    multi:    Array.from({length:30}, (_,i) =>
                [+(10+i*0.5+(Math.random()-.5)).toFixed(2),
                 +(20-i*0.2+(Math.random()-.5)*2).toFixed(2)].join(' ')
                ).join('\n'),
  };
  ta.value = type === 'multi' ? presets.multi : presets[type].join(',');
  showToast(`Loaded ${type} preset`, 'ok');
}

// ── Parse series ─────────────────────────────────────────────────────────
function parseSeries(raw) {
  const lines = raw.trim().split(/\n+/);
  if (lines.length > 1 && lines[0].includes(' ')) {
    // multivariate: space-separated columns
    return lines.map(l => l.trim().split(/\s+/).map(Number));
  }
  const flat = raw.replace(/\n/g, ',').split(',').map(s => s.trim()).filter(Boolean).map(Number);
  return flat.map(v => [v]);
}

// ── Run ───────────────────────────────────────────────────────────────────
async function runForecast() {
  const btn = document.getElementById('run-btn');
  btn.classList.add('loading');
  btn.textContent = 'Running...';

  const raw     = document.getElementById('series').value;
  const horizon = parseInt(document.getElementById('horizon').value) || 24;
  const mode    = document.getElementById('mode').value;
  const series  = parseSeries(raw);
  const events  = [...activeEvents];
  const t0      = performance.now();

  try {
    let url, body;
    if (mode === 'forecast') {
      url  = '/forecast';
      body = { series, horizon, events: events.length ? events : null };
    } else if (mode === 'anomaly') {
      url  = '/anomaly';
      body = { series: series.map(r => r[0]) };
    } else {
      url  = '/simulate';
      body = { base_series: series.map(r=>r[0]), horizon, events,
               interventions:[{type:'scale',factor:1.15}] };
    }

    const res  = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const ms   = (performance.now()-t0).toFixed(0);
    document.getElementById('latency').textContent = `${ms}ms`;

    if (mode === 'forecast') renderForecast(data, series, horizon, ms);
    else if (mode === 'anomaly') renderAnomaly(data, series, ms);
    else renderSimulate(data, series, ms);

    showToast(`Forecast complete — ${ms}ms`, 'ok');
  } catch(e) {
    showToast('Error: ' + e.message, 'err');
    console.error(e);
  } finally {
    btn.classList.remove('loading');
    btn.textContent = 'Run Forecast →';
  }
}

// ── Render forecast ───────────────────────────────────────────────────────
function renderForecast(data, inputSeries, horizon, ms) {
  const fc     = data.forecast;
  const histVals = inputSeries.map(r => r[0]);
  const histLen  = Math.min(histVals.length, 60);
  const hist     = histVals.slice(-histLen);

  const labels = [
    ...Array.from({length:histLen}, (_,i) => `t-${histLen-i}`),
    ...Array.from({length:horizon}, (_,i) => `t+${i+1}`),
  ];

  const histData = [...hist, ...Array(horizon).fill(null)];
  const meanData = [...Array(histLen).fill(null), ...fc.mean];
  const p10Data  = [...Array(histLen).fill(null), ...fc.p10];
  const p90Data  = [...Array(histLen).fill(null), ...fc.p90];

  // Stats
  document.getElementById('stats-card').style.display = '';
  document.getElementById('s-mean').textContent = avg(fc.mean).toFixed(2);
  document.getElementById('s-p90').textContent  = Math.max(...fc.p90).toFixed(2);
  document.getElementById('s-p10').textContent  = Math.min(...fc.p10).toFixed(2);
  document.getElementById('s-std').textContent  = avg(fc.std).toFixed(3);

  document.getElementById('chart-meta').textContent = `${horizon} steps · ${ms}ms · ${data.metadata?.events_applied?.length||0} events`;

  drawChart(labels, histData, meanData, p10Data, p90Data);
  renderTable(fc, horizon);
  lastResult = data;
}

function avg(arr) { return arr.reduce((a,b)=>a+b,0)/arr.length; }

function drawChart(labels, histData, meanData, p10Data, p90Data) {
  document.getElementById('chart-empty').style.display = 'none';
  const canvas = document.getElementById('forecast-chart');
  canvas.style.display = '';
  if (chartInstance) chartInstance.destroy();

  const ctx = canvas.getContext('2d');
  const gradFill = ctx.createLinearGradient(0,0,canvas.width,0);
  gradFill.addColorStop(0,'rgba(0,255,163,0)');
  gradFill.addColorStop(.5,'rgba(0,255,163,0.12)');
  gradFill.addColorStop(1,'rgba(0,200,255,0.08)');

  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label:'P90', data:p90Data, borderColor:'rgba(0,200,255,0.3)',
          borderWidth:1, borderDash:[4,3], pointRadius:0, fill:'+1',
          backgroundColor:'rgba(0,200,255,0.06)', tension:.4 },
        { label:'P50 (mean)', data:meanData, borderColor:'#00ffa3',
          borderWidth:2.5, pointRadius:0, tension:.4, fill:false,
          shadowBlur:12, shadowColor:'#00ffa380' },
        { label:'P10', data:p10Data, borderColor:'rgba(0,200,255,0.3)',
          borderWidth:1, borderDash:[4,3], pointRadius:0, fill:false, tension:.4 },
        { label:'History', data:histData, borderColor:'rgba(120,140,180,0.6)',
          borderWidth:1.5, pointRadius:0, tension:.4, fill:false },
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:false, animation:{duration:600},
      interaction:{mode:'index',intersect:false},
      scales:{
        x:{ grid:{color:'rgba(255,255,255,0.04)',drawBorder:false},
            ticks:{color:'#4a5568',font:{family:'IBM Plex Mono',size:9},maxTicksLimit:12} },
        y:{ grid:{color:'rgba(255,255,255,0.04)',drawBorder:false},
            ticks:{color:'#4a5568',font:{family:'IBM Plex Mono',size:9}} }
      },
      plugins:{
        legend:{labels:{color:'#8899aa',font:{family:'IBM Plex Mono',size:10},
          boxWidth:12,usePointStyle:true}},
        tooltip:{backgroundColor:'#0c1018',borderColor:'#1a2332',borderWidth:1,
          titleColor:'#e8edf5',bodyColor:'#8899aa',
          titleFont:{family:'Syne',size:12},bodyFont:{family:'IBM Plex Mono',size:11}}
      }
    }
  });
}

function renderTable(fc, horizon) {
  const tbody = document.getElementById('forecast-tbody');
  tbody.innerHTML = '';
  document.getElementById('table-card').style.display = '';
  document.getElementById('table-meta').textContent = `${horizon} rows`;
  fc.mean.slice(0, 20).forEach((m, i) => {
    const delta = i === 0 ? 0 : m - fc.mean[i-1];
    const cls   = delta > 0.005 ? 'up' : delta < -0.005 ? 'down' : '';
    tbody.innerHTML += `<tr>
      <td class="accent">${i+1}</td>
      <td>${fc.timestamps?.[i] ?? 't+'+(i+1)}</td>
      <td>${fc.p10[i].toFixed(3)}</td>
      <td style="color:var(--accent2)">${fc.p50[i].toFixed(3)}</td>
      <td style="color:var(--accent)">${fc.p90[i].toFixed(3)}</td>
      <td>${m.toFixed(3)}</td>
      <td class="${cls}">${i===0?'—':(delta>0?'+':'')+delta.toFixed(3)}</td>
    </tr>`;
  });
  if (horizon > 20) tbody.innerHTML += `<tr><td colspan="7" style="color:var(--muted);text-align:center">… ${horizon-20} more rows</td></tr>`;
}

function renderAnomaly(data, inputSeries, ms) {
  const hist = inputSeries.map(r=>r[0]).slice(-60);
  const anomalies = data.anomalies || [];
  const labels = hist.map((_,i)=>'t-'+(hist.length-i));
  const histData = hist;
  const anomalyData = Array(hist.length).fill(null);
  anomalies.forEach(a => { if(a.idx < hist.length) anomalyData[a.idx] = hist[a.idx]; });
  document.getElementById('stats-card').style.display = '';
  document.getElementById('s-mean').textContent = anomalies.length;
  document.getElementById('s-mean').className = 'stat-value' + (anomalies.length>0?' red':' green');
  document.getElementById('s-p90').textContent = data.total_checked||'—';
  document.getElementById('s-p10').textContent = '—';
  document.getElementById('s-std').textContent = ms+'ms';
  document.getElementById('chart-meta').textContent = `${anomalies.length} anomalies detected`;
  document.getElementById('chart-empty').style.display = 'none';
  const canvas = document.getElementById('forecast-chart');
  canvas.style.display='';
  if(chartInstance) chartInstance.destroy();
  chartInstance = new Chart(canvas.getContext('2d'),{
    type:'line', data:{labels, datasets:[
      {label:'Signal',data:histData,borderColor:'rgba(120,140,180,0.6)',borderWidth:1.5,pointRadius:0,tension:.4},
      {label:'Anomaly',data:anomalyData,type:'scatter',backgroundColor:'#ff6b6b',pointRadius:6,borderColor:'#ff6b6b'},
    ]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:400},
      scales:{x:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#4a5568',font:{size:9}}},
              y:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#4a5568',font:{size:9}}}},
      plugins:{legend:{labels:{color:'#8899aa',font:{size:10}}},
               tooltip:{backgroundColor:'#0c1018',borderColor:'#1a2332',borderWidth:1,titleColor:'#e8edf5',bodyColor:'#8899aa'}}}
  });
}

function renderSimulate(data, inputSeries, ms) {
  const base     = data.base.mean;
  const scenario = data.scenario.mean;
  const horizon  = base.length;
  const labels   = Array.from({length:horizon},(_,i)=>'t+'+(i+1));
  document.getElementById('stats-card').style.display='';
  document.getElementById('s-mean').textContent = avg(scenario).toFixed(2);
  document.getElementById('s-p90').textContent  = Math.max(...scenario).toFixed(2);
  document.getElementById('s-p10').textContent  = Math.min(...base).toFixed(2);
  document.getElementById('s-std').textContent  = avg(data.delta_mean.map(Math.abs)).toFixed(3);
  document.getElementById('chart-meta').textContent = `Simulation: base vs scenario · ${ms}ms`;
  document.getElementById('chart-empty').style.display='none';
  const canvas = document.getElementById('forecast-chart');
  canvas.style.display='';
  if(chartInstance) chartInstance.destroy();
  chartInstance = new Chart(canvas.getContext('2d'),{
    type:'line', data:{labels, datasets:[
      {label:'Base', data:base, borderColor:'rgba(120,140,180,.7)', borderWidth:1.5, pointRadius:0, tension:.4, borderDash:[5,4]},
      {label:'Scenario (+15%)', data:scenario, borderColor:'#00ffa3', borderWidth:2.5, pointRadius:0, tension:.4},
    ]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:500},
      scales:{x:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#4a5568',font:{size:9}}},
              y:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#4a5568',font:{size:9}}}},
      plugins:{legend:{labels:{color:'#8899aa',font:{size:10}}},
               tooltip:{backgroundColor:'#0c1018',borderColor:'#1a2332',borderWidth:1,titleColor:'#e8edf5',bodyColor:'#8899aa'}}}
  });
}

// ── Toast ─────────────────────────────────────────────────────────────────
function showToast(msg, type='ok') {
  const t = document.getElementById('toast');
  t.textContent = msg; t.className = `toast ${type} show`;
  setTimeout(() => t.classList.remove('show'), 2800);
}

// ── Keyboard shortcut ────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if ((e.metaKey||e.ctrlKey) && e.key==='Enter') runForecast();
});
</script>
</body>
</html>'''

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML

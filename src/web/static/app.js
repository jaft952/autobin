/* AutoBin dashboard — React (UMD) + htm, no build step.
   Runs on Windows (web/dashboard.py) or served by the Pi itself; either way
   the browser talks DIRECTLY to the robot server (web/server.py on the Pi):
     - /api/events  SSE push: status ~5 Hz, log lines as they happen
     - POST buttons: single round-trip commands
     - /api/camera/stream: MJPEG
   The Pi address comes from ?pi=..., is remembered in localStorage, and can
   be changed in the header connect bar. */
"use strict";

const { useState, useEffect, useRef, useCallback } = React;
const html = htm.bind(React.createElement);

/* ── Pi address handling ─────────────────────────────────────────────── */

function normalizeAddr(a) {
  a = (a || "").trim();
  if (!a) return "";
  if (!/^https?:\/\//.test(a)) a = "http://" + a;
  return a.replace(/\/+$/, "");
}

function initialApiBase() {
  const fromUrl = new URLSearchParams(location.search).get("pi");
  if (fromUrl) {
    localStorage.setItem("pi_addr", fromUrl);
    return normalizeAddr(fromUrl);
  }
  const saved = localStorage.getItem("pi_addr");
  if (saved) return normalizeAddr(saved);
  // Served by the Pi's own Flask (single-machine mode): same origin works.
  if (location.protocol.startsWith("http")) return location.origin;
  return "";
}

/* ── API helpers ─────────────────────────────────────────────────────── */

async function apiGet(base, path) {
  const r = await fetch(base + path);
  return r.json();
}
async function apiPost(base, path, body) {
  const r = await fetch(base + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return r.json();
}

const fmtUptime = (s) => {
  if (s == null) return "--";
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = s % 60;
  return (h ? h + "h " : "") + m + "m " + String(ss).padStart(2, "0") + "s";
};
const dash = (v, suffix = "") => (v == null ? "--" : v + suffix);

/* ── Header: connect bar + system controls ───────────────────────────── */

function Header({ base, connected, onConnect, status, onDead, onRestarting }) {
  const state = connected && status ? status.state : "OFFLINE";
  const [addr, setAddr] = useState(base.replace(/^https?:\/\//, ""));
  const [confirmOff, setConfirmOff] = useState(false);
  const [confirmRestart, setConfirmRestart] = useState(false);

  const connect = (e) => {
    e.preventDefault();
    const norm = normalizeAddr(addr);
    if (norm) {
      localStorage.setItem("pi_addr", norm.replace(/^https?:\/\//, ""));
      onConnect(norm);
    }
  };

  const shutdown = async () => {
    if (!confirmOff) {
      setConfirmOff(true);
      setTimeout(() => setConfirmOff(false), 3000);
      return;
    }
    await apiPost(base, "/api/system/shutdown");
    onDead("Pi powering off...");
  };

  const restart = async () => {
    if (!confirmRestart) {
      setConfirmRestart(true);
      setTimeout(() => setConfirmRestart(false), 3000);
      return;
    }
    setConfirmRestart(false);
    await apiPost(base, "/api/system/restart");
    onRestarting();
  };

  return html`
    <div class="header">
      <div class="brand">AutoBin<small>control</small></div>
      <form class="conn-bar" onSubmit=${connect}>
        <span class="dot ${connected ? "on" : "off"}"></span>
        <input value=${addr} placeholder="pi address, e.g. 192.168.137.50:8000"
               onChange=${(e) => setAddr(e.target.value)} />
        <button type="submit">Connect</button>
      </form>
      <span class="badge ${state}">${state}</span>
      <div class="spacer"></div>

      <button class="btn-start" disabled=${!connected || state === "AUTO"}
              onClick=${() => apiPost(base, "/api/system/start")}>▶ START</button>
      <button class="btn-scan" disabled=${!connected || state === "SCAN"}
              onClick=${() => apiPost(base, "/api/system/scan")}>⌕ SCAN ONLY</button>
      <button class="btn-estop" disabled=${!connected}
              onClick=${() => apiPost(base, "/api/system/estop")}>■ EMERGENCY STOP</button>

      <button class="btn-restart ${confirmRestart ? "armed" : ""}"
              disabled=${!connected} onClick=${restart}>
        ${confirmRestart ? "Confirm?" : "⟲ Restart server"}
      </button>
      <button class="btn-shutdown ${confirmOff ? "armed" : ""}"
              disabled=${!connected} onClick=${shutdown}>
        ${confirmOff ? "Confirm?" : "⏻ Shutdown"}
      </button>
    </div>`;
}

/* ── Camera ──────────────────────────────────────────────────────────── */

function CameraCard({ base, status, camEpoch }) {
  const hasCam = status && status.camera;
  return html`
    <div class="card">
      <div class="card-title">CAMERA VIEW
        <span class="right">${hasCam ? "live · YOLO overlay" : "offline"}</span>
      </div>
      <div class="cam-wrap">
        ${hasCam
          ? html`<img key=${camEpoch} src="${base}/api/camera/stream?e=${camEpoch}" alt="camera" />`
          : html`<div class="cam-off">no camera on this run<br/>
                   <small>(--no-camera, or webcam/YOLO failed — see log)</small>
                 </div>`}
        ${status && status.message
          ? html`<div class="cam-overlay">[L${status.layer}] ${status.message}</div>` : null}
      </div>
    </div>`;
}

/* ── Power + performance ─────────────────────────────────────────────── */

function Bar({ pct }) {
  if (pct == null) return null;
  const cls = pct > 90 ? "crit" : pct > 70 ? "warn" : "";
  return html`<div class="bar"><i class=${cls} style=${{ width: pct + "%" }}></i></div>`;
}

function PowerCard({ s }) {
  return html`
    <div class="card">
      <div class="card-title">POWER STATUS</div>
      <div class="card-body stat-grid">
        <div class="stat">
          <div class="k">CPU TEMP</div>
          <div class="v">${dash(s && s.temp_c, " °C")}</div>
        </div>
        <div class="stat">
          <div class="k">CPU LOAD</div>
          <div class="v">${dash(s && s.cpu_pct, "%")}</div>
          <${Bar} pct=${s && s.cpu_pct} />
        </div>
        <div class="stat">
          <div class="k">MEMORY</div>
          <div class="v">${dash(s && s.mem_pct, "%")}</div>
          <${Bar} pct=${s && s.mem_pct} />
        </div>
      </div>
    </div>`;
}

function PerfCard({ s }) {
  return html`
    <div class="card">
      <div class="card-title">PERFORMANCE</div>
      <div class="card-body stat-grid">
        <div class="stat"><div class="k">LOOP RATE</div>
          <div class="v">${dash(s && s.loop_hz)} <small>Hz</small></div></div>
        <div class="stat"><div class="k">TICK TIME</div>
          <div class="v">${dash(s && s.tick_ms)} <small>ms</small></div></div>
        <div class="stat"><div class="k">ULTRASONIC</div>
          <div class="v">${dash(s && s.distance_cm)} <small>cm</small></div></div>
        <div class="stat"><div class="k">UPTIME</div>
          <div class="v">${fmtUptime(s && s.uptime_s)}</div></div>
      </div>
    </div>`;
}

/* ── Manual arm ──────────────────────────────────────────────────────── */

const CH_LABELS = ["CH1 base", "CH2 shoulder", "CH3 elbow", "CH4 wrist", "CH5 roll", "CH6 grip"];

function ArmCard({ base, status, connected }) {
  const [pose, setPose] = useState(null);
  const [err, setErr] = useState("");
  const enabled = connected && status && status.arm
    && (status.state === "STOPPED" || status.state === "ESTOP");

  const refresh = useCallback(async () => {
    if (connected && status && status.arm) {
      try {
        const p = await apiGet(base, "/api/arm/pose");
        if (p.ok) setPose(p);
      } catch (e) { /* offline */ }
    }
  }, [base, connected, status && status.arm]);
  useEffect(() => { refresh(); }, [refresh, status && status.state]);

  const act = async (path, body) => {
    setErr("");
    try {
      const r = await apiPost(base, path, body);
      if (!r.ok) setErr(r.error || "failed");
      else setPose(r);
    } catch (e) { setErr("request failed"); }
  };

  const angles = pose ? [...pose.arm, pose.gripper] : null;

  return html`
    <div class="card">
      <div class="card-title">MANUAL ARM CONTROL
        <span class="right">${status && status.arm ? "" : "no arm"}</span>
      </div>
      <div class="card-body">
        <div class="arm-actions">
          <button disabled=${!enabled} onClick=${() => act("/api/arm/pose", { name: "home" })}>Home</button>
          <button disabled=${!enabled} onClick=${() => act("/api/arm/pose", { name: "bin" })}>Bin pose</button>
          <button disabled=${!enabled} onClick=${() => act("/api/arm/gripper", { action: "open" })}>Grip open</button>
          <button disabled=${!enabled} onClick=${() => act("/api/arm/gripper", { action: "close" })}>Grip close</button>
        </div>
        ${CH_LABELS.map((label, ch) => html`
          <div class="jog-row" key=${ch}>
            <span class="ch">${label}</span>
            <button disabled=${!enabled} onClick=${() => act("/api/arm/jog", { channel: ch, delta: -5 })}>-5</button>
            <button disabled=${!enabled} onClick=${() => act("/api/arm/jog", { channel: ch, delta: -1 })}>-1</button>
            <span class="val">${angles ? angles[ch].toFixed(1) + "°" : "--"}</span>
            <button disabled=${!enabled} onClick=${() => act("/api/arm/jog", { channel: ch, delta: +1 })}>+1</button>
            <button disabled=${!enabled} onClick=${() => act("/api/arm/jog", { channel: ch, delta: +5 })}>+5</button>
          </div>`)}
        ${!enabled && connected && status && status.arm
          ? html`<div class="arm-hint">manual control needs the base halted (STOPPED or ESTOP)</div>` : null}
        ${err ? html`<div class="arm-hint">${err}</div>` : null}
      </div>
    </div>`;
}

/* ── Quick settings ──────────────────────────────────────────────────── */

function SettingsCard({ base, connected }) {
  const [settings, setSettings] = useState([]);
  const [draft, setDraft] = useState({});

  const load = useCallback(async () => {
    if (!connected) return;
    try {
      const r = await apiGet(base, "/api/settings");
      if (r.ok) { setSettings(r.settings); setDraft({}); }
    } catch (e) { /* offline */ }
  }, [base, connected]);
  useEffect(() => { load(); }, [load]);

  const commit = async (s) => {
    const v = draft[s.key];
    if (v === undefined || v === "" || Number(v) === s.value) return;
    try { await apiPost(base, "/api/settings", { key: s.key, value: Number(v) }); } catch (e) {}
    load();
  };

  // Section by owning layer, in the order the backend already returns them.
  const groups = [];
  for (const s of settings) {
    let g = groups.find((x) => x.name === s.group);
    if (!g) { g = { name: s.group, items: [] }; groups.push(g); }
    g.items.push(s);
  }

  return html`
    <div class="card">
      <div class="card-title">QUICK SETTINGS
        <span class="right"><button onClick=${load}>↻</button></span>
      </div>
      <div class="card-body">
        ${groups.map((g) => html`
          <div class="set-group" key=${g.name}>
            <div class="set-group-title">${g.name}</div>
            ${g.items.map((s) => html`
              <div class="set-row" key=${s.key}>
                <label title=${s.key}>${s.label}</label>
                <input type="number" min=${s.min} max=${s.max} step=${s.step}
                       value=${draft[s.key] !== undefined ? draft[s.key] : s.value}
                       onChange=${(e) => setDraft({ ...draft, [s.key]: e.target.value })}
                       onBlur=${() => commit(s)}
                       onKeyDown=${(e) => e.key === "Enter" && e.target.blur()} />
              </div>`)}
          </div>`)}
        <div class="set-note">applied live, session-only — edit the source constants to keep them</div>
      </div>
    </div>`;
}

/* ── Log panel (entries arrive via SSE, held by App) ─────────────────── */

function LogPanel({ base, lines, onClear }) {
  const [filter, setFilter] = useState("ALL");
  const [follow, setFollow] = useState(true);
  const bodyRef = useRef(null);

  useEffect(() => {
    if (follow && bodyRef.current)
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
  }, [lines, follow]);

  const clear = async () => {
    try { await apiPost(base, "/api/logs/clear"); } catch (e) {}
    onClear();
  };

  const shown = filter === "ALL" ? lines : lines.filter((l) => l.level === filter);

  return html`
    <div class="card">
      <div class="card-title">LOG
        <span class="right log-controls">
          <select value=${filter} onChange=${(e) => setFilter(e.target.value)}>
            <option>ALL</option><option>DEBUG</option><option>INFO</option>
            <option>SUCCESS</option><option>WARNING</option><option>FAIL</option><option>ERROR</option>
          </select>
          <button onClick=${() => setFollow(!follow)}>${follow ? "⏸ pause" : "▶ follow"}</button>
          <button onClick=${clear}>clear</button>
        </span>
      </div>
      <div class="log-body" ref=${bodyRef}
           onScroll=${(e) => {
             const el = e.target;
             const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 30;
             if (!atBottom && follow) setFollow(false);
           }}>
        ${shown.map((l) => html`
          <div class="log-line ${l.level}" key=${l.seq}>
            <span class="ts">${l.ts}</span><span class="src">${l.source}</span>${l.msg}
          </div>`)}
      </div>
    </div>`;
}

/* ── App: owns the SSE connection ────────────────────────────────────── */

function App() {
  const [base, setBase] = useState(initialApiBase());
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState(null);
  const [lines, setLines] = useState([]);
  const [camEpoch, setCamEpoch] = useState(0);
  const [dead, setDead] = useState("");
  const [restarting, setRestarting] = useState(false);
  const lastSeqRef = useRef(0);   // highest log seq seen, so a reconnect asks for only what it missed

  useEffect(() => {
    if (!base) return;
    lastSeqRef.current = 0;   // switching servers (Connect) — nothing carries over
    const es = new EventSource(`${base}/api/events?after=${lastSeqRef.current}`);
    es.onopen = () => {
      setConnected(true);
      setRestarting(false);            // server is back — drop the banner
      setCamEpoch((n) => n + 1);       // (re)start the MJPEG <img>
    };
    es.onerror = () => setConnected(false);   // EventSource auto-retries
    es.addEventListener("status", (e) => setStatus(JSON.parse(e.data)));
    es.addEventListener("logs", (e) => {
      const entries = JSON.parse(e.data);
      if (entries.length) lastSeqRef.current = entries[entries.length - 1].seq;
      setLines((old) => [...old, ...entries].slice(-800));
    });
    return () => es.close();
  }, [base]);

  return html`
    <${Header} base=${base} connected=${connected} status=${status}
               onConnect=${(b) => { setLines([]); setStatus(null); setBase(b); }}
               onDead=${setDead} onRestarting=${() => setRestarting(true)} />
    ${dead ? html`<div class="dead-overlay">${dead}</div>` : null}
    ${!dead && restarting
      ? html`<div class="conn-banner">server restarting, re-reads edited source…</div>` : null}
    ${!dead && !restarting && base && !connected
      ? html`<div class="conn-banner">connecting to ${base} … is web/server.py running on the Pi?</div>` : null}
    ${!base
      ? html`<div class="conn-banner">enter the Pi address above and press Connect</div>` : null}
    <div class="cols">
      <div class="col">
        <${CameraCard} base=${base} status=${connected ? status : null} camEpoch=${camEpoch} />
        <${LogPanel} base=${base} lines=${lines} onClear=${() => setLines([])} />
      </div>
      <div class="col">
        <${PowerCard} s=${connected ? status : null} />
        <${PerfCard} s=${connected ? status : null} />
        <${ArmCard} base=${base} status=${status} connected=${connected} />
        <${SettingsCard} base=${base} connected=${connected} />
      </div>
    </div>`;
}

ReactDOM.createRoot(document.getElementById("root")).render(html`<${App} />`);

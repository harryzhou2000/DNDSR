(() => {
  "use strict";

  const PALETTE = [
    "#38bdf8", "#f97316", "#22c55e", "#e879f9",
    "#facc15", "#14b8a6", "#f43f5e", "#a78bfa",
  ];

  const state = {
    runs: [],
    selectedRunId: null,
    detail: null,
    history: null,
    boundaries: null,
    domain: 0,
    bcName: null,
    tab: "charts",
    verbose: false,
    live: true,
    timer: null,
    dirtyOutput: {},
    dirtySteering: {},
    renderedSchemaKey: null,
  };

  const $ = (id) => document.getElementById(id);

  function fmtNum(v, digits = 3) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
    const n = Number(v);
    if (n === 0) return "0";
    const a = Math.abs(n);
    if (a >= 1e4 || a < 1e-2) return n.toExponential(digits);
    return n.toPrecision(digits + 1);
  }

  function fmtWall(sec) {
    if (sec === null || sec === undefined) return "—";
    const s = Number(sec);
    if (!Number.isFinite(s)) return "—";
    if (s < 60) return `${s.toFixed(1)} s`;
    if (s < 3600) return `${Math.floor(s / 60)}m ${(s % 60).toFixed(0)}s`;
    return `${(s / 3600).toFixed(2)} h`;
  }

  function statusClass(st) {
    if (!st) return "";
    if (st === "running") return "status-running";
    if (st === "stopping") return "status-stopping";
    if (st === "steady-converged" || st === "target-reached") return "status-converged";
    return "status-graceful-stop";
  }

  function toast(msg, isError = false) {
    const el = $("toast");
    el.textContent = msg;
    el.style.borderColor = isError ? "var(--danger)" : "var(--accent)";
    el.hidden = false;
    clearTimeout(el._timer);
    el._timer = setTimeout(() => { el.hidden = true; }, 3800);
  }

  async function api(url, opts = {}) {
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
      ...opts,
    });
    const text = await res.text();
    let data = {};
    try { data = text ? JSON.parse(text) : {}; } catch (_) { data = { raw: text }; }
    if (!res.ok) {
      throw new Error(data.message || data.error || `HTTP ${res.status}`);
    }
    return data;
  }

  function getPointer(obj, pointer) {
    if (!obj || !pointer) return undefined;
    const parts = pointer.replace(/^\//, "").split("/");
    let cur = obj;
    for (const p of parts) {
      if (cur === null || cur === undefined) return undefined;
      cur = Array.isArray(cur) ? cur[Number(p)] : cur[p];
    }
    return cur;
  }

  function themeColors() {
    const light = document.documentElement.getAttribute("data-theme") === "light";
    return {
      paper: "rgba(0,0,0,0)",
      plot: "rgba(0,0,0,0)",
      text: light ? "#111827" : "#e6edf5",
      grid: light ? "#e2e8f0" : "#253141",
    };
  }

  // Zero-dependency Canvas fallback if Plotly is absent in a minimal Python env
  function renderCanvasFallback(container, traces, { logY = false, xTitle = "", yTitle = "" } = {}) {
    container.innerHTML = "";
    const canvas = document.createElement("canvas");
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    container.appendChild(canvas);
    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(260, Math.floor(rect.width));
    const h = Math.max(200, Math.floor(rect.height));
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const tc = themeColors();
    const pad = { l: 58, r: 16, t: 24, b: 34 };
    const pts = [];
    traces.forEach((tr) => {
      for (let i = 0; i < tr.x.length; i++) {
        const xv = Number(tr.x[i]);
        const yv = Number(tr.y[i]);
        if (Number.isFinite(xv) && Number.isFinite(yv) && (!logY || yv > 0)) {
          pts.push([xv, logY ? Math.log10(yv) : yv]);
        }
      }
    });
    if (!pts.length) {
      ctx.fillStyle = tc.text;
      ctx.font = "13px sans-serif";
      ctx.fillText("No plottable samples yet", pad.l, h / 2);
      return;
    }
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const [x, y] of pts) {
      if (x < xmin) xmin = x;
      if (x > xmax) xmax = x;
      if (y < ymin) ymin = y;
      if (y > ymax) ymax = y;
    }
    if (xmax === xmin) { xmin -= 1; xmax += 1; }
    if (ymax === ymin) { ymin -= 1; ymax += 1; }
    const sx = (x) => pad.l + ((x - xmin) / (xmax - xmin)) * (w - pad.l - pad.r);
    const sy = (y) => h - pad.b - ((y - ymin) / (ymax - ymin)) * (h - pad.t - pad.b);

    ctx.strokeStyle = tc.grid;
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const y = pad.t + (g / 4) * (h - pad.t - pad.b);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      const val = ymax - (g / 4) * (ymax - ymin);
      ctx.fillStyle = tc.text;
      ctx.font = "11px monospace";
      ctx.fillText(logY ? `1e${val.toFixed(1)}` : fmtNum(val, 2), 4, y + 4);
    }
    traces.forEach((tr, ti) => {
      ctx.strokeStyle = tr.line?.color || PALETTE[ti % PALETTE.length];
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < tr.x.length; i++) {
        const xv = Number(tr.x[i]);
        const rawY = Number(tr.y[i]);
        if (!Number.isFinite(xv) || !Number.isFinite(rawY) || (logY && rawY <= 0)) {
          started = false; continue;
        }
        const yv = logY ? Math.log10(rawY) : rawY;
        const px = sx(xv), py = sy(yv);
        if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
      }
      ctx.stroke();
    });
    ctx.fillStyle = tc.text;
    ctx.font = "11px sans-serif";
    ctx.fillText(xTitle || "x", w / 2, h - 6);
    if (yTitle) ctx.fillText(yTitle, pad.l, 14);
  }

  function renderPlot(containerId, traces, { logY = false, xTitle = "", yTitle = "", y2Title = null } = {}) {
    const el = $(containerId);
    if (!el) return;
    if (typeof window.Plotly === "undefined") {
      renderCanvasFallback(el, traces, { logY, xTitle, yTitle });
      return;
    }
    const tc = themeColors();
    const layout = {
      margin: { l: 58, r: y2Title ? 54 : 18, t: 18, b: 42 },
      paper_bgcolor: tc.paper,
      plot_bgcolor: tc.plot,
      font: { color: tc.text, size: 11.5 },
      hovermode: "x unified",
      legend: { orientation: "h", y: -0.22, x: 0 },
      xaxis: { title: xTitle, gridcolor: tc.grid, zerolinecolor: tc.grid },
      yaxis: {
        title: yTitle,
        type: logY ? "log" : "linear",
        gridcolor: tc.grid,
        zerolinecolor: tc.grid,
        exponentformat: "e",
      },
      autosize: true,
    };
    if (y2Title) {
      layout.yaxis2 = {
        title: y2Title,
        overlaying: "y",
        side: "right",
        showgrid: false,
      };
    }
    if (!traces.length) {
      layout.annotations = [{
        text: "Waiting for samples...",
        xref: "paper", yref: "paper", x: 0.5, y: 0.5, showarrow: false,
        font: { color: tc.text, size: 13 },
      }];
    }
    window.Plotly.react(el, traces, layout, {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
    });
  }

  function pickX(columns, indices) {
    const mode = $("sel-xaxis").value;
    if (mode === "index") return { x: indices, title: "Sample Index" };
    if (columns[mode] && columns[mode].some((v) => v !== null)) {
      const labels = { step: "Accepted Step", time: "Physical Time", wall: "Wall Seconds" };
      return { x: columns[mode], title: labels[mode] || mode };
    }
    return { x: indices, title: "Sample Index" };
  }

  function makeTrace(name, x, y, idx, extra = {}) {
    const type = x.length > 500 ? "scattergl" : "scatter";
    return {
      name,
      x,
      y,
      type,
      mode: x.length <= 40 ? "lines+markers" : "lines",
      marker: { size: 5 },
      line: { width: 2, color: PALETTE[idx % PALETTE.length] },
      ...extra,
    };
  }

  function renderRunList() {
    const list = $("run-list");
    $("run-count").textContent = String(state.runs.length);
    if (!state.runs.length) {
      list.innerHTML = `<div class="empty-hint">No runs discovered yet.<br>Start <code>cauchy.exe</code> or click <strong>+ Attach</strong>.</div>`;
      return;
    }
    list.innerHTML = "";
    for (const run of state.runs) {
      const btn = document.createElement("button");
      btn.className = "run-item" + (run.runId === state.selectedRunId ? " active" : "");
      btn.innerHTML = `
        <div class="run-item-top">
          <span class="run-item-name"></span>
          <span class="status-badge ${statusClass(run.state)}"></span>
        </div>
        <div class="run-item-meta">
          <span>Step ${run.step ?? 0}${run.iteration >= 0 ? ` · it ${run.iteration}` : ""}</span>
          <span>${fmtWall(run.wallSeconds)}</span>
        </div>
      `;
      btn.querySelector(".run-item-name").textContent = run.label || run.runId;
      btn.querySelector(".status-badge").textContent = run.state;
      btn.addEventListener("click", () => {
        selectRun(run.runId);
        closeDrawer();
      });
      list.appendChild(btn);
    }
  }

  function renderHeader() {
    const d = state.detail;
    if (!d) return;
    const s = d.summary;
    $("run-title").textContent = s.label || s.runId;
    const badge = $("run-status-badge");
    badge.textContent = s.state;
    badge.className = "status-badge " + statusClass(s.state);

    const stepStr = s.iteration >= 0 ? `${s.step} / it ${s.iteration}` : String(s.step ?? 0);
    $("kpi-step").textContent = s.targetSteps && !s.steady ? `${stepStr} (/${s.targetSteps})` : stepStr;
    $("kpi-time").textContent = fmtNum(s.time);
    $("kpi-wall").textContent = fmtWall(s.wallSeconds);

    const cols = state.history?.columns || {};
    const lastOf = (name) => {
      const arr = cols[name];
      if (!arr || !arr.length) return null;
      for (let i = arr.length - 1; i >= 0; i--) {
        if (arr[i] !== null && arr[i] !== undefined) return arr[i];
      }
      return null;
    };
    const resKey = Object.keys(cols).find((k) => k.startsWith("res_")) || "residual_l2";
    $("kpi-res").textContent = fmtNum(lastOf(resKey));
    $("kpi-ratio").textContent = fmtNum(lastOf("residual_target_ratio"));
    $("kpi-cfl").textContent = fmtNum(lastOf("pseudo_cfl"));

    $("run-path-line").textContent =
      `Run ID: ${s.runId} | Prefix: ${s.prefix || "—"} | Endpoint: ${s.url || "disk-only"} | Rev: ${s.appliedConfigRevision ?? 0}`;

    // Domain selector
    const domains = d.manifest?.domains || [];
    const sel = $("sel-domain");
    const count = Math.max(1, domains.length || s.domainCount || 1);
    if (sel.options.length !== count) {
      sel.innerHTML = "";
      for (let i = 0; i < count; i++) {
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = domains[i]?.name ? `D${i}: ${domains[i].name}` : `Domain ${i}`;
        sel.appendChild(opt);
      }
    }
    sel.value = String(state.domain);
    $("tab-bc-count").textContent = String(s.boundaryCount || 0);
  }

  function renderCharts() {
    const h = state.history;
    if (!h) return;
    const cols = h.columns || {};
    const { x, title: xTitle } = pickX(cols, h.indices || []);
    const logY = $("chk-log").checked;

    $("history-meta").textContent =
      `${h.totalRows} rows (${h.source}${h.downsampled ? `, shown ${h.returnedRows}` : ""})`;

    // 1. Residuals
    const resKeys = (h.header || []).filter((k) => k.startsWith("res_"));
    const resTraces = [];
    resKeys.forEach((k, idx) => {
      if (cols[k]?.some((v) => v !== null)) {
        resTraces.push(makeTrace(k, x, cols[k], idx));
      }
    });
    if (!resTraces.length && cols.residual_l2?.some((v) => v !== null)) {
      resTraces.push(makeTrace("residual_l2", x, cols.residual_l2, 0));
    }
    if (cols.residual_target_ratio?.some((v) => v !== null)) {
      resTraces.push(makeTrace("target_ratio", x, cols.residual_target_ratio, resTraces.length, {
        line: { width: 1.6, dash: "dot", color: "#94a3b8" },
      }));
    }
    $("res-series-hint").textContent = resKeys.join(", ");
    renderPlot("plot-residuals", resTraces, { logY, xTitle, yTitle: "Residual" });

    // 2. Solver dynamics
    const dynMode = $("sel-dynamics").value;
    const dynTraces = [];
    let yTitle = "", y2Title = null;
    if (dynMode === "cfl_linear") {
      if (cols.pseudo_cfl?.some((v) => v !== null)) {
        dynTraces.push(makeTrace("pseudo_cfl", x, cols.pseudo_cfl, 0));
      }
      if (cols.linear_iterations?.some((v) => v !== null)) {
        dynTraces.push(makeTrace("linear_iterations", x, cols.linear_iterations, 1, { yaxis: "y2" }));
        y2Title = "Linear Iters";
      }
      yTitle = "Pseudo-CFL";
    } else if (dynMode === "vr") {
      if (cols.vr_sweeps?.some((v) => v !== null)) dynTraces.push(makeTrace("vr_sweeps", x, cols.vr_sweeps, 2));
      if (cols.vr_relative_update?.some((v) => v !== null)) {
        dynTraces.push(makeTrace("vr_relative_update", x, cols.vr_relative_update, 3, { yaxis: "y2" }));
        y2Title = "VR Rel Update";
      }
      yTitle = "VR Sweeps";
    } else {
      const k = cols.iteration_wall?.some((v) => v !== null) ? "iteration_wall" : "step_wall";
      if (cols[k]?.some((v) => v !== null)) dynTraces.push(makeTrace(k, x, cols[k], 4));
      yTitle = "Seconds";
    }
    renderPlot("plot-dynamics", dynTraces, { logY: false, xTitle, yTitle, y2Title });

    // 3. Verbose state bounds
    const minKeys = (h.header || []).filter((k) => k.startsWith("min_"));
    const selBound = $("sel-bound-comp");
    if (minKeys.length) {
      const comps = minKeys.map((k) => k.slice(4));
      if (selBound.options.length !== comps.length) {
        selBound.innerHTML = comps.map((c) => `<option value="${c}">${c}</option>`).join("");
      }
      if (state.verbose) {
        const comp = selBound.value || comps[0];
        const bTraces = [
          makeTrace(`min_${comp}`, x, cols[`min_${comp}`] || [], 0),
          makeTrace(`max_${comp}`, x, cols[`max_${comp}`] || [], 1),
        ];
        renderPlot("plot-bounds", bTraces, { logY: false, xTitle, yTitle: comp });
      }
    }
  }

  function renderBoundaries() {
    const b = state.boundaries;
    if (!b) return;
    const avail = b.available || [];
    $("bc-empty").hidden = avail.length > 0;
    $("bc-charts").hidden = avail.length === 0;

    const sel = $("sel-bc");
    const desired = avail.map((a) => a.canonicalName).join("|");
    if (sel.dataset.key !== desired) {
      sel.dataset.key = desired;
      sel.innerHTML = avail.map((a) =>
        `<option value="${a.canonicalName}">${a.canonicalName} (zone ${a.zone}${a.rawName ? ` · ${a.rawName}` : ""})</option>`
      ).join("");
    }
    if (b.selected) sel.value = b.selected.canonicalName;

    const series = b.series || {};
    const cols = series.columns || {};
    const { x, title: xTitle } = pickX(cols, series.indices || []);
    const areaArr = cols.area || [];
    const lastArea = areaArr.length ? areaArr[areaArr.length - 1] : null;
    $("bc-area-badge").textContent = `Area: ${fmtNum(lastArea, 4)}`;
    $("bc-meta").textContent = `${series.totalRows || 0} samples`;

    const fluxKeys = (series.header || []).filter((k) => k.startsWith("flux_"));
    const fluxTraces = fluxKeys.map((k, idx) => makeTrace(k, x, cols[k] || [], idx));
    renderPlot("plot-bc-flux", fluxTraces, { logY: false, xTitle, yTitle: "Integrated Flux" });

    const tanTraces = [];
    ["force_t_x", "force_t_y", "force_t_z"].forEach((k, idx) => {
      if (cols[k]?.some((v) => v !== null)) tanTraces.push(makeTrace(k, x, cols[k], idx));
    });
    if (cols.force_t_x && cols.force_t_y && cols.force_t_z) {
      const mag = cols.force_t_x.map((_, i) => {
        const fx = Number(cols.force_t_x[i] || 0);
        const fy = Number(cols.force_t_y[i] || 0);
        const fz = Number(cols.force_t_z[i] || 0);
        return Math.sqrt(fx * fx + fy * fy + fz * fz);
      });
      tanTraces.push(makeTrace("|force_t|", x, mag, 3, { line: { width: 2.2, dash: "dash", color: "#e879f9" } }));
    }
    renderPlot("plot-bc-tangent", tanTraces, { logY: false, xTitle, yTitle: "Tangential Force" });

    // Verbose registry table
    const tbl = $("tbl-boundaries");
    if (avail.length) {
      tbl.innerHTML = `<thead><tr><th>Zone</th><th>Canonical Name</th><th>Raw Mesh Name</th><th>CSV</th></tr></thead><tbody>` +
        avail.map((a) => `<tr><td>${a.zone}</td><td><code>${a.canonicalName}</code></td><td>${a.rawName || "—"}</td><td><code>${a.csv || "—"}</code></td></tr>`).join("") +
        `</tbody>`;
    } else {
      tbl.innerHTML = "";
    }
  }

  function renderSteering() {
    const d = state.detail;
    if (!d) return;
    const online = Boolean(d.summary?.online);
    $("steering-offline-banner").hidden = online;
    $("btn-apply-output").disabled = !online;
    $("btn-trigger-snapshot").disabled = !online;
    $("btn-apply-steering").disabled = !online;
    $("btn-stop-run").disabled = !online;

    const schema = d.mutableSchema;
    const cfg = d.configuration;
    const schemaKey = `${d.summary.runId}:${d.summary.appliedConfigRevision}:${online}`;
    if (schema && schemaKey !== state.renderedSchemaKey) {
      state.renderedSchemaKey = schemaKey;
      state.dirtyOutput = {};
      state.dirtySteering = {};

      const buildForm = (containerId, fields, dirtyMap) => {
        const box = $(containerId);
        box.innerHTML = "";
        for (const f of fields || []) {
          const curVal = getPointer(cfg, f.pointer);
          const wrap = document.createElement("div");
          wrap.className = "field";
          const label = document.createElement("label");
          label.textContent = f.pointer.replace(/^\/(output|time)\//, "").replace(/^\/domains\/(\d+)\/settings\/implicit\//, "D$1 implicit.");
          label.title = `${f.pointer}\n${f.description || ""}`;
          const input = document.createElement("input");
          input.type = "number";
          input.step = f.type === "integer" ? "1" : "any";
          if (f.minimum !== undefined) input.min = String(f.minimum);
          if (f.maximum !== undefined) input.max = String(f.maximum);
          input.value = curVal !== undefined && curVal !== null ? String(curVal) : "";
          input.disabled = !online;
          input.addEventListener("input", () => {
            const raw = input.value.trim();
            if (raw === "" || Number(raw) === Number(curVal)) {
              delete dirtyMap[f.pointer];
              input.classList.remove("dirty");
            } else {
              dirtyMap[f.pointer] = f.type === "integer" ? parseInt(raw, 10) : Number(raw);
              input.classList.add("dirty");
            }
          });
          wrap.appendChild(label);
          wrap.appendChild(input);
          box.appendChild(wrap);
        }
      };

      buildForm("form-output-fields", schema.categories?.output || [], state.dirtyOutput);
      buildForm("form-steering-fields", schema.categories?.steering || [], state.dirtySteering);
    }

    // Updates table
    const updates = d.updates || [];
    $("updates-count").textContent = String(updates.length);
    const tbl = $("tbl-updates");
    if (!updates.length) {
      tbl.innerHTML = `<tbody><tr><td class="muted-xs">No live commands sent yet.</td></tr></tbody>`;
    } else {
      tbl.innerHTML = `<thead><tr><th>Rev</th><th>Kind</th><th>Category</th><th>Boundary</th><th>State</th><th>Applied At</th><th>Details</th></tr></thead><tbody>` +
        [...updates].reverse().map((u) => {
          const where = u.state === "applied" ? `S${u.appliedStep} st${u.appliedStage} it${u.appliedIteration}` : "—";
          return `<tr>
            <td>#${u.revision}</td>
            <td>${u.kind}</td>
            <td><span class="cat-badge ${u.category === "steering" ? "cat-steering" : "cat-output"}">${u.category}</span></td>
            <td>${u.applyAt}</td>
            <td><strong>${u.state}</strong></td>
            <td>${where}</td>
            <td><code>${u.patch ? JSON.stringify(u.patch) : (u.message || "")}</code></td>
          </tr>`;
        }).join("") + `</tbody>`;
    }
  }

  function renderDetails() {
    const d = state.detail;
    if (!d) return;
    $("pre-config").textContent = JSON.stringify(d.configuration || {}, null, 2);
    const cols = state.history?.columns || {};
    const rows = [];
    for (const key of state.history?.header || []) {
      const arr = cols[key] || [];
      let latest = null;
      for (let i = arr.length - 1; i >= 0; i--) {
        if (arr[i] !== null && arr[i] !== undefined) { latest = arr[i]; break; }
      }
      if (latest !== null) rows.push([key, fmtNum(latest, 5)]);
    }
    $("tbl-latest").innerHTML = `<thead><tr><th>Metric</th><th>Latest Value</th></tr></thead><tbody>` +
      (rows.length ? rows.map(([k, v]) => `<tr><td><code>${k}</code></td><td><code>${v}</code></td></tr>`).join("")
        : `<tr><td colspan="2" class="muted-xs">No history rows yet.</td></tr>`) + `</tbody>`;
  }

  function applyVerboseVisibility() {
    document.querySelectorAll(".verbose-only").forEach((el) => {
      el.hidden = !state.verbose;
    });
    if (state.verbose && state.tab === "charts") renderCharts();
  }

  async function refreshRuns(rescan = true) {
    try {
      const [hub, data] = await Promise.all([
        api("/api/hub"),
        api(`/api/runs?rescan=${rescan ? "1" : "0"}`),
      ]);
      state.runs = data.runs || [];
      $("hub-meta").textContent = `${hub.onlineCount} live · ${hub.runCount} total · ${hub.plotlyAvailable ? "Plotly" : "Canvas"}`;
      renderRunList();
      if (!state.selectedRunId && state.runs.length) {
        await selectRun(state.runs[0].runId);
      }
    } catch (err) {
      toast(`Refresh failed: ${err.message}`, true);
    }
  }

  async function refreshSelected() {
    if (!state.selectedRunId) return;
    const rid = state.selectedRunId;
    const kind = $("sel-kind").value;
    try {
      const [detail, history, boundaries] = await Promise.all([
        api(`/api/runs/${encodeURIComponent(rid)}/detail`),
        api(`/api/runs/${encodeURIComponent(rid)}/history?domain=${state.domain}&kind=${kind}&maxPoints=2000`),
        api(`/api/runs/${encodeURIComponent(rid)}/boundaries?domain=${state.domain}${state.bcName ? `&bc=${encodeURIComponent(state.bcName)}` : ""}&maxPoints=2000`),
      ]);
      state.detail = detail;
      state.history = history;
      state.boundaries = boundaries;
      $("empty-state").hidden = true;
      $("run-workspace").hidden = false;
      renderHeader();
      renderCharts();
      renderBoundaries();
      renderSteering();
      renderDetails();
    } catch (err) {
      toast(`Failed to load run: ${err.message}`, true);
    }
  }

  async function selectRun(runId) {
    state.selectedRunId = runId;
    state.domain = 0;
    state.bcName = null;
    state.renderedSchemaKey = null;
    renderRunList();
    await refreshSelected();
  }

  function openDrawer() {
    $("sidebar").classList.add("open");
    $("drawer-backdrop").hidden = false;
  }
  function closeDrawer() {
    $("sidebar").classList.remove("open");
    $("drawer-backdrop").hidden = true;
  }

  function setTab(name) {
    state.tab = name;
    document.querySelectorAll(".tabbar .tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
    document.querySelectorAll(".tab-panel").forEach((p) => { p.hidden = p.id !== `panel-${name}`; });
    if (name === "charts") renderCharts();
    if (name === "boundaries") renderBoundaries();
    if (name === "steering") renderSteering();
    if (name === "details") renderDetails();
  }

  function startLoop() {
    clearInterval(state.timer);
    state.timer = setInterval(async () => {
      if (!state.live || document.hidden) return;
      await refreshRuns(true);
      if (state.selectedRunId) await refreshSelected();
    }, 2000);
  }

  function initEvents() {
    $("btn-menu").addEventListener("click", () => {
      if ($("sidebar").classList.contains("open")) closeDrawer(); else openDrawer();
    });
    $("drawer-backdrop").addEventListener("click", closeDrawer);

    $("btn-theme").addEventListener("click", () => {
      const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("cauchy_hub_theme", next);
      renderCharts();
      renderBoundaries();
    });
    const savedTheme = localStorage.getItem("cauchy_hub_theme");
    if (savedTheme) document.documentElement.setAttribute("data-theme", savedTheme);

    // Default verbose on wide desktops, concise on narrow mobile screens
    state.verbose = window.innerWidth >= 1200;
    $("chk-verbose").checked = state.verbose;
    applyVerboseVisibility();
    $("chk-verbose").addEventListener("change", (e) => {
      state.verbose = e.target.checked;
      applyVerboseVisibility();
    });

    $("chk-live").addEventListener("change", (e) => { state.live = e.target.checked; });
    $("btn-rescan").addEventListener("click", () => refreshRuns(true));
    $("btn-rescan-empty").addEventListener("click", () => refreshRuns(true));
    $("btn-refresh-now").addEventListener("click", () => refreshSelected());

    $("sel-domain").addEventListener("change", (e) => {
      state.domain = Number(e.target.value);
      state.bcName = null;
      refreshSelected();
    });
    $("sel-kind").addEventListener("change", refreshSelected);
    $("sel-xaxis").addEventListener("change", () => { renderCharts(); renderBoundaries(); });
    $("chk-log").addEventListener("change", renderCharts);
    $("sel-dynamics").addEventListener("change", renderCharts);
    $("sel-bound-comp").addEventListener("change", renderCharts);
    $("sel-bc").addEventListener("change", (e) => { state.bcName = e.target.value; refreshSelected(); });

    document.querySelectorAll(".tabbar .tab").forEach((btn) => {
      btn.addEventListener("click", () => setTab(btn.dataset.tab));
    });

    // Steering actions
    $("btn-apply-output").addEventListener("click", async () => {
      if (!Object.keys(state.dirtyOutput).length) {
        toast("No output fields modified"); return;
      }
      try {
        const applyAt = $("sel-output-boundary").value;
        const rec = await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}/config`, {
          method: "PATCH",
          body: JSON.stringify({ applyAt, patch: state.dirtyOutput }),
        });
        toast(`Queued output update #${rec.revision} (apply at ${rec.applyAt})`);
        state.dirtyOutput = {};
        state.renderedSchemaKey = null;
        await refreshSelected();
      } catch (err) { toast(`Output update failed: ${err.message}`, true); }
    });

    $("btn-trigger-snapshot").addEventListener("click", async () => {
      try {
        const rec = await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}/snapshot`, {
          method: "POST",
          body: JSON.stringify({
            checkpoint: $("chk-snap-ckpt").checked,
            visualization: $("chk-snap-vis").checked,
            applyAt: $("sel-output-boundary").value,
          }),
        });
        toast(`Queued snapshot #${rec.revision} at ${rec.applyAt} boundary`);
        await refreshSelected();
      } catch (err) { toast(`Snapshot failed: ${err.message}`, true); }
    });

    $("btn-apply-steering").addEventListener("click", async () => {
      const keys = Object.keys(state.dirtySteering);
      if (!keys.length) { toast("No steering fields modified"); return; }
      const applyAt = $("sel-steering-boundary").value;
      if (!window.confirm(`Apply simulation steering patch (${keys.length} field(s)) at next ${applyAt} boundary?\n\n${JSON.stringify(state.dirtySteering, null, 2)}`)) {
        return;
      }
      try {
        const rec = await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}/config`, {
          method: "PATCH",
          body: JSON.stringify({ applyAt, patch: state.dirtySteering }),
        });
        toast(`Queued steering update #${rec.revision} (apply at ${rec.applyAt})`);
        state.dirtySteering = {};
        state.renderedSchemaKey = null;
        await refreshSelected();
      } catch (err) { toast(`Steering update failed: ${err.message}`, true); }
    });

    $("btn-stop-run").addEventListener("click", async () => {
      const applyAt = $("sel-stop-boundary").value;
      if (!window.confirm(`Send graceful stop to "${state.detail?.summary?.label}" at ${applyAt} boundary?\nThe solver will write a restartable -stop checkpoint and exit.`)) {
        return;
      }
      try {
        await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}/stop`, {
          method: "POST",
          body: JSON.stringify({ applyAt }),
        });
        toast(`Graceful stop queued (${applyAt})`);
        await refreshSelected();
      } catch (err) { toast(`Stop failed: ${err.message}`, true); }
    });

    $("btn-detach").addEventListener("click", async () => {
      if (!state.selectedRunId) return;
      try {
        await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}`, { method: "DELETE" });
        state.selectedRunId = null;
        state.detail = null;
        $("run-workspace").hidden = true;
        $("empty-state").hidden = false;
        await refreshRuns(false);
        toast("Run detached from hub list");
      } catch (err) { toast(err.message, true); }
    });

    $("btn-copy-config").addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText($("pre-config").textContent);
        toast("Configuration copied");
      } catch (_) { toast("Copy failed", true); }
    });

    // Attach modal
    const openAttach = () => { $("attach-error").hidden = true; $("attach-modal").hidden = false; };
    const closeAttach = () => { $("attach-modal").hidden = true; };
    $("btn-attach").addEventListener("click", openAttach);
    $("btn-attach-empty").addEventListener("click", openAttach);
    $("btn-close-attach").addEventListener("click", closeAttach);
    $("attach-modal").addEventListener("click", (e) => { if (e.target === $("attach-modal")) closeAttach(); });

    document.querySelectorAll(".tab-mini").forEach((b) => {
      b.addEventListener("click", () => {
        document.querySelectorAll(".tab-mini").forEach((x) => x.classList.toggle("active", x === b));
        $("form-attach-local").hidden = b.dataset.attach !== "local";
        $("form-attach-remote").hidden = b.dataset.attach !== "remote";
        $("attach-error").hidden = true;
      });
    });

    $("form-attach-local").addEventListener("submit", async (e) => {
      e.preventDefault();
      try {
        const res = await api("/api/runs/attach", {
          method: "POST",
          body: JSON.stringify({
            path: $("inp-attach-path").value.trim(),
            label: $("inp-attach-label-local").value.trim() || undefined,
          }),
        });
        closeAttach();
        await refreshRuns(false);
        await selectRun(res.runId);
        toast(`Attached ${res.label}`);
      } catch (err) {
        $("attach-error").textContent = err.message;
        $("attach-error").hidden = false;
      }
    });

    $("form-attach-remote").addEventListener("submit", async (e) => {
      e.preventDefault();
      try {
        const res = await api("/api/runs/attach", {
          method: "POST",
          body: JSON.stringify({
            url: $("inp-attach-url").value.trim(),
            token: $("inp-attach-token").value.trim(),
            label: $("inp-attach-label-remote").value.trim() || undefined,
          }),
        });
        closeAttach();
        await refreshRuns(false);
        await selectRun(res.runId);
        toast(`Connected to ${res.label}`);
      } catch (err) {
        $("attach-error").textContent = err.message;
        $("attach-error").hidden = false;
      }
    });
  }

  initEvents();
  refreshRuns(true);
  startLoop();
})();

// Resource Status mini-panel — polls /api/resources/status every 30s.
//
// Multi-instance: any element with class `.res-status-mount` becomes a panel.
// All instances on a page share ONE poll (single fetch → render into each).
//
// Per-instance DOM contract (scoped lookups within the mount):
//   [data-res-content]  — where rendered rows are appended (required)
//   [data-res-checked]  — receives "checked: <iso>" timestamp (optional)
//   [data-res-body]     — wrapper toggled by [data-res-toggle] (optional)
//   [data-res-toggle]   — collapse/expand button toggling [data-res-body]
//   [data-res-help]     — opens tooltip with HELP_TEXT
//
// Backward-compat: a legacy element with id="res-status-panel" is treated as
// an implicit `.res-status-mount` so the floating dashboard panel keeps working
// without a class change. Internal `id` -> `data-` lookups inside fall back to
// id when scoped query returns null (covers the legacy floating panel).
(function () {
    const POLL_MS = 30000;
    const HELP_TEXT =
`Resource Status — at-a-glance health of every backend the platform talks to.

Color key:
  GREEN  — reachable / configured / healthy
  AMBER  — reachable but degraded (e.g. δ² engine up but model not loaded)
  RED    — unreachable, key missing, or probe error
  GREY   — not configured (no env var set)

Rows:
  Ollama — three local Ollama endpoints (OLLAMA_URL_1/2/3). Tooltip shows
           the URL, version, and last successful probe time.
  Together.ai / Anthropic API — hosted-model keys (set / not set).
  RunPod — management key + active pod ID if any.
  δ² engine — /health probe; surfaces model_loaded / training_status / bassin_size.
  Hosts — TCP reachability of registered machines.
  Machines — CPU/RAM/GPU bars per known box.

Polls every 30s. Read-only. Same data renders into every mount on the page.`;

    function pickEl(root, dataAttr, legacyId) {
        return root.querySelector(`[${dataAttr}]`) ||
               (legacyId ? document.getElementById(legacyId) : null);
    }

    function escapeText(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, c =>
            ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    function makeRow(klass, label, meta, title) {
        const div = document.createElement('div');
        div.className = 'res-row';
        if (title) div.title = title;
        div.innerHTML =
            '<span class="res-dot ' + klass + '"></span>' +
            '<span class="res-label">' + escapeText(label) + '</span>' +
            (meta ? '<span class="res-meta">' + escapeText(meta) + '</span>' : '');
        return div;
    }

    function makeSection(text) {
        const d = document.createElement('div');
        d.className = 'res-section-title';
        d.textContent = text;
        return d;
    }

    function makeBarRow(label, percent, bcls, meta) {
        const wrap = document.createElement('div');
        wrap.style.cssText = 'display:flex;align-items:center;gap:4px;line-height:1.3';
        const fillColor = bcls === 'res-dot-ok' ? '#4caf50'
                        : bcls === 'res-dot-warn' ? '#f5a623'
                        : '#d4556e';
        const p = Math.max(0, Math.min(100, percent || 0));
        wrap.innerHTML =
            '<span style="display:inline-block;width:34px;flex:0 0 auto">' + escapeText(label) + '</span>' +
            '<span style="display:inline-block;width:54px;height:6px;background:#222;border-radius:3px;overflow:hidden;flex:0 0 auto">' +
              '<span style="display:block;height:100%;width:' + p.toFixed(0) + '%;background:' + fillColor + '"></span>' +
            '</span>' +
            '<span style="width:34px;text-align:right;flex:0 0 auto">' + p.toFixed(0) + '%</span>' +
            '<span style="flex:1;color:var(--muted,#7d8590);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + escapeText(meta || '') + '</span>';
        return wrap;
    }

    // ── Wire up one mount: collapse toggle + help tooltip. Returns a render
    //    function that paints fresh data into this mount's content area.
    function wireMount(mount) {
        const content = pickEl(mount, 'data-res-content', 'res-status-content');
        const checked = pickEl(mount, 'data-res-checked', 'res-status-checked');
        const body    = pickEl(mount, 'data-res-body',    'res-status-body');
        const toggle  = pickEl(mount, 'data-res-toggle',  'res-status-toggle');
        const helpBtn = pickEl(mount, 'data-res-help',    'res-status-help');

        if (!content) return null;  // malformed mount — skip

        if (toggle && body) {
            let collapsed = false;
            toggle.addEventListener('click', () => {
                collapsed = !collapsed;
                body.style.display = collapsed ? 'none' : '';
                toggle.textContent = collapsed ? '+' : '−';
            });
        }

        if (helpBtn) {
            helpBtn.addEventListener('click', (e) => {
                const tt = document.getElementById('d2-tooltip');
                if (tt && tt.style) {
                    tt.innerHTML =
                        '<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px">' +
                        '<strong style="font-size:12px">Resource Status</strong>' +
                        '<button onclick="document.getElementById(\'d2-tooltip\').style.display=\'none\'" style="background:none;border:0;color:var(--muted);cursor:pointer;font-size:14px">×</button>' +
                        '</div>' +
                        '<pre style="white-space:pre-wrap;font-family:inherit;margin:6px 0 0 0;font-size:11px;line-height:1.45">' + HELP_TEXT + '</pre>';
                    const r = helpBtn.getBoundingClientRect();
                    tt.style.left = (r.left - 290) + 'px';
                    tt.style.top  = (r.bottom + 6 + window.scrollY) + 'px';
                    tt.style.display = 'block';
                    e.stopPropagation();
                } else {
                    alert(HELP_TEXT);
                }
            });
        }

        return function render(d, errMsg) {
            if (errMsg) {
                content.innerHTML = '<div style="color:#d4556e">' + escapeText(errMsg) + '</div>';
                return;
            }
            content.innerHTML = '';

            content.appendChild(makeSection('Ollama (' + (d.ollama || []).length + ')'));
            for (const o of (d.ollama || [])) {
                let cls, meta;
                if (o.ok) { cls = 'res-dot-ok'; meta = 'v' + (o.version || '?'); }
                else if (o.stale) { cls = 'res-dot-warn'; meta = 'stale (' + (o.consecutive_failures || 1) + ')'; }
                else { cls = 'res-dot-err'; meta = 'down'; }
                const title = (o.url || '') +
                              (o.last_ok ? '\nlast OK: ' + o.last_ok : '') +
                              (o.consecutive_failures ? '\nconsecutive failures: ' + o.consecutive_failures : '') +
                              (o.error ? '\nerror: ' + o.error : '');
                content.appendChild(makeRow(cls, o.label || o.url, meta, title));
            }

            content.appendChild(makeSection('API keys'));
            const t = d.together_ai || {};
            content.appendChild(makeRow(
                t.configured ? 'res-dot-ok' : 'res-dot-off',
                'Together.ai',
                t.configured ? (t.masked || 'set') : 'not set',
                t.configured ? 'configured' : 'TOGETHER_AI_KEY env var missing'
            ));
            const a = d.anthropic || {};
            content.appendChild(makeRow(
                a.configured ? 'res-dot-ok' : 'res-dot-off',
                'Anthropic API',
                a.configured ? (a.masked || 'set') : 'not set',
                a.configured ? 'configured' : 'ANTHROPIC_API_KEY env var missing'
            ));

            content.appendChild(makeSection('RunPod'));
            const rp = d.runpod || {};
            let rpClass = 'res-dot-off', rpMeta = 'not set';
            if (rp.configured) {
                if (rp.active_pod && rp.active_pod.pod_id) {
                    rpClass = 'res-dot-ok';
                    rpMeta = rp.active_pod.pod_id.slice(0, 8) + ' ' + (rp.active_pod.gpu || '');
                } else {
                    rpClass = 'res-dot-warn';
                    rpMeta = 'no pod';
                }
            }
            content.appendChild(makeRow(rpClass, 'RunPod', rpMeta,
                rp.configured ? ('key set' + (rp.active_pod ? '\npod: ' + rp.active_pod.pod_id : '\nno active pod')) : 'RUNPOD_API_KEY missing'
            ));

            content.appendChild(makeSection('δ² engine'));
            const d2 = d.d2_engine || {};
            let d2Class = 'res-dot-off', d2Meta = 'not set';
            if (d2.configured) {
                if (d2.ok && d2.active_lora_adapter) { d2Class = 'res-dot-ok'; d2Meta = 'live · ' + d2.active_lora_adapter; }
                else if (d2.ok && d2.model_loaded) { d2Class = 'res-dot-ok'; d2Meta = 'live · model loaded'; }
                else if (d2.ok) { d2Class = 'res-dot-warn'; d2Meta = (d2.training_status || 'idle') + ' · no model'; }
                else { d2Class = 'res-dot-err'; d2Meta = 'unreachable'; }
            }
            content.appendChild(makeRow(d2Class, 'δ² engine', d2Meta,
                (d2.endpoint || 'no endpoint') +
                (d2.bassin_size != null ? '\nbassin: ' + d2.bassin_size : '') +
                (d2.error ? '\nerror: ' + d2.error : '')
            ));

            // Active Jobs — operational view: what is actually running right now
            // across services. Currently sources from d² engine /health (training
            // status, loaded LoRA, bassin growth); trainer jobs would land here
            // too once those endpoints expose /jobs.
            content.appendChild(makeSection('Active jobs'));
            const aj = [];
            const _d2 = d.d2_engine || {};
            if (_d2.configured && _d2.ok) {
                if (_d2.training_status && _d2.training_status !== 'idle') {
                    const runId = _d2.current_run_id ? ' · run=' + String(_d2.current_run_id).slice(0,8) : '';
                    const opt = _d2.current_optimizer ? ' · ' + _d2.current_optimizer : '';
                    aj.push({cls:'res-dot-warn', label:'d² training', meta:_d2.training_status + runId + opt});
                }
                if (_d2.active_lora_adapter) {
                    aj.push({cls:'res-dot-ok', label:'d² LoRA loaded', meta:_d2.active_lora_adapter});
                } else if (_d2.model_loaded) {
                    aj.push({cls:'res-dot-ok', label:'d² base model loaded', meta:''});
                }
                if (_d2.loaded_lora_count > 0) {
                    aj.push({cls:'res-dot-ok', label:'d² LoRA pool', meta:_d2.loaded_lora_count + ' adapter(s) resident'});
                }
                if (typeof _d2.bassin_size === 'number') {
                    const cls2 = _d2.bassin_size > 0 ? 'res-dot-ok' : 'res-dot-off';
                    aj.push({cls:cls2, label:'d² bassin size', meta:_d2.bassin_size.toLocaleString() + ' tensors'});
                }
            }
            // TODO: trainer jobs once trainers expose /jobs or /active.
            if (!aj.length) {
                content.appendChild(makeRow('res-dot-off', '(no active jobs)', 'idle across all services'));
            } else {
                for (const j of aj) content.appendChild(makeRow(j.cls, j.label, j.meta));
            }

            content.appendChild(makeSection('Hosts'));
            const _machineHostFrags = (d.machines || [])
                .filter(mm => mm.host_endpoint && mm.host)
                .map(mm => mm.host);
            for (const h of (d.hosts || [])) {
                const cls = h.ok ? 'res-dot-ok' : 'res-dot-err';
                const meta = h.ok ? (h.rtt_ms + 'ms') : 'down';
                let label = h.label;
                for (const ip of _machineHostFrags) {
                    if (h.host && (h.host.indexOf(ip) >= 0 || ip.indexOf(h.host) >= 0)) {
                        label += ' ·t';
                        break;
                    }
                }
                content.appendChild(makeRow(cls, label, meta, h.host + ':' + h.port + (h.error ? '\nerror: ' + h.error : '')));
            }

            if ((d.machines || []).length) {
                content.appendChild(makeSection('Machines'));
                for (const m of d.machines) {
                    let cls = 'res-dot-err';
                    if (m.ok) cls = 'res-dot-ok';
                    else if (m.stale) cls = 'res-dot-warn';
                    else if (!m.host_endpoint) cls = 'res-dot-off';

                    const header = document.createElement('div');
                    header.className = 'res-row';
                    header.style.cursor = 'pointer';
                    const labelClean = (m.label || '').replace(/\s*\([^)]*\)\s*$/, '').trim();
                    let hostName;
                    if (labelClean && m.hostname && labelClean !== m.hostname) {
                        hostName = labelClean + ' · ' + m.hostname;
                    } else {
                        hostName = labelClean || m.hostname || m.host || '?';
                    }
                    let status;
                    if (m.ok) status = 'live';
                    else if (m.stale) status = 'stale (' + (m.consecutive_failures||1) + ')';
                    else if (!m.host_endpoint) status = 'no telemetry';
                    else status = 'down';
                    header.innerHTML =
                        '<span class="res-dot ' + cls + '"></span>' +
                        '<span class="res-label">' + escapeText(hostName) + '</span>' +
                        '<span class="res-meta">' + escapeText(status) + '</span>';
                    if (m.error) header.title = m.error;
                    content.appendChild(header);

                    const det = document.createElement('div');
                    det.style.cssText = 'margin:2px 0 6px 17px;font-size:10px;font-family:monospace;color:var(--muted,#7d8590)';

                    if (!m.ok && !m.host_endpoint) {
                        // "no probe agent" specifically means: no /host endpoint is exposed
                        // on this machine to report CPU/RAM/GPU. The machine itself may be
                        // running plenty of services (visible elsewhere in this panel —
                        // Ollama section, Active Jobs, etc.); we just can't draw resource bars.
                        det.textContent = '(no /host probe endpoint — services may be running, see other sections; per-machine resource bars unavailable)';
                    } else if (!m.ok) {
                        det.textContent = m.error || 'unreachable';
                    } else {
                        if (m.cpu) {
                            const p = m.cpu.percent != null ? m.cpu.percent : 0;
                            const bcls = p < 60 ? 'res-dot-ok' : (p < 85 ? 'res-dot-warn' : 'res-dot-err');
                            const la = (m.cpu.load_1 != null) ? ('  la ' + m.cpu.load_1 + '/' + (m.cpu.load_5 != null ? m.cpu.load_5 : '?')) : '';
                            det.appendChild(makeBarRow('CPU', p, bcls, (m.cpu.cores ? m.cpu.cores + 'c' : '') + la));
                        }
                        if (m.ram) {
                            const p = m.ram.percent != null ? m.ram.percent : 0;
                            const bcls = p < 60 ? 'res-dot-ok' : (p < 85 ? 'res-dot-warn' : 'res-dot-err');
                            det.appendChild(makeBarRow('RAM', p, bcls, m.ram.used_gb + '/' + m.ram.total_gb + ' GB'));
                        }
                        for (const g of (m.gpus || [])) {
                            const vp = g.vram_percent != null ? g.vram_percent : 0;
                            const up = g.util_pct     != null ? g.util_pct     : 0;
                            const bcls = vp < 60 ? 'res-dot-ok' : (vp < 85 ? 'res-dot-warn' : 'res-dot-err');
                            const extra = (g.temp_c != null ? ' ' + g.temp_c + '°C' : '')
                                        + (g.power_w != null ? ' ' + g.power_w + 'W' : '');
                            det.appendChild(makeBarRow(
                                'GPU ' + (g.name || '?').replace(/^NVIDIA\s+/i,'').slice(0,20),
                                vp, bcls,
                                'vram ' + (g.vram_used_mib||0) + '/' + (g.vram_total_mib||0) + ' MiB · util ' + up + '%' + extra
                            ));
                        }
                        if (!(m.gpus || []).length) {
                            const noGpu = document.createElement('div');
                            noGpu.textContent = 'GPU: none';
                            det.appendChild(noGpu);
                        }
                        const meta2 = document.createElement('div');
                        meta2.style.marginTop = '2px';
                        const upMin = m.uptime_s != null ? Math.floor(m.uptime_s/60) + 'm' : '?';
                        meta2.textContent = (m.roles || []).join(',') + ' · up ' + upMin;
                        det.appendChild(meta2);
                        for (const k of ['cpu_probe_error','gpu_probe_error']) {
                            if (m[k]) {
                                const e = document.createElement('div');
                                e.style.color = '#d4556e';
                                e.textContent = k + ': ' + m[k];
                                det.appendChild(e);
                            }
                        }
                    }
                    content.appendChild(det);
                    header.addEventListener('click', () => {
                        det.style.display = (det.style.display === 'none') ? '' : 'none';
                    });
                }
            }

            if (checked) checked.textContent = 'checked: ' + (d.checked_at || '?');
        };
    }

    // ── Draggable floating panel (legacy #res-status-panel only).
    //    Per-tab mounts live in document flow and don't need drag. Position
    //    persists in localStorage under 'resStatusPanelPos'. Double-click
    //    the drag handle to reset to default (top-right).
    function wireDrag(panel) {
        const handle = document.getElementById('res-status-drag-handle');
        if (!handle) return;
        const LS_KEY = 'resStatusPanelPos';

        // Restore saved position if any. We switch from right/top anchoring to
        // left/top on first drag so the math stays simple.
        try {
            const saved = JSON.parse(localStorage.getItem(LS_KEY) || 'null');
            if (saved && typeof saved.left === 'number' && typeof saved.top === 'number') {
                panel.style.right = 'auto';
                panel.style.left = saved.left + 'px';
                panel.style.top  = saved.top + 'px';
            }
        } catch (e) { /* malformed LS value — ignore */ }

        let dragging = false, startX = 0, startY = 0, originLeft = 0, originTop = 0;

        handle.addEventListener('mousedown', (e) => {
            // Don't start a drag if the user clicked the toggle button or help icon
            // inside the handle.
            if (e.target.closest('#res-status-toggle, #res-status-help')) return;
            dragging = true;
            startX = e.clientX; startY = e.clientY;
            const rect = panel.getBoundingClientRect();
            originLeft = rect.left; originTop = rect.top;
            panel.style.right = 'auto';
            panel.style.left = originLeft + 'px';
            panel.style.top  = originTop + 'px';
            document.body.style.userSelect = 'none';
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!dragging) return;
            const dx = e.clientX - startX, dy = e.clientY - startY;
            const newLeft = Math.max(0, Math.min(window.innerWidth  - panel.offsetWidth,  originLeft + dx));
            const newTop  = Math.max(0, Math.min(window.innerHeight - panel.offsetHeight, originTop  + dy));
            panel.style.left = newLeft + 'px';
            panel.style.top  = newTop  + 'px';
        });

        document.addEventListener('mouseup', () => {
            if (!dragging) return;
            dragging = false;
            document.body.style.userSelect = '';
            try {
                localStorage.setItem(LS_KEY, JSON.stringify({
                    left: parseInt(panel.style.left, 10),
                    top:  parseInt(panel.style.top, 10),
                }));
            } catch (e) { /* private mode / quota — ignore */ }
        });

        handle.addEventListener('dblclick', (e) => {
            if (e.target.closest('#res-status-toggle, #res-status-help')) return;
            panel.style.left = '';
            panel.style.top  = '14px';
            panel.style.right = '14px';
            try { localStorage.removeItem(LS_KEY); } catch (e) { /* ignore */ }
        });
    }

    // ── Discover all mounts on the page (class + legacy id), wire each up,
    //    then share a single polling loop that re-renders every mount.
    const mountEls = Array.from(document.querySelectorAll('.res-status-mount'));
    const legacy = document.getElementById('res-status-panel');
    if (legacy && !legacy.classList.contains('res-status-mount')) {
        mountEls.push(legacy);
        wireDrag(legacy);  // only the floating panel gets drag; per-tab mounts are inline
    }
    if (!mountEls.length) return;

    const renderers = mountEls.map(wireMount).filter(Boolean);

    async function pollOnce() {
        try {
            const r = await fetch('/api/resources/status');
            if (!r.ok) {
                renderers.forEach(fn => fn(null, 'probe failed (HTTP ' + r.status + ')'));
                return;
            }
            const d = await r.json();
            renderers.forEach(fn => fn(d, null));
        } catch (e) {
            renderers.forEach(fn => fn(null, 'fetch error: ' + (e.message || e)));
        }
    }

    pollOnce();
    setInterval(pollOnce, POLL_MS);
})();

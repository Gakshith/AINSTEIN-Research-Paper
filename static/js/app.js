/* AINSTEIN dashboard — vanilla JS: theming, charts, table, live demo polling. */
(function () {
  "use strict";

  // ---- Theme + nav ----------------------------------------------------------
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  function reducedMotion() {
    return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }
  var _themeBound = false;
  function setupTheme() {
    if (_themeBound) return; // guard against double-binding (DOMContentLoaded + init*)
    _themeBound = true;
    var btn = document.getElementById("themeToggle");
    if (btn) {
      btn.addEventListener("click", function () {
        var cur = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
        document.documentElement.setAttribute("data-theme", cur);
        localStorage.setItem("ainstein-theme", cur);
        window.dispatchEvent(new Event("themechange"));
      });
    }
    var path = window.location.pathname;
    document.querySelectorAll(".nav-links a").forEach(function (a) {
      var nav = a.getAttribute("data-nav");
      if ((nav === "demo" && path.startsWith("/demo")) || (nav === "dashboard" && path === "/")) {
        a.classList.add("active");
      }
    });
    setupNavScroll();
    setupReveal();
  }

  // Hairline border on the nav once the page scrolls.
  function setupNavScroll() {
    var nav = document.getElementById("nav");
    if (!nav) return;
    function onScroll() { nav.classList.toggle("scrolled", window.scrollY > 4); }
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
  }

  // Reveal sections as they enter the viewport (skipped when reduced motion).
  function setupReveal() {
    var els = document.querySelectorAll(".reveal");
    if (!els.length) return;
    if (reducedMotion() || !("IntersectionObserver" in window)) {
      els.forEach(function (el) { el.classList.add("in"); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -8% 0px" });
    els.forEach(function (el) { io.observe(el); });
  }

  function pct(v) { return v === null || v === undefined ? "—" : (Number(v) * 100).toFixed(1) + "%"; }
  function num(v) { return v === null || v === undefined ? "—" : Number(v).toFixed(3); }
  function tierChipClass(t) {
    var k = (t || "").toLowerCase();
    return "chip chip-" + (["oral", "spotlight", "poster"].includes(k) ? k : "poster");
  }
  async function getJSON(url) {
    // Static mode (GitHub Pages): no backend — map /api/x?… to api/x.json files.
    if (window.AINSTEIN_STATIC) {
      url = "api/" + url.split("?")[0].replace(/^\/api\//, "").replace(/\/$/, "") + ".json";
    }
    var r = await fetch(url); return r.json();
  }

  // ---- Dashboard ------------------------------------------------------------
  var charts = [];
  function tickFont() { return { color: cssVar("--text-dim"), font: { family: "-apple-system, Inter, sans-serif", size: 12 } }; }
  function gridColor() { return cssVar("--border"); }

  function statCard(label, value, sub) {
    return '<div class="stat fade-in"><div class="label">' + label + '</div>' +
      '<div class="value">' + value + '</div><div class="sub">' + (sub || "") + '</div></div>';
  }

  async function initDashboard() {
    setupTheme();
    var summaryResp = await getJSON("/api/summary");
    var s = summaryResp.summary || {};
    var cards = document.getElementById("statCards");
    if (cards) {
      cards.innerHTML =
        statCard("Papers evaluated", s.num_papers != null ? s.num_papers : "—", (s.num_errors || 0) + " errors") +
        statCard("Success (relaxed)", pct(s.success_rate_relaxed), "at least one judge approves") +
        statCard("Success (strict)", pct(s.success_rate_strict), "both judges approve") +
        statCard("Rediscovery", pct(s.rediscovery_relaxed), "matches the paper's idea") +
        statCard("Novel &amp; valid", pct(s.novel_and_valid_relaxed), "valid but different") +
        statCard("Judge agreement", pct(s.judge_agreement), "inter-judge consensus");
    }

    var tiersResp = await getJSON("/api/tiers");
    var tiers = tiersResp.tiers || [];
    var baseResp = await getJSON("/api/baselines");
    var baselines = baseResp.baselines || [];

    buildTierChart(tiers);
    buildJudgeChart(tiers);
    buildBaselineChart(baselines);
    await loadPapers();

    var search = document.getElementById("search");
    var tierFilter = document.getElementById("tierFilter");
    var t;
    function reload() { clearTimeout(t); t = setTimeout(loadPapers, 220); }
    if (search) search.addEventListener("input", reload);
    if (tierFilter) tierFilter.addEventListener("change", loadPapers);

    window.addEventListener("themechange", function () {
      charts.forEach(function (c) { c.destroy(); });
      charts = [];
      buildTierChart(tiers); buildJudgeChart(tiers); buildBaselineChart(baselines);
    });
  }

  // Shared bar styling: rounded bars, hairline y grid only, restrained legend.
  function baseOptions() {
    return {
      responsive: true, maintainAspectRatio: false,
      layout: { padding: { top: 6 } },
      plugins: {
        legend: { labels: { color: cssVar("--text-dim"), boxWidth: 10, boxHeight: 10,
          usePointStyle: true, font: { family: "-apple-system, Inter, sans-serif", size: 12 } } },
        tooltip: {
          backgroundColor: cssVar("--ink"), titleColor: cssVar("--bg"), bodyColor: cssVar("--bg"),
          padding: 11, cornerRadius: 8, displayColors: false,
          callbacks: { label: function (c) { return c.dataset.label + ": " + (c.parsed.y * 100).toFixed(1) + "%"; } },
        },
      },
      scales: {
        x: { ticks: tickFont(), grid: { display: false }, border: { color: gridColor() } },
        y: { ticks: { color: cssVar("--text-faint"), font: { size: 11 },
               callback: function (v) { return (v * 100) + "%"; } },
             grid: { color: gridColor() }, border: { display: false }, beginAtZero: true, max: 1 },
      },
      animation: { duration: reducedMotion() ? 0 : 750, easing: "easeOutQuart" },
    };
  }

  var BAR = { borderRadius: 6, borderSkipped: false, maxBarThickness: 46 };

  function buildTierChart(tiers) {
    var el = document.getElementById("tierChart"); if (!el) return;
    var labels = tiers.map(function (r) { return r.tier; });
    charts.push(new Chart(el, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          Object.assign({ label: "Success (relaxed)", data: tiers.map(function (r) { return r.success_rate_relaxed; }), backgroundColor: cssVar("--brand") }, BAR),
          Object.assign({ label: "Success (strict)", data: tiers.map(function (r) { return r.success_rate_strict; }), backgroundColor: cssVar("--brand-2") }, BAR),
          Object.assign({ label: "Rediscovery", data: tiers.map(function (r) { return r.rediscovery_relaxed; }), backgroundColor: cssVar("--accent") }, BAR),
        ],
      },
      options: baseOptions(),
    }));
  }

  function buildJudgeChart(tiers) {
    var el = document.getElementById("judgeChart"); if (!el) return;
    charts.push(new Chart(el, {
      type: "bar",
      data: {
        labels: tiers.map(function (r) { return r.tier; }),
        datasets: [Object.assign({ label: "Judge agreement", data: tiers.map(function (r) { return r.judge_agreement; }), backgroundColor: cssVar("--warn") }, BAR)],
      },
      options: baseOptions(),
    }));
  }

  function buildBaselineChart(baselines) {
    var el = document.getElementById("baselineChart"); if (!el) return;
    charts.push(new Chart(el, {
      type: "bar",
      data: {
        labels: baselines.map(function (r) { return r.method_name; }),
        datasets: [
          Object.assign({ label: "Success (relaxed)", data: baselines.map(function (r) { return r.success_rate_relaxed; }), backgroundColor: cssVar("--brand") }, BAR),
          Object.assign({ label: "Rediscovery", data: baselines.map(function (r) { return r.rediscovery_relaxed; }), backgroundColor: cssVar("--accent") }, BAR),
        ],
      },
      options: baseOptions(),
    }));
  }

  async function loadPapers() {
    var rows = document.getElementById("paperRows"); if (!rows) return;
    var tier = (document.getElementById("tierFilter") || {}).value || "";
    var q = ((document.getElementById("search") || {}).value || "").toLowerCase();
    var data = await getJSON("/api/papers?limit=500" + (tier ? "&tier=" + encodeURIComponent(tier) : ""));
    var items = (data.items || []).filter(function (p) {
      return (!q || String(p.title || "").toLowerCase().includes(q)) &&
             (!tier || String(p.tier || "") === tier);
    });
    if (!items.length) {
      rows.innerHTML = '<tr><td colspan="6" style="color:var(--text-faint);padding:22px;">No matching papers.</td></tr>';
      return;
    }
    rows.innerHTML = items.map(function (p) {
      var href = window.AINSTEIN_STATIC
        ? "paper-" + encodeURIComponent(p.paper_id) + ".html"
        : "/paper/" + encodeURIComponent(p.paper_id);
      return '<tr onclick="window.location=\'' + href + '\'">' +
        '<td class="title">' + (p.title || p.paper_id) + '</td>' +
        '<td><span class="' + tierChipClass(p.tier) + '">' + (p.tier || "—") + '</span></td>' +
        '<td>' + (p.success_rate_relaxed ? "✓" : "—") + '</td>' +
        '<td>' + (p.rediscovery_relaxed ? "✓" : "—") + '</td>' +
        '<td>' + (p.judge_agreement ? "✓" : "—") + '</td>' +
        '<td>' + num(p.token_f1) + '</td></tr>';
    }).join("");
  }

  // ---- Metric bars (paper page) ---------------------------------------------
  function renderMetricBars(elId, metrics) {
    var el = document.getElementById(elId); if (!el) return;
    el.innerHTML = Object.keys(metrics).map(function (label) {
      var v = Math.max(0, Math.min(1, Number(metrics[label]) || 0));
      return '<div class="metric-row"><div class="m-label">' + label + '</div>' +
        '<div class="m-track"><div class="m-fill" style="width:0%"></div></div>' +
        '<div class="m-val">' + v.toFixed(3) + '</div></div>';
    }).join("");
    requestAnimationFrame(function () {
      var fills = el.querySelectorAll(".m-fill");
      Object.keys(metrics).forEach(function (label, i) {
        var v = Math.max(0, Math.min(1, Number(metrics[label]) || 0));
        if (fills[i]) fills[i].style.width = (v * 100).toFixed(1) + "%";
      });
    });
  }

  // ---- Live demo ------------------------------------------------------------
  function initDemo() {
    setupTheme();
    var btn = document.getElementById("runBtn"); if (!btn) return;
    btn.addEventListener("click", startRun);
  }

  function renderStages(stages) {
    var list = document.getElementById("stageList"); if (!list) return;
    var icon = { pending: "○", running: "", done: "✓", failed: "✕" };
    list.innerHTML = stages.map(function (s) {
      var inner = s.state === "running" ? '<span class="spinner"></span>' : icon[s.state];
      return '<div class="stage ' + s.state + '"><div class="st-ico">' + inner + '</div>' +
        '<div class="st-name">' + s.name + '</div></div>';
    }).join("");
  }

  async function startRun() {
    var btn = document.getElementById("runBtn");
    var hint = document.getElementById("runHint");
    var rowIndex = document.getElementById("rowIndex").value;
    var paperId = document.getElementById("paperId").value.trim();
    btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Running…';
    document.getElementById("progressSection").style.display = "block";
    document.getElementById("resultSection").style.display = "none";

    var payload = {};
    if (paperId) payload.paper_id = paperId;
    else if (rowIndex !== "") payload.row_index = Number(rowIndex);

    var resp = await fetch("/api/demo/run", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    if (resp.status === 503 || resp.status === 409) {
      var err = await resp.json();
      hint.textContent = err.detail;
      btn.disabled = false; btn.textContent = "Run pipeline";
      return;
    }
    var job = await resp.json();
    renderStages(job.stages);
    poll(job.id);
  }

  async function poll(jobId) {
    var btn = document.getElementById("runBtn");
    var job = await getJSON("/api/demo/" + jobId);
    renderStages(job.stages);
    if (job.status === "running" || job.status === "pending") {
      setTimeout(function () { poll(jobId); }, 1500);
      return;
    }
    btn.disabled = false; btn.textContent = "Run pipeline";
    var box = document.getElementById("resultBox");
    document.getElementById("resultSection").style.display = "block";
    if (job.status === "failed") {
      box.innerHTML = '<div class="panel"><h4>Run failed</h4><div class="body">' + (job.error || "Unknown error") + '</div></div>';
      return;
    }
    var r = job.result || {};
    var badge = r.critique_passed
      ? '<span class="badge ok"><span class="dot"></span>Passed critique</span>'
      : '<span class="badge no"><span class="dot"></span>Best-effort (critique not passed)</span>';
    box.innerHTML =
      '<div style="margin-bottom:14px;">' + badge + '</div>' +
      '<div class="sol-grid">' +
        '<div class="panel"><h4>Problem statement</h4><div class="body">' + (r.problem_statement || "—") + '</div></div>' +
        '<div class="panel"><h4>AI-generated solution</h4><div class="body">' + (r.solution || "—") + '</div></div>' +
      '</div>' +
      (r.evaluation ? '<div class="panel" style="margin-top:16px;"><h4>Evaluation</h4><div class="body">' + r.evaluation + '</div></div>' : "");
  }

  // Expose + init
  window.AINSTEIN = { initDashboard: initDashboard, initDemo: initDemo, renderMetricBars: renderMetricBars };
  document.addEventListener("DOMContentLoaded", setupTheme);
})();

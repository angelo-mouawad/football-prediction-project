const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const api = async (path, params = {}) => {
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([k, v]) => {
    if (v !== null && v !== undefined && v !== "") url.searchParams.set(k, v);
  });
  const response = await fetch(url);
  const data = await response.json().catch(() => ({ error: "Bad response from the server" }));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
};

const pct = (x) => `${(x * 100).toFixed(1)}%`;
const signed = (x) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(1)}`;

const money = (x) => {
  if (x === null || x === undefined || Number.isNaN(x)) return "unknown";
  if (x >= 1e6) return `€${(x / 1e6).toFixed(1)}m`;
  if (x >= 1e3) return `€${Math.round(x / 1e3)}k`;
  return `€${Math.round(x)}`;
};

const escapeHtml = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));

const skeleton = (n = 1) => Array.from({ length: n }, () => '<div class="skeleton"></div>').join("");
const notice = (msg) => `<div class="notice">${escapeHtml(msg)}</div>`;

$$(".rail-link").forEach((link) => {
  link.addEventListener("click", () => {
    $$(".rail-link").forEach((l) => l.classList.remove("is-active"));
    $$(".view").forEach((v) => v.classList.remove("is-active"));
    link.classList.add("is-active");
    $(`#view-${link.dataset.view}`).classList.add("is-active");
  });
});

const loadStatus = async () => {
  try {
    const { available, match_metrics } = await api("/api/status");
    const labels = { match: "match model", value: "value model", similar: "similarity index", news: "news branch" };
    const strip = Object.entries(labels).map(([key, label]) => `
      <div class="status-row"><span class="dot ${available[key] ? "on" : "off"}"></span>${label}</div>
    `).join("");
    const acc = match_metrics && match_metrics.accuracy
      ? `<div class="status-row">accuracy ${pct(match_metrics.accuracy)}</div>` : "";
    $("#status-strip").innerHTML = strip + acc;
  } catch (err) {
    $("#status-strip").innerHTML = `<div class="status-row"><span class="dot off"></span>backend offline</div>`;
  }
};

const loadTeams = async () => {
  try {
    const { current } = await api("/api/teams");
    const options = current.map((t) => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join("");
    $("#home-team").innerHTML = options;
    $("#away-team").innerHTML = options;
    if (current.length > 1) $("#away-team").selectedIndex = 1;
  } catch (err) {
    $("#match-output").innerHTML = notice(err.message);
  }
};

const loadFixtures = async () => {
  try {
    const { fixtures } = await api("/api/fixtures", { limit: 12 });
    $("#fixture-rail").innerHTML = fixtures.map((f) => `
      <button class="fixture-chip" data-home="${escapeHtml(f.home_team)}" data-away="${escapeHtml(f.away_team)}">
        <b>${escapeHtml(f.home_team)}</b> v <b>${escapeHtml(f.away_team)}</b>
        <span>${escapeHtml(f.date)}</span>
      </button>
    `).join("");
    $$(".fixture-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        $("#home-team").value = chip.dataset.home;
        $("#away-team").value = chip.dataset.away;
        runPrediction();
      });
    });
  } catch (err) {
    $("#fixture-rail").innerHTML = "";
  }
};

const barRow = (name, klass, final, network) => `
  <div class="bar-row">
    <div class="name">${escapeHtml(name)}</div>
    <div class="bar-track">
      <div class="bar-fill ${klass}" data-width="${(final * 100).toFixed(1)}"></div>
      <div class="bar-ghost" data-width="${(network * 100).toFixed(1)}"></div>
    </div>
    <div>
      <div class="pct">${pct(final)}</div>
      <div class="delta ${final - network > 0.001 ? "up" : final - network < -0.001 ? "down" : "neutral"}">
        ${signed(final - network)}
      </div>
    </div>
  </div>
`;

const renderNews = (data) => {
  const facts = data.facts || [];
  const articles = data.articles || [];
  const scoreChip = (label, score) => `
    <div class="news-chip ${score > 0 ? "up" : score < 0 ? "down" : "neutral"}">
      ${escapeHtml(label)} ${score > 0 ? "+" : ""}${score.toFixed(2)}
    </div>
  `;

  if (!facts.length && !articles.length) {
    return `<div class="panel">
      <p class="section-title">News branch</p>
      <p class="muted">${escapeHtml(data.news_explanation || "No team news found. This is the network alone.")}</p>
    </div>`;
  }

  return `<div class="panel">
    <p class="section-title">News branch</p>
    <div class="news-scores">
      ${scoreChip(data.home_team, data.news_home)}
      ${scoreChip(data.away_team, data.news_away)}
    </div>
    ${facts.map((f) => `
      <div class="fact">
        <div class="who">
          <b>${escapeHtml(f.player)}</b>
          <span>${escapeHtml(f.status)} · ${f.side === "home" ? escapeHtml(data.home_team) : escapeHtml(data.away_team)}</span>
        </div>
        <div class="meter"><i data-width="${(f.importance * 100).toFixed(0)}"></i></div>
        <div class="impact ${f.impact > 0 ? "up" : f.impact < 0 ? "down" : "neutral"}">${f.impact > 0 ? "+" : ""}${f.impact.toFixed(2)}</div>
      </div>
    `).join("")}
    ${articles.length ? `<div class="links">${articles.map((a) => `
      <a href="${escapeHtml(a.href)}" target="_blank" rel="noopener">${escapeHtml(a.title)}</a>
    `).join("")}</div>` : ""}
  </div>`;
};

const animateWidths = (root) => {
  requestAnimationFrame(() => {
    $$("[data-width]", root).forEach((el) => { el.style.width = `${el.dataset.width}%`; });
  });
};

const runPrediction = async () => {
  const home = $("#home-team").value;
  const away = $("#away-team").value;
  const button = $("#run-predict");
  const out = $("#match-output");

  if (home === away) {
    out.innerHTML = notice("Pick two different teams");
    return;
  }

  button.disabled = true;
  button.querySelector("span").textContent = $("#use-news").checked ? "Reading the news" : "Running";
  out.innerHTML = skeleton(2);

  try {
    const data = await api("/api/predict", { home, away, news: $("#use-news").checked });
    const n = data.network;
    const f = data.final;

    out.innerHTML = `
      <div class="panel">
        <div class="verdict">
          <div>
            <p class="section-title">Verdict</p>
            <h2>${escapeHtml(data.verdict)}</h2>
          </div>
          <div class="confidence">${pct(data.confidence)}</div>
        </div>
        <div class="bars">
          ${barRow(`${data.home_team} win`, "fill-home", f.home_win, n.home_win)}
          ${barRow("Draw", "fill-draw", f.draw, n.draw)}
          ${barRow(`${data.away_team} win`, "fill-away", f.away_win, n.away_win)}
        </div>
        <p class="muted mono" style="margin-top:18px;font-size:11px">
          Dashed marker is the neural network alone. The number underneath is what the news moved it by.
        </p>
      </div>
      ${renderNews(data)}
    `;
    animateWidths(out);
  } catch (err) {
    out.innerHTML = notice(err.message);
  } finally {
    button.disabled = false;
    button.querySelector("span").textContent = "Run prediction";
  }
};

$("#run-predict").addEventListener("click", runPrediction);

const wireSearch = (inputSel, boxSel, source, onPick) => {
  const input = $(inputSel);
  const box = $(boxSel);
  let timer;

  const close = () => box.classList.remove("is-open");

  input.addEventListener("input", () => {
    clearTimeout(timer);
    const q = input.value.trim();
    if (q.length < 2) return close();
    timer = setTimeout(async () => {
      try {
        const { players } = await api("/api/players", { q, source });
        if (!players.length) return close();
        box.innerHTML = players.map((p) => `
          <div class="suggestion" data-name="${escapeHtml(p.player)}">
            <span>${escapeHtml(p.player)}</span>
            <small>${escapeHtml(p.team || "")}${p.age ? ` · ${Math.round(p.age)}` : ""}</small>
          </div>
        `).join("");
        box.classList.add("is-open");
        $$(".suggestion", box).forEach((s) => {
          s.addEventListener("click", () => {
            input.value = s.dataset.name;
            close();
            onPick(s.dataset.name);
          });
        });
      } catch (err) {
        close();
      }
    }, 180);
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { close(); onPick(input.value.trim()); }
    if (e.key === "Escape") close();
  });

  document.addEventListener("click", (e) => {
    if (!box.contains(e.target) && e.target !== input) close();
  });
};

const showValue = async (name) => {
  const out = $("#value-output");
  if (!name) return;
  out.innerHTML = skeleton(1);
  try {
    const d = await api("/api/value", { player: name });
    const over = d.gap !== null && d.gap > 0;
    out.innerHTML = `
      <div class="panel">
        <div class="value-hero">
          <div>
            <p class="label">Model valuation</p>
            <div class="big-number accent">${money(d.predicted)}</div>
            <div class="chips">
              <div class="chip">${escapeHtml(d.club || "")}</div>
              <div class="chip">${escapeHtml(d.position || "")}</div>
              ${d.age ? `<div class="chip">age <b>${Math.round(d.age)}</b></div>` : ""}
              ${d.minutes ? `<div class="chip"><b>${Math.round(d.minutes)}</b> minutes</div>` : ""}
              ${d.goals !== null ? `<div class="chip"><b>${d.goals}</b> goals</div>` : ""}
              ${d.assists !== null ? `<div class="chip"><b>${d.assists}</b> assists</div>` : ""}
            </div>
          </div>
          <div>
            <p class="label">Transfermarkt</p>
            <div class="big-number">${money(d.actual)}</div>
            ${d.gap !== null ? `
              <p class="mono ${over ? "up" : "down"}" style="margin-top:10px">
                model is ${Math.abs(d.gap * 100).toFixed(0)}% ${over ? "higher" : "lower"}
              </p>` : ""}
            <p class="muted" style="font-size:12px">
              Median player that season: ${money(d.market_level)}
            </p>
          </div>
        </div>
        <div class="bars" style="margin-top:24px">
          <div class="bar-row">
            <div class="name">Model</div>
            <div class="bar-track"><div class="bar-fill fill-home" data-width="${Math.min(100, (d.predicted / Math.max(d.predicted, d.actual)) * 100).toFixed(1)}"></div></div>
            <div class="pct">${money(d.predicted)}</div>
          </div>
          <div class="bar-row">
            <div class="name">Market</div>
            <div class="bar-track"><div class="bar-fill fill-draw" data-width="${Math.min(100, (d.actual / Math.max(d.predicted, d.actual)) * 100).toFixed(1)}"></div></div>
            <div class="pct">${money(d.actual)}</div>
          </div>
        </div>
      </div>
    `;
    animateWidths(out);
  } catch (err) {
    out.innerHTML = notice(err.message);
  }
};

const radarSvg = (points) => {
  if (!points.length) return "";
  const size = 300;
  const mid = size / 2;
  const radius = mid - 54;
  const step = (Math.PI * 2) / points.length;
  const at = (i, value) => {
    const angle = i * step - Math.PI / 2;
    const r = (Math.max(0, Math.min(100, value)) / 100) * radius;
    return [mid + Math.cos(angle) * r, mid + Math.sin(angle) * r];
  };
  const ring = (frac) => points.map((_, i) => at(i, frac * 100).join(",")).join(" ");
  const shape = (key) => points.map((p, i) => at(i, p[key]).join(",")).join(" ");

  const labels = points.map((p, i) => {
    const [x, y] = at(i, 118);
    const anchor = x < mid - 10 ? "end" : x > mid + 10 ? "start" : "middle";
    return `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="${anchor}" fill="#5c6788" font-size="9">${escapeHtml(p.feature)}</text>`;
  }).join("");

  return `<svg viewBox="0 0 ${size} ${size}" width="100%" style="max-width:320px">
    ${[0.25, 0.5, 0.75, 1].map((f) => `<polygon points="${ring(f)}" fill="none" stroke="rgba(126,160,255,0.12)"/>`).join("")}
    <polygon points="${shape("query")}" fill="rgba(53,230,255,0.22)" stroke="#35e6ff" stroke-width="2"/>
    <polygon points="${shape("match")}" fill="rgba(139,92,246,0.18)" stroke="#8b5cf6" stroke-width="2"/>
    ${labels}
  </svg>`;
};

const showSimilar = async (name) => {
  const out = $("#replace-output");
  if (!name) return;
  const maxAge = Number($("#max-age").value);
  const budget = Number($("#max-value").value);

  out.innerHTML = skeleton(2);
  try {
    const d = await api("/api/similar", {
      player: name,
      n: 10,
      max_age: maxAge >= 40 ? "" : maxAge,
      max_value: budget >= 200 ? "" : budget * 1e6,
    });

    const q = d.query;
    const list = d.matches.length ? d.matches.map((m, i) => `
      <div class="match-row" style="animation-delay:${i * 45}ms">
        <div class="rank">${String(i + 1).padStart(2, "0")}</div>
        <div class="who"><b>${escapeHtml(m.player)}</b><small>${escapeHtml(m.team)} · ${escapeHtml(m.position)}</small></div>
        <div class="num hide-sm">${Math.round(m.minutes)} min · age ${Math.round(m.age)}</div>
        <div class="num hide-sm">${money(m.market_value_eur)}</div>
        <div class="num sim">${m.similarity.toFixed(3)}</div>
      </div>
    `).join("") : `<p class="muted">Nobody passed those filters. Widen the budget or the age limit.</p>`;

    out.innerHTML = `
      <div class="panel">
        <p class="section-title">Replacing</p>
        <div class="verdict">
          <div><h2>${escapeHtml(q.player)}</h2>
            <p class="muted">${escapeHtml(q.team)} · ${escapeHtml(q.position)} · ${Math.round(q.minutes)} minutes${q.age ? ` · age ${Math.round(q.age)}` : ""}</p>
          </div>
          <div class="confidence">${money(q.value)}</div>
        </div>
      </div>
      <div class="panel">
        <p class="section-title">Closest by playing style</p>
        <div class="stack" style="margin:0;gap:8px">${list}</div>
      </div>
      ${d.radar && d.radar.length ? `
      <div class="split">
        <div class="panel radar-wrap">
          <div style="width:100%">
            <p class="section-title" style="text-align:center">Percentile profile</p>
            ${radarSvg(d.radar)}
            <div class="legend">
              <span><i style="background:#35e6ff"></i>${escapeHtml(q.player)}</span>
              <span><i style="background:#8b5cf6"></i>${escapeHtml(d.top_match || "")}</span>
            </div>
          </div>
        </div>
        <div class="panel">
          <p class="section-title">Stat by stat</p>
          ${d.comparison.map((c) => `
            <div class="compare-row">
              <div class="feature">${escapeHtml(c.feature)}</div>
              <span style="color:#35e6ff">${Math.round(c.query)}</span>
              <span style="color:#8b5cf6">${Math.round(c.match)}</span>
            </div>
          `).join("")}
        </div>
      </div>` : ""}
    `;
  } catch (err) {
    out.innerHTML = notice(err.message);
  }
};

$("#max-age").addEventListener("input", (e) => {
  $("#age-readout").textContent = e.target.value >= 40 ? "any" : e.target.value;
});
$("#max-value").addEventListener("input", (e) => {
  $("#budget-readout").textContent = e.target.value >= 200 ? "any" : `€${e.target.value}m`;
});

let lastSimilar = null;
const rerunSimilar = () => { if (lastSimilar) showSimilar(lastSimilar); };
$("#max-age").addEventListener("change", rerunSimilar);
$("#max-value").addEventListener("change", rerunSimilar);

wireSearch("#value-search", "#value-suggestions", "value", showValue);
wireSearch("#replace-search", "#replace-suggestions", "profiles", (name) => {
  lastSimilar = name;
  showSimilar(name);
});

loadStatus();
loadTeams();
loadFixtures();

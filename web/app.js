const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, c => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const TYPE_LABEL = {
  eu_law: "EU regulation",
  ie_legislation: "Acts of the Oireachtas",
  ie_guidance: "Irish immigration guidance",
  case_law: "CJEU case law",
  philosophy: "Legal philosophy",
};

const EXAMPLES = [
  "What is the minimum salary threshold for an Irish EU Blue Card?",
  "Under Article 6 of the AI Act, when is an AI system high-risk?",
  "Can the non-EEA spouse of an EU citizen be refused residence in Ireland?",
  "How does Ireland implement the EU AI Act in national law?",
];

const state = { mode: "agentic", fusion: "hybrid", types: [], last: null };

/* ---------------------------------------------------------------- theme --- */
$("#theme-toggle").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme !== "dark";
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  try { localStorage.setItem("theme", dark ? "dark" : "light"); } catch (e) {}
});

/* ------------------------------------------------------------ segments --- */
document.querySelectorAll(".seg").forEach(group => {
  group.addEventListener("click", e => {
    const btn = e.target.closest("button");
    if (!btn) return;
    group.querySelectorAll("button").forEach(b =>
      b.setAttribute("aria-pressed", String(b === btn)));
    if (btn.dataset.mode) state.mode = btn.dataset.mode;
    if (btn.dataset.fusion) state.fusion = btn.dataset.fusion;
  });
});

/* ------------------------------------------------------ provenance spine --- */
/* Orange above, teal below, split by which retriever found this source. The
   bar is the legend: no key needed once you have seen two rows. */
function spine(hit) {
  const lex = hit.bm25_rank != null, sem = hit.dense_rank != null;
  if (lex && sem) return '<span class="spine"><i class="lex"></i><i class="sem"></i></span>';
  if (lex) return '<span class="spine"><i class="lex"></i></span>';
  if (sem) return '<span class="spine"><i class="sem"></i></span>';
  return '<span class="spine"><i class="none"></i></span>';
}

function badge(hit) {
  const lex = hit.bm25_rank != null, sem = hit.dense_rank != null;
  if (lex && sem) return '<span class="badge both">both</span>';
  if (lex) return '<span class="badge bm25">lexical</span>';
  if (sem) return '<span class="badge dense">semantic</span>';
  return '<span class="badge agent">retrieved</span>';
}

const rank = (v) => (v == null ? "—" : "#" + v);

/* --------------------------------------------------------------- render --- */
function renderAnswer(d) {
  if (d.abstained) {
    const rest = d.answer.replace(/^The corpus does not contain an answer to this\.?\s*/i, "");
    return `<div class="refusal">
      <svg width="40" height="40" viewBox="0 0 40 40" aria-hidden="true">
        <rect x="1.5" y="1.5" width="37" height="37" rx="5" fill="none"
              stroke="var(--refusal)" stroke-width="2.5"/>
        <path d="M12 12 L28 28 M28 12 L12 28" stroke="var(--refusal)"
              stroke-width="2.5" stroke-linecap="square"/>
      </svg>
      <div>
        <h3>Not in the corpus</h3>
        <p>${esc(rest) || "No source supports an answer to this."}</p>
      </div>
    </div>`;
  }
  // [n] becomes a real pointer to the evidence below
  // escaped first, so this cannot introduce markup from the model
  const html = esc(d.answer)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d+)\]/g, (_, n) => `<a class="cite" href="#src-${n}">${n}</a>`)
    .split(/\n{2,}/).map(p => `<p>${p}</p>`).join("");
  return `<div class="answer">${html}</div>`;
}

function renderMargin(d) {
  const s = d.scores || {};
  const pct = v => v == null ? "n/a" : Math.round(v * 100) + "%";
  const num = v => v == null ? "n/a" : v.toFixed(2);

  let out = `<div>
    <div class="label">Quality</div>
    <dl class="metrics">
      <div class="metric"><dt>Grounding</dt><dd>${num(s.grounding)}</dd></div>
      <div class="metric"><dt>Citations</dt><dd>${pct(s.citation_valid)}</dd></div>
    </dl>
    <p class="note" style="margin-top:var(--s-1)">Grounding measures how closely the
      answer resembles its sources — a similarity signal, not a truth signal. The
      judge below is the real measurement.</p>
    <button class="btn" id="judge" style="margin-top:var(--s-2)">Grade every claim</button>
    <div id="verdict"></div>
  </div>`;

  if (d.trace && d.trace.length > 1) {
    out += `<div><div class="label">Searches run</div><div class="trace">` +
      d.trace.map(t => `<div class="trace-step">
        <code>${esc(t.query)}</code>
        <span>${esc(TYPE_LABEL[t.source_type] || "all sources")} → ${t.n_new}</span>
      </div>`).join("") + `</div></div>`;
  }
  return out;
}

function renderSources(hits) {
  if (!hits.length) return "";
  return `<section class="evidence">
    <div class="label">Evidence — ${hits.length} sources</div>
    ${hits.map(h => `<article class="source" id="src-${h.n}">
      ${spine(h)}
      <div>
        <div class="source-head">[${h.n}] ${esc(h.citation)}</div>
        <div class="source-meta">bm25 ${rank(h.bm25_rank)} · dense ${rank(h.dense_rank)}</div>
        ${h.note ? `<div class="source-note">${esc(h.note)}</div>` : ""}
        <p class="source-text">${/^[a-z]/.test(h.text) ? "…" : ""}${esc(h.text.slice(0, 420))}${h.text.length > 420 ? "…" : ""}</p>
        ${h.url ? `<a href="${esc(h.url)}" target="_blank" rel="noopener">Open source ↗</a>` : ""}
      </div>
      ${badge(h)}
    </article>`).join("")}
  </section>`;
}

function renderVerdict(v) {
  if (v.faithfulness == null) return `<p class="note" style="margin-top:var(--s-2)">Nothing to grade.</p>`;
  const icon = { supported: "✓", unsupported: "?", contradicted: "✕" };
  return `<dl class="metrics" style="margin-top:var(--s-2)">
      <div class="metric"><dt>Faithful</dt><dd>${Math.round(v.faithfulness * 100)}%</dd></div>
      <div class="metric"><dt>Hallucinated</dt><dd>${Math.round(v.hallucination_rate * 100)}%</dd></div>
    </dl>
    <div class="trace" style="margin-top:var(--s-2)">
      ${v.claims.map(c => `<div class="trace-step">
        <code>${icon[c.verdict] || "?"}</code><span>${esc(c.claim)}</span>
      </div>`).join("")}
    </div>`;
}

/* ------------------------------------------------------------------ ask --- */
const STEP_TEXT = {
  retrieving: () => "Searching the corpus",
  search: (e) => `Searched <code>${esc(e.query)}</code> in ${esc(TYPE_LABEL[e.source_type] || "all sources")} — ${e.n_new} new sources`,
  writing: (e) => `Reading ${e.n} sources and writing the answer`,
  scoring: () => "Scoring the answer against its sources",
};

/* The agentic path makes several model calls and takes 10-25s. Showing the
   steps as they land turns a blank wait into something worth watching, and it
   happens to be the most interesting part of the system to watch. */
function renderProgress(steps, finished) {
  return `<div class="progress">` + steps.map((s, i) => {
    const last = i === steps.length - 1 && !finished;
    return `<div class="step${last ? " live" : ""}">
      <span class="step-mark">${last ? "" : "✓"}</span>
      <span>${s}</span>
    </div>`;
  }).join("") + `</div>`;
}

async function ask(question) {
  const out = $("#out");
  const steps = [];
  out.innerHTML = renderProgress(["Starting"], false);
  $("#ask").disabled = true;

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question, mode: state.mode, fusion: state.fusion,
        source_types: state.types, k: 8,
      }),
    });

    // Refusals (400 / 429 / 503) come back as plain JSON, not a stream.
    if (!res.ok) {
      const d = await res.json().catch(() => ({ detail: "Request refused." }));
      out.innerHTML = `<div class="notice">${esc(d.detail || "Request refused.")}</div>`
        + (d.hits ? renderSources(d.hits) : "");
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "", result = null, failed = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const ev = JSON.parse(line);
        if (ev.done) { result = ev.done; continue; }
        if (ev.error) { failed = ev.error; continue; }
        const fmt = STEP_TEXT[ev.step];
        if (fmt) {
          steps.push(fmt(ev));
          out.innerHTML = renderProgress(steps, false);
        }
      }
    }

    if (failed) {
      out.innerHTML = `<div class="error">Could not answer: ${esc(failed)}</div>`;
      return;
    }
    if (!result) {
      out.innerHTML = `<div class="error">The answer stream ended early.</div>`;
      return;
    }

    state.last = result;
    out.innerHTML = `<div class="result enter">
        <div>${renderAnswer(result)}</div>
        <aside class="margin">${renderMargin(result)}</aside>
      </div>${renderSources(result.hits)}`;

    const judgeBtn = $("#judge");
    if (judgeBtn) judgeBtn.addEventListener("click", grade);
  } catch (err) {
    out.innerHTML = `<div class="error">Could not answer: ${esc(err.message).slice(0, 300)}</div>`;
  } finally {
    $("#ask").disabled = false;
  }
}

async function grade() {
  const btn = $("#judge");
  btn.disabled = true;
  btn.textContent = "Grading…";
  try {
    const res = await fetch("/api/judge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        answer: state.last.answer,
        hit_ids: state.last.hits.map(h => h.id),
      }),
    });
    const v = await res.json().catch(() => ({}));
    if (!res.ok) {
      $("#verdict").innerHTML = `<div class="notice" style="margin-top:var(--s-2)">${esc(v.detail || "Refused.")}</div>`;
      btn.disabled = false;
      btn.textContent = "Grade every claim";
      return;
    }
    $("#verdict").innerHTML = renderVerdict(v);
    btn.remove();
  } catch (err) {
    $("#verdict").innerHTML = `<div class="error" style="margin-top:var(--s-2)">${esc(err.message).slice(0, 200)}</div>`;
    btn.disabled = false;
    btn.textContent = "Grade every claim";
  }
}

$("#composer").addEventListener("submit", e => {
  e.preventDefault();
  const q = $("#q").value.trim();
  if (q) ask(q);
});

$("#q").addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) $("#composer").requestSubmit();
});

/* ----------------------------------------------------------------- boot --- */
$("#examples").innerHTML = EXAMPLES.map(e =>
  `<button class="example" type="button">${esc(e)}</button>`).join("");
$("#examples").addEventListener("click", e => {
  const b = e.target.closest(".example");
  if (!b) return;
  $("#q").value = b.textContent;
  $("#composer").requestSubmit();
});

function boot(attempt = 0) {
  fetch("/api/meta").then(r => r.ok ? r.json() : Promise.reject(r.status)).then(m => {
    state.types = m.law_types.slice();
    $("#corpus-stat").textContent =
      `${m.chunks.toLocaleString()} chunks · ${m.instruments.toLocaleString()} instruments`;
    $("#types").innerHTML = Object.keys(TYPE_LABEL)
      .filter(t => m.by_type[t])
      .map(t => `<button class="chip" type="button" data-type="${t}"
         aria-pressed="${state.types.includes(t)}"><span class="dot"></span>${TYPE_LABEL[t]}
         <span style="color:var(--text-4);font-family:var(--mono);font-size:11px">${m.by_type[t].toLocaleString()}</span></button>`)
      .join("");
    $("#types").addEventListener("click", e => {
      const b = e.target.closest(".chip");
      if (!b) return;
      const t = b.dataset.type, on = b.getAttribute("aria-pressed") === "true";
      b.setAttribute("aria-pressed", String(!on));
      state.types = on ? state.types.filter(x => x !== t) : state.types.concat(t);
    });
    if (!m.has_key) {
      $("#out").innerHTML = `<div class="error">GROQ_API_KEY is not set — retrieval works, generation will not.</div>`;
    }
  }).catch(() => {
    if (attempt < 4) return setTimeout(() => boot(attempt + 1), 1500);
    $("#corpus-stat").textContent = "index unavailable";
    $("#out").innerHTML = `<div class="error">Could not reach the index. If the server just started, give it a moment and reload.</div>`;
  });
}
boot();

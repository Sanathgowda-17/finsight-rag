/**
 * app.js — FinSight Multimodal Financial & Legal Hybrid RAG UI Controller
 */

let presets = [];

document.addEventListener("DOMContentLoaded", async () => {
  initTabs();
  initSlider();
  await loadPresets();
  await loadCorpus();
  await loadBenchmark();
  await executeQuery();

  document.getElementById("ragQueryForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    await executeQuery();
  });

  document.getElementById("ingestForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    await ingestDocument();
  });
});

function initTabs() {
  const btns = document.querySelectorAll(".nav-btn");
  const panes = document.querySelectorAll(".tab-pane");
  btns.forEach((b) => {
    b.addEventListener("click", () => {
      btns.forEach((x) => x.classList.remove("active"));
      panes.forEach((p) => p.classList.remove("active"));
      b.classList.add("active");
      const pane = document.getElementById(b.dataset.tab);
      if (pane) pane.classList.add("active");
    });
  });
}

function initSlider() {
  const slider = document.getElementById("alphaSlider");
  const label = document.getElementById("alphaLabel");
  slider.addEventListener("input", () => {
    const a = parseFloat(slider.value);
    const densePct = Math.round(a * 100);
    const sparsePct = 100 - densePct;
    label.textContent = `${a.toFixed(2)} (${densePct}% Dense Vector / ${sparsePct}% Sparse BM25)`;
  });
}

async function loadPresets() {
  try {
    const res = await fetch("/api/presets");
    presets = await res.json();
    const bar = document.getElementById("presetBar");
    bar.innerHTML = "";
    presets.forEach((p) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "preset-btn";
      btn.textContent = p.label;
      btn.addEventListener("click", async () => {
        document.getElementById("queryInput").value = p.query;
        document.getElementById("alphaSlider").value = p.alpha;
        document.getElementById("alphaSlider").dispatchEvent(new Event("input"));
        await executeQuery();
      });
      bar.appendChild(btn);
    });
  } catch (err) {
    console.error("Failed loading presets:", err);
  }
}

async function executeQuery() {
  const btn = document.getElementById("runRagBtn");
  btn.textContent = "⏳ Fusing BM25 + Dense RRF...";
  btn.disabled = true;

  try {
    const payload = {
      query: document.getElementById("queryInput").value,
      alpha_dense_weight: parseFloat(document.getElementById("alphaSlider").value),
      top_k: 4
    };
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    renderRagResponse(data);
    await loadHistory();
  } catch (err) {
    console.error("RAG execution error:", err);
  } finally {
    btn.textContent = "⚡ Execute Hybrid RRF Retrieval";
    btn.disabled = false;
  }
}

function renderRagResponse(data) {
  const ragas = data.ragas_metrics || {};
  const chunks = data.retrieved_chunks || [];
  const top = chunks[0] || {};

  // 1. Update KPIs
  document.getElementById("kpiFaithfulness").textContent = `${(ragas.faithfulness * 100).toFixed(1)}%`;
  document.getElementById("kpiHallucination").textContent = ragas.hallucination_risk || "100% Grounded";
  document.getElementById("kpiPrecision").textContent = `${(ragas.context_precision * 100).toFixed(1)}%`;
  document.getElementById("kpiRelevance").textContent = `${(ragas.answer_relevance * 100).toFixed(1)}%`;
  document.getElementById("kpiTopDoc").textContent = top.doc_id || "—";
  document.getElementById("kpiTopCE").textContent = `Rerank Score: ${top.cross_encoder_score ?? "—"}`;
  document.getElementById("kpiLatency").textContent = `${data.latency_ms} ms`;

  // 2. Render Answer with Citation Pills
  let htmlAns = escapeHtml(data.answer_markdown || "")
    .replace(/^### (.*)$/gm, "<strong style='font-size:0.95rem;color:#67e8f9;'>$1</strong>")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`(.*?)`/g, "<code class='mono'>$1</code>")
    .replace(/\[([A-Z0-9\-]+)\]/g, "<span class='cite-pill'>Cite: $1</span>")
    .replace(/\n/g, "<br>");
  document.getElementById("answerBox").innerHTML = htmlAns;

  // 3. Render Verified Claims
  const claimsList = document.getElementById("claimsList");
  claimsList.innerHTML = (data.verified_claims || []).map((c) => `
    <div class="chunk-card green-left">
      <div class="chunk-top">
        <span class="chunk-title">✅ Citation Verified: <span class="mono">${escapeHtml(c.citation)}</span></span>
        <span class="badge badge-green">${escapeHtml(c.status)} (${(c.confidence * 100).toFixed(1)}%)</span>
      </div>
      <div class="chunk-body">${escapeHtml(c.claim)}</div>
      <div class="score-pills">
        <span class="badge badge-cyan">${escapeHtml(c.section)}</span>
      </div>
    </div>
  `).join("");

  // 4. Render Top-K Chunks
  const chunksList = document.getElementById("chunksList");
  chunksList.innerHTML = chunks.map((ch, idx) => `
    <div class="chunk-card">
      <div class="chunk-top">
        <span class="chunk-title">#${idx + 1} [${escapeHtml(ch.doc_id)}] ${escapeHtml(ch.title)}</span>
        <span class="badge badge-green">Rerank: ${ch.cross_encoder_score}</span>
      </div>
      <div class="chunk-body">${escapeHtml(ch.content.split("\n")[0])}</div>
      <div class="score-pills">
        <span class="badge badge-amber">BM25: ${ch.bm25_score} (Rank #${ch.bm25_rank})</span>
        <span class="badge badge-blue">Dense Cosine: ${ch.dense_score} (Rank #${ch.dense_rank})</span>
        <span class="badge badge-cyan">RRF: ${ch.rrf_score}</span>
      </div>
    </div>
  `).join("");

  // 5. Render RRF Mathematical Table (Tab 2)
  const rrfBody = document.getElementById("rrfMatrixBody");
  rrfBody.innerHTML = chunks.map((ch) => `
    <tr>
      <td class="mono"><strong>${escapeHtml(ch.doc_id)}</strong></td>
      <td><span class="badge badge-blue">${escapeHtml(ch.category)}</span></td>
      <td class="mono">${ch.bm25_score} <span style="color:#94a3b8;">(#${ch.bm25_rank})</span></td>
      <td class="mono">${ch.dense_score} <span style="color:#94a3b8;">(#${ch.dense_rank})</span></td>
      <td class="mono"><strong>${ch.rrf_score}</strong></td>
      <td><span class="badge badge-green">${ch.cross_encoder_score}</span></td>
      <td class="mono">${escapeHtml((ch.key_figures || []).join(", ") || "—")}</td>
    </tr>
  `).join("");

  // 6. Render Extracted Financial Table (Tab 2)
  const tableChunk = chunks.find((c) => c.extracted_table) || top;
  const tblContainer = document.getElementById("extractedTableContainer");
  if (tableChunk && tableChunk.extracted_table) {
    const t = tableChunk.extracted_table;
    const ths = t.headers.map((h) => `<th>${escapeHtml(h)}</th>`).join("");
    const trs = t.rows.map((r) => `<tr>${r.map((col) => `<td class="mono">${escapeHtml(col)}</td>`).join("")}</tr>`).join("");
    tblContainer.innerHTML = `
      <div style="margin-bottom:8px;font-size:0.8rem;color:#67e8f9;">
        Extracted from <strong>${escapeHtml(tableChunk.doc_id)}</strong> (${escapeHtml(tableChunk.section)})
      </div>
      <table class="data-table">
        <thead><tr>${ths}</tr></thead>
        <tbody>${trs}</tbody>
      </table>
    `;
  } else {
    tblContainer.innerHTML = `<div style="color:#94a3b8;font-size:0.8rem;">No Markdown schedule table embedded in the primary chunk. Select NVIDIA 10-K or JPMorgan Basel III preset to inspect parsed GAAP/Capital tables.</div>`;
  }
}

async function loadCorpus() {
  try {
    const res = await fetch("/api/corpus");
    const docs = await res.json();
    document.getElementById("navCorpusCount").textContent = docs.length;
    document.getElementById("corpusTotalSpan").textContent = docs.length;

    const box = document.getElementById("corpusCardsBox");
    box.innerHTML = docs.map((d) => `
      <div class="chunk-card">
        <div class="chunk-top">
          <span class="chunk-title mono">[${escapeHtml(d.doc_id)}] ${escapeHtml(d.title)}</span>
          <span class="badge badge-cyan">${escapeHtml(d.category)}</span>
        </div>
        <div class="chunk-body">${escapeHtml(d.content.slice(0, 220))}...</div>
        <div class="score-pills">
          <span class="badge badge-blue">${escapeHtml(d.section)}</span>
        </div>
      </div>
    `).join("");
  } catch (err) {
    console.error("Failed loading corpus:", err);
  }
}

async function ingestDocument() {
  try {
    const payload = {
      doc_id: document.getElementById("docIdInput").value,
      category: document.getElementById("docCategoryInput").value,
      title: document.getElementById("docTitleInput").value,
      section: document.getElementById("docSectionInput").value,
      content: document.getElementById("docContentInput").value
    };
    const res = await fetch("/api/corpus", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      document.getElementById("ingestForm").reset();
      await loadCorpus();
    }
  } catch (err) {
    console.error("Ingest failed:", err);
  }
}

async function loadBenchmark() {
  try {
    const res = await fetch("/api/benchmark", { method: "POST" });
    const data = await res.json();
    const tbody = document.getElementById("benchmarkBody");
    tbody.innerHTML = (data.architectures || []).map((a, idx) => {
      const badge = idx === 2 ? "badge-green" : idx === 1 ? "badge-amber" : "badge-blue";
      return `
        <tr>
          <td><strong>${escapeHtml(a.name)}</strong></td>
          <td class="mono"><span class="badge ${badge}">${escapeHtml(a.recall_at_3)}</span></td>
          <td class="mono">${a.mrr}</td>
          <td class="mono">${escapeHtml(a.exact_clause_match)}</td>
          <td class="mono"><strong>${(a.faithfulness * 100).toFixed(1)}%</strong></td>
          <td style="color:#94a3b8;">${escapeHtml(a.notes)}</td>
        </tr>
      `;
    }).join("");
  } catch (err) {
    console.error("Benchmark error:", err);
  }
}

async function loadHistory() {
  try {
    const res = await fetch("/api/history");
    const rows = await res.json();
    const tbody = document.getElementById("historyBody");
    tbody.innerHTML = rows.map((r) => `
      <tr>
        <td class="mono">${escapeHtml(r.query_id)}</td>
        <td>${escapeHtml(r.query_text.slice(0, 68))}...</td>
        <td class="mono"><span class="badge badge-cyan">${escapeHtml(r.top_doc_id)}</span></td>
        <td class="mono">${Number(r.alpha_hybrid).toFixed(2)}</td>
        <td class="mono"><span class="badge badge-green">${(r.faithfulness * 100).toFixed(1)}%</span></td>
        <td class="mono">${(r.context_precision * 100).toFixed(1)}%</td>
        <td class="mono">${(r.answer_relevance * 100).toFixed(1)}%</td>
        <td class="mono">${r.latency_ms} ms</td>
      </tr>
    `).join("");
  } catch (err) {
    console.error("History error:", err);
  }
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

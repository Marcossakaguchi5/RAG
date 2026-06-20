const api = {
  async request(path, options = {}) {
    const response = await fetch(path, options);
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = Array.isArray(body?.detail)
        ? body.detail.map((error) => `${error.filename}: ${error.detail}`).join(" · ")
        : body?.detail || "Não foi possível concluir a operação.";
      throw new Error(detail);
    }
    return body;
  },
  documents() { return this.request("/api/documents"); },
  points() { return this.request("/api/points?limit=12"); },
  search(payload) {
    return this.request("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },
  upload(data) { return this.request("/api/documents/upload", { method: "POST", body: data }); },
};

const byId = (id) => document.getElementById(id);
const escapeHtml = (value = "") => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");
const shortId = (id) => id.length > 13 ? `${id.slice(0, 8)}…${id.slice(-4)}` : id;
const bytes = (value) => value < 1024 * 1024
  ? `${Math.max(1, Math.round(value / 1024))} KB`
  : `${(value / (1024 * 1024)).toFixed(1)} MB`;

function setNotice(id, message = "", isError = false) {
  const element = byId(id);
  element.hidden = !message;
  element.textContent = message;
  element.classList.toggle("error", isError);
}

function renderDocuments(documents) {
  const container = byId("documents-list");
  if (!documents.length) {
    container.className = "document-list empty-state";
    container.textContent = "Nenhum PDF ingerido ainda.";
    return;
  }
  container.className = "document-list";
  container.innerHTML = documents.map((document) => `
    <article class="document-row">
      <div>
        <div class="document-name" title="${escapeHtml(document.original_name)}">${escapeHtml(document.original_name)}</div>
        <span class="document-meta">${document.page_count} pág. · ${bytes(document.size_bytes)}</span>
      </div>
      <span class="document-tag">${document.chunks_count} chunks</span>
    </article>`).join("");
}

function renderPoints(data) {
  const container = byId("points-list");
  byId("points-total").textContent = `(${data.total})`;
  if (!data.points.length) {
    container.className = "points-list empty-state";
    container.textContent = "Os primeiros pontos aparecerão aqui após a ingestão.";
    return;
  }
  container.className = "points-list";
  container.innerHTML = data.points.map((point) => `
    <article class="point">
      <div class="point-top"><span>${escapeHtml(point.document_name)} · p. ${point.page_number}</span><code title="${point.id}">${shortId(point.id)}</code></div>
      <p>${escapeHtml(point.content)}</p>
    </article>`).join("");
}

async function loadDocuments() {
  const container = byId("documents-list");
  container.className = "document-list loading";
  container.textContent = "Carregando documentos…";
  try {
    renderDocuments(await api.documents());
  } catch (error) {
    container.className = "document-list empty-state";
    container.textContent = error.message;
  }
}

async function loadPoints() {
  const container = byId("points-list");
  container.className = "points-list loading";
  container.textContent = "Carregando pontos…";
  try {
    renderPoints(await api.points());
  } catch (error) {
    container.className = "points-list empty-state";
    container.textContent = error.message;
  }
}

function renderMetrics(metrics, topK) {
  const values = [
    [`Precision@${topK}`, metrics.precision_at_k],
    [`Recall@${topK}`, metrics.recall_at_k],
    ["MAP", metrics.map],
    [`NDCG@${topK}`, metrics.ndcg_at_k],
    ["MRR", metrics.mrr],
  ];
  byId("metrics").innerHTML = values.map(([name, value]) => `
    <div class="metric"><span>${name}</span><strong>${value === null || value === undefined ? "—" : value.toFixed(3)}</strong></div>`).join("");
  byId("metrics-note").textContent = metrics.evaluated
    ? "Cálculo binário com os IDs relevantes informados nesta consulta."
    : metrics.message;
}

function renderResults(data) {
  const container = byId("search-results");
  byId("result-method").textContent = data.method === "dense"
    ? "DENSE · MiniLM"
    : data.method === "bm25" ? "SPARSE · BM25" : "HÍBRIDO · RRF";
  renderMetrics(data.metrics, data.top_k);
  if (!data.results.length) {
    container.className = "search-results empty-state";
    container.textContent = "Nenhum chunk correspondeu à consulta.";
    return;
  }
  container.className = "search-results";
  container.innerHTML = data.results.map((result, index) => {
    const details = [
      result.dense_score !== null ? `dense ${result.dense_score.toFixed(3)}` : null,
      result.bm25_score !== null ? `sparse bm25 ${result.bm25_score.toFixed(3)}` : null,
    ].filter(Boolean).join(" · ");
    return `
      <article class="result">
        <div class="result-top">
          <div><span class="result-source">#${index + 1} · ${escapeHtml(result.document_name)} · página ${result.page_number}</span><code class="result-id" title="Use este ID como relevância">${result.chunk_id}</code></div>
          <span class="score">${result.score.toFixed(4)}</span>
        </div>
        <p class="result-content">${escapeHtml(result.content)}</p>
        ${details ? `<div class="score-detail">${details}</div>` : ""}
      </article>`;
  }).join("");
}

function bindTabs() {
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    const selected = tab.dataset.tab;
    document.querySelectorAll(".tab").forEach((button) => {
      const active = button === tab;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
    });
    ["upload", "search"].forEach((name) => {
      const panel = byId(`${name}-panel`);
      const active = name === selected;
      panel.classList.toggle("active", active);
      panel.hidden = !active;
    });
  }));
}

function bindUpload() {
  const input = byId("pdf-files");
  const button = byId("upload-button");
  const summary = byId("file-summary");
  const zone = document.querySelector(".dropzone");
  const updateFiles = () => {
    const count = input.files.length;
    summary.textContent = count ? `${count} arquivo${count > 1 ? "s" : ""} selecionado${count > 1 ? "s" : ""}` : "Nenhum arquivo selecionado";
    button.disabled = !count;
  };
  input.addEventListener("change", updateFiles);
  ["dragenter", "dragover"].forEach((event) => zone.addEventListener(event, (item) => { item.preventDefault(); zone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((event) => zone.addEventListener(event, (item) => { item.preventDefault(); zone.classList.remove("dragging"); }));
  zone.addEventListener("drop", (event) => {
    const files = [...event.dataTransfer.files].filter((file) => file.name.toLowerCase().endsWith(".pdf"));
    if (!files.length) return;
    const transfer = new DataTransfer();
    files.forEach((file) => transfer.items.add(file));
    input.files = transfer.files;
    updateFiles();
  });
  byId("upload-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = new FormData();
    [...input.files].forEach((file) => data.append("files", file));
    button.disabled = true;
    button.textContent = "Indexando…";
    setNotice("upload-message");
    try {
      const response = await api.upload(data);
      const errors = response.errors.map((error) => `${error.filename}: ${error.detail}`);
      setNotice("upload-message", `${response.documents.length} documento(s) ingerido(s).${errors.length ? ` ${errors.join(" · ")}` : ""}`, Boolean(errors.length));
      input.value = "";
      updateFiles();
      await Promise.all([loadDocuments(), loadPoints()]);
    } catch (error) {
      setNotice("upload-message", error.message, true);
    } finally {
      button.textContent = "Ingerir arquivos";
      updateFiles();
    }
  });
}

function bindSearch() {
  byId("search-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const submit = event.submitter;
    submit.disabled = true;
    submit.textContent = "Buscando…";
    setNotice("search-message");
    try {
      const data = await api.search({
        query: byId("query").value.trim(),
        method: byId("method").value,
        top_k: Number(byId("top-k").value),
        relevant_chunk_ids: byId("relevant-ids").value.split(",").map((id) => id.trim()).filter(Boolean),
      });
      renderResults(data);
    } catch (error) {
      setNotice("search-message", error.message, true);
    } finally {
      submit.disabled = false;
      submit.textContent = "Buscar";
    }
  });
}

bindTabs();
bindUpload();
bindSearch();
byId("refresh-documents").addEventListener("click", loadDocuments);
byId("refresh-points").addEventListener("click", loadPoints);
renderMetrics({ evaluated: false, message: "Informe chunks relevantes para habilitar a avaliação supervisionada." }, 5);
Promise.all([loadDocuments(), loadPoints()]);

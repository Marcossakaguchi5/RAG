const byId = (id) => document.getElementById(id);
const REQUEST_TIMEOUT_MS = 30000;

const state = {
  collectionStorageKey: "rag-ingest-selected-collection",
  collections: [],
  lastSearchPayload: null,
  lastSearchResult: null,
  relevantIds: new Set(),
  get collectionName() {
    return sessionStorage.getItem(this.collectionStorageKey) || this.collections[0]?.name || "rag_chunks";
  },
  set collectionName(value) {
    sessionStorage.setItem(this.collectionStorageKey, value);
  },
};

const auth = {
  storageKey: "rag-ingest-access-token",
  get token() { return sessionStorage.getItem(this.storageKey); },
  set token(value) { sessionStorage.setItem(this.storageKey, value); },
  clear() { sessionStorage.removeItem(this.storageKey); },
  unlock() {
    document.body.classList.add("authenticated");
    byId("auth-gate").hidden = true;
    byId("app-shell").hidden = false;
    byId("auth-password").value = "";
    setNotice("auth-message");
  },
  lock(message = "") {
    this.clear();
    document.body.classList.remove("authenticated");
    byId("app-shell").hidden = true;
    byId("auth-gate").hidden = false;
    setNotice("auth-message", message, Boolean(message));
    byId("auth-password").focus();
  },
};

const api = {
  async request(path, options = {}) {
    const { timeoutMs = REQUEST_TIMEOUT_MS, ...fetchOptions } = options;
    const headers = new Headers(fetchOptions.headers || {});
    const accessToken = auth.token;
    if (accessToken && path !== "/api/auth/login") {
      headers.set("Authorization", `Bearer ${accessToken}`);
    }
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
    let response;
    try {
      response = await fetch(path, { ...fetchOptions, headers, signal: controller.signal });
    } catch (error) {
      if (error.name === "AbortError") {
        throw new Error("A API demorou para responder. Tente novamente em alguns segundos.");
      }
      throw new Error("Não foi possível conectar com a API.");
    } finally {
      window.clearTimeout(timeoutId);
    }
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      if (response.status === 401 && path !== "/api/auth/login" && auth.token === accessToken) {
        auth.lock("Sua sessão expirou. Informe a senha novamente.");
      }
      const detail = Array.isArray(body?.detail)
        ? body.detail.map((error) => `${error.filename}: ${error.detail}`).join(" · ")
        : body?.detail || "Não foi possível concluir a operação.";
      throw new Error(detail);
    }
    return body;
  },
  login(password) {
    return this.request("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });
  },
  session() { return this.request("/api/auth/session"); },
  collections() { return this.request("/api/collections"); },
  chunkingStrategies() { return this.request("/api/chunking-strategies"); },
  createCollection(name) {
    return this.request("/api/collections", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
  },
  documents(collectionName) {
    return this.request(`/api/documents?collection_name=${encodeURIComponent(collectionName)}`);
  },
  points(collectionName) {
    return this.request(`/api/points?limit=12&collection_name=${encodeURIComponent(collectionName)}`);
  },
  search(payload) {
    return this.request("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },
  upload(data) { return this.request("/api/documents/upload", { method: "POST", timeoutMs: 900000, body: data }); },
};

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
const relevantIdsFromInput = () => byId("relevant-ids").value
  .split(",")
  .map((id) => id.trim())
  .filter(Boolean);
const fileTimestamp = () => new Date().toISOString().replaceAll(":", "-").replace(/\..+/, "");

function downloadFile(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = typeof value === "object" ? JSON.stringify(value) : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(rows, columns) {
  return [
    columns.join(","),
    ...rows.map((row) => columns.map((column) => csvCell(row[column])).join(",")),
  ].join("\n");
}

function setNotice(id, message = "", isError = false) {
  const element = byId(id);
  element.hidden = !message;
  element.textContent = message;
  element.classList.toggle("error", isError);
}

function updateRelevanceControls({ writeInput = true } = {}) {
  const ids = [...state.relevantIds];
  if (writeInput) {
    byId("relevant-ids").value = ids.join(", ");
  }
  byId("relevant-count").textContent = `${ids.length} marcado${ids.length === 1 ? "" : "s"}`;
  byId("clear-relevance").disabled = !ids.length;
  byId("recompute-metrics").disabled = !state.lastSearchPayload;
  document.querySelectorAll("#search-results input[data-chunk-id]").forEach((input) => {
    input.checked = state.relevantIds.has(input.dataset.chunkId);
  });
}

function updateExportAction() {
  const enabled = Boolean(state.lastSearchPayload && state.lastSearchResult);
  byId("export-search-json").disabled = !enabled;
  byId("export-search-csv").disabled = !enabled;
}

function syncRelevantIdsFromInput({ writeInput = false } = {}) {
  state.relevantIds = new Set(relevantIdsFromInput());
  updateRelevanceControls({ writeInput });
}

function resetSearchState() {
  state.lastSearchPayload = null;
  state.lastSearchResult = null;
  state.relevantIds = new Set();
  byId("search-results").className = "search-results empty-state";
  byId("search-results").textContent = "Faça uma pergunta para ver os chunks recuperados.";
  byId("result-method").textContent = "Aguardando busca";
  renderMetrics({ evaluated: false, message: "Informe chunks relevantes para habilitar a avaliação supervisionada." }, 5);
  updateRelevanceControls();
  updateExportAction();
}

function renderCollections(collections, preferredName = state.collectionName) {
  const select = byId("collection-select");
  state.collections = collections;
  if (!collections.length) {
    select.innerHTML = "";
    select.disabled = true;
    return;
  }
  const selected = collections.some((collection) => collection.name === preferredName)
    ? preferredName
    : collections[0].name;
  state.collectionName = selected;
  select.disabled = false;
  select.innerHTML = collections.map((collection) => `
    <option value="${escapeHtml(collection.name)}">
      ${escapeHtml(collection.name)} · ${collection.documents_count} doc${collection.documents_count === 1 ? "" : "s"}
    </option>
  `).join("");
  select.value = selected;
}

function renderDocuments(documents) {
  const container = byId("documents-list");
  if (!documents.length) {
    container.className = "document-list empty-state";
    container.textContent = `Nenhum PDF ingerido em ${state.collectionName}.`;
    return;
  }
  container.className = "document-list";
  container.innerHTML = documents.map((document) => `
    <article class="document-row">
      <div>
        <div class="document-name" title="${escapeHtml(document.original_name)}">${escapeHtml(document.original_name)}</div>
        <span class="document-meta">${document.page_count} pág. · ${bytes(document.size_bytes)} · ${escapeHtml(document.chunking_strategy || "recursive_text")}</span>
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
      <div class="point-top">
        <span>${escapeHtml(point.file_name || point.document_name)} · p. ${point.page_number} · chunk ${point.chunk_index || point.ordinal + 1}${point.chunk_total ? `/${point.chunk_total}` : ""}</span>
        <code title="${point.id}">${shortId(point.id)}</code>
      </div>
      <div class="point-meta">${point.word_count || 0} palavras · ${point.char_count || point.content.length} caracteres · ${escapeHtml(point.chunking_strategy || "recursive_text")} · doc ${shortId(point.document_id)}</div>
      <p>${escapeHtml(point.content)}</p>
    </article>`).join("");
}

async function loadDocuments() {
  const container = byId("documents-list");
  container.className = "document-list loading";
  container.textContent = "Carregando documentos…";
  try {
    renderDocuments(await api.documents(state.collectionName));
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
    renderPoints(await api.points(state.collectionName));
  } catch (error) {
    container.className = "points-list empty-state";
    container.textContent = error.message;
  }
}

async function loadCollections(preferredName = state.collectionName) {
  const select = byId("collection-select");
  select.disabled = true;
  try {
    renderCollections(await api.collections(), preferredName);
    setNotice("collection-message");
  } catch (error) {
    select.innerHTML = "";
    select.disabled = true;
    setNotice("collection-message", error.message, true);
  }
}

async function loadChunkingStrategies() {
  const select = byId("chunking-strategy");
  try {
    const data = await api.chunkingStrategies();
    select.innerHTML = data.strategies.map((strategy) => `
      <option value="${escapeHtml(strategy.value)}">${escapeHtml(strategy.label)}</option>
    `).join("");
    select.value = data.default;
  } catch {
    select.innerHTML = '<option value="recursive_text">Baseline 2: recursive_text</option>';
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

function buildSearchPayload() {
  syncRelevantIdsFromInput({ writeInput: true });
  return {
    query: byId("query").value.trim(),
    collection_name: state.collectionName,
    method: byId("method").value,
    top_k: Number(byId("top-k").value),
    relevant_chunk_ids: [...state.relevantIds],
  };
}

function buildSearchExport() {
  if (!state.lastSearchPayload || !state.lastSearchResult) return null;
  return {
    schema_version: "1.0",
    exported_at: new Date().toISOString(),
    app: "ingest",
    export_type: "retrieval_search",
    question: state.lastSearchPayload.query,
    answer: "",
    answer_available: false,
    collection_name: state.lastSearchPayload.collection_name,
    query: state.lastSearchPayload.query,
    request: state.lastSearchPayload,
    method: state.lastSearchResult.method,
    top_k: state.lastSearchResult.top_k,
    result_count: state.lastSearchResult.results.length,
    relevant_count: state.lastSearchPayload.relevant_chunk_ids.length,
    relevant_chunk_ids: state.lastSearchPayload.relevant_chunk_ids,
    metrics: state.lastSearchResult.metrics,
    results: state.lastSearchResult.results,
  };
}

function searchExportRows(exportData) {
  const base = {
    app: exportData.app,
    export_type: exportData.export_type,
    schema_version: exportData.schema_version,
    exported_at: exportData.exported_at,
    collection_name: exportData.collection_name,
    question: exportData.question,
    answer: exportData.answer,
    answer_available: exportData.answer_available,
    query: exportData.query,
    method: exportData.method,
    top_k: exportData.top_k,
    result_count: exportData.result_count,
    relevant_count: exportData.relevant_count,
    relevant_chunk_ids: exportData.relevant_chunk_ids,
    evaluated: exportData.metrics?.evaluated,
    metrics_message: exportData.metrics?.message,
    precision_at_k: exportData.metrics?.precision_at_k,
    recall_at_k: exportData.metrics?.recall_at_k,
    map: exportData.metrics?.map,
    ndcg_at_k: exportData.metrics?.ndcg_at_k,
    mrr: exportData.metrics?.mrr,
    request_json: exportData.request,
    metrics_json: exportData.metrics,
  };
  const results = exportData.results.length ? exportData.results : [{}];
  return results.map((result, index) => ({
    ...base,
    rank: result.chunk_id ? index + 1 : "",
    chunk_id: result.chunk_id,
    document_id: result.document_id,
    document_name: result.document_name,
    page_number: result.page_number,
    ordinal: result.ordinal,
    score: result.score,
    dense_score: result.dense_score,
    bm25_score: result.bm25_score,
    is_relevant: result.chunk_id ? exportData.relevant_chunk_ids.includes(result.chunk_id) : "",
    content: result.content,
    result_json: result.chunk_id ? result : "",
  }));
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
    updateExportAction();
    return;
  }
  container.className = "search-results";
  container.innerHTML = data.results.map((result, index) => {
    const details = [
      result.dense_score !== null ? `dense ${result.dense_score.toFixed(3)}` : null,
      result.bm25_score !== null ? `sparse bm25 ${result.bm25_score.toFixed(3)}` : null,
    ].filter(Boolean).join(" · ");
    const checked = state.relevantIds.has(result.chunk_id) ? "checked" : "";
    return `
      <article class="result">
        <div class="result-top">
          <div><span class="result-source">#${index + 1} · ${escapeHtml(result.document_name)} · página ${result.page_number}</span><code class="result-id" title="Use este ID como relevância">${result.chunk_id}</code></div>
          <div class="result-actions">
            <label class="relevance-check" title="Marcar como chunk relevante">
              <input type="checkbox" data-chunk-id="${escapeHtml(result.chunk_id)}" ${checked} />
              <span>Relevante</span>
            </label>
            <span class="score">${result.score.toFixed(4)}</span>
          </div>
        </div>
        <p class="result-content">${escapeHtml(result.content)}</p>
        ${details ? `<div class="score-detail">${details}</div>` : ""}
      </article>`;
  }).join("");
  updateRelevanceControls();
  updateExportAction();
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
  const form = byId("upload-form");
  const input = byId("pdf-files");
  const button = byId("upload-button");
  const summary = byId("file-summary");
  const zone = document.querySelector(".dropzone");
  const strategy = byId("chunking-strategy");
  const progress = byId("upload-progress");
  let isIndexing = false;
  const updateFiles = () => {
    const count = input.files.length;
    summary.textContent = isIndexing
      ? "Indexação em andamento"
      : count ? `${count} arquivo${count > 1 ? "s" : ""} selecionado${count > 1 ? "s" : ""}` : "Nenhum arquivo selecionado";
    button.disabled = isIndexing || !count;
  };
  const setIndexing = (value) => {
    isIndexing = value;
    form.classList.toggle("indexing", value);
    input.disabled = value;
    strategy.disabled = value;
    progress.hidden = !value;
    button.textContent = value ? "Indexando…" : "Ingerir arquivos";
    updateFiles();
  };
  input.addEventListener("change", updateFiles);
  ["dragenter", "dragover"].forEach((event) => zone.addEventListener(event, (item) => { item.preventDefault(); zone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((event) => zone.addEventListener(event, (item) => { item.preventDefault(); zone.classList.remove("dragging"); }));
  zone.addEventListener("drop", (event) => {
    if (isIndexing) return;
    const files = [...event.dataTransfer.files].filter((file) => file.name.toLowerCase().endsWith(".pdf"));
    if (!files.length) return;
    const transfer = new DataTransfer();
    files.forEach((file) => transfer.items.add(file));
    input.files = transfer.files;
    updateFiles();
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (isIndexing) return;
    const data = new FormData();
    data.append("collection_name", state.collectionName);
    data.append("chunking_strategy", strategy.value);
    [...input.files].forEach((file) => data.append("files", file));
    setIndexing(true);
    setNotice("upload-message");
    try {
      const response = await api.upload(data);
      const errors = response.errors.map((error) => `${error.filename}: ${error.detail}`);
      setNotice("upload-message", `${response.documents.length} documento(s) ingerido(s).${errors.length ? ` ${errors.join(" · ")}` : ""}`, Boolean(errors.length));
      input.value = "";
      updateFiles();
      await loadCollections(state.collectionName);
      await Promise.all([loadDocuments(), loadPoints()]);
    } catch (error) {
      setNotice("upload-message", error.message, true);
    } finally {
      setIndexing(false);
    }
  });
}

function bindCollections() {
  byId("collection-select").addEventListener("change", async (event) => {
    state.collectionName = event.target.value;
    setNotice("collection-message");
    resetSearchState();
    await Promise.all([loadDocuments(), loadPoints()]);
  });

  byId("collection-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = byId("collection-name");
    const submit = byId("collection-submit");
    const name = input.value.trim();
    if (!name) return;
    submit.disabled = true;
    submit.textContent = "Criando…";
    setNotice("collection-message");
    try {
      const collection = await api.createCollection(name);
      input.value = "";
      await loadCollections(collection.name);
      await Promise.all([loadDocuments(), loadPoints()]);
      setNotice("collection-message", `Collection ${collection.name} pronta para receber arquivos.`);
    } catch (error) {
      setNotice("collection-message", error.message, true);
    } finally {
      submit.disabled = false;
      submit.textContent = "Adicionar";
    }
  });
}

function bindSearch() {
  async function runSearch(button) {
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = button.id === "recompute-metrics" ? "Calculando…" : "Buscando…";
    setNotice("search-message");
    try {
      const payload = buildSearchPayload();
      const data = await api.search(payload);
      state.lastSearchPayload = payload;
      state.lastSearchResult = data;
      renderResults(data);
    } catch (error) {
      setNotice("search-message", error.message, true);
    } finally {
      button.disabled = false;
      button.textContent = originalText;
      updateRelevanceControls();
    }
  }

  byId("search-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    await runSearch(event.submitter || byId("search-form").querySelector(".button.primary"));
  });

  byId("relevant-ids").addEventListener("input", () => syncRelevantIdsFromInput());

  byId("clear-relevance").addEventListener("click", () => {
    state.relevantIds = new Set();
    updateRelevanceControls();
  });

  byId("recompute-metrics").addEventListener("click", async (event) => {
    await runSearch(event.currentTarget);
  });

  byId("search-results").addEventListener("change", (event) => {
    const input = event.target;
    if (!(input instanceof HTMLInputElement) || !input.dataset.chunkId) {
      return;
    }
    if (input.checked) {
      state.relevantIds.add(input.dataset.chunkId);
    } else {
      state.relevantIds.delete(input.dataset.chunkId);
    }
    updateRelevanceControls();
  });
}

function bindExport() {
  byId("export-search-json").addEventListener("click", () => {
    const exportData = buildSearchExport();
    if (!exportData) return;
    downloadFile(
      `ingest-retrieval-${fileTimestamp()}.json`,
      JSON.stringify(exportData, null, 2),
      "application/json;charset=utf-8",
    );
  });

  byId("export-search-csv").addEventListener("click", () => {
    const exportData = buildSearchExport();
    if (!exportData) return;
    const columns = [
      "app",
      "export_type",
      "schema_version",
      "exported_at",
      "collection_name",
      "question",
      "answer",
      "answer_available",
      "query",
      "method",
      "top_k",
      "result_count",
      "relevant_count",
      "rank",
      "chunk_id",
      "is_relevant",
      "document_id",
      "document_name",
      "page_number",
      "ordinal",
      "score",
      "dense_score",
      "bm25_score",
      "evaluated",
      "metrics_message",
      "precision_at_k",
      "recall_at_k",
      "map",
      "ndcg_at_k",
      "mrr",
      "relevant_chunk_ids",
      "content",
      "request_json",
      "metrics_json",
      "result_json",
    ];
    downloadFile(
      `ingest-retrieval-${fileTimestamp()}.csv`,
      `${toCsv(searchExportRows(exportData), columns)}\n`,
      "text/csv;charset=utf-8",
    );
  });
}

async function loadDashboard() {
  renderMetrics({ evaluated: false, message: "Informe chunks relevantes para habilitar a avaliação supervisionada." }, 5);
  await Promise.all([loadCollections(), loadChunkingStrategies()]);
  await Promise.all([loadDocuments(), loadPoints()]);
}

function bindAuthentication() {
  byId("auth-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const submit = byId("auth-submit");
    submit.disabled = true;
    submit.textContent = "Validando…";
    setNotice("auth-message");
    try {
      const response = await api.login(byId("auth-password").value);
      auth.token = response.access_token;
      auth.unlock();
      loadDashboard();
    } catch (error) {
      setNotice("auth-message", error.message, true);
    } finally {
      submit.disabled = false;
      submit.textContent = "Acessar painel";
    }
  });
  byId("logout-button").addEventListener("click", () => auth.lock());
}

async function restoreSession() {
  const accessToken = auth.token;
  if (!accessToken) {
    auth.lock();
    return;
  }
  try {
    await api.session();
    if (auth.token !== accessToken) return;
    auth.unlock();
    loadDashboard();
  } catch {
    if (auth.token === accessToken) auth.lock();
  }
}

bindTabs();
bindCollections();
bindUpload();
bindSearch();
bindExport();
bindAuthentication();
byId("refresh-documents").addEventListener("click", loadDocuments);
byId("refresh-points").addEventListener("click", loadPoints);
restoreSession();

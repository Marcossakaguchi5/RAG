const byId = (id) => document.getElementById(id);
const REQUEST_TIMEOUT_MS = 30000;
const RAGAS_TIMEOUT_MS = 600000;

const state = {
  collectionStorageKey: "rag-chat-selected-collection",
  collections: [],
  lastRagPayload: null,
  lastRagResult: null,
  get collectionName() {
    return sessionStorage.getItem(this.collectionStorageKey) || this.collections[0]?.name || "rag_chunks";
  },
  set collectionName(value) {
    sessionStorage.setItem(this.collectionStorageKey, value);
  },
};

const auth = {
  storageKey: "rag-chat-access-token",
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
      throw new Error("Nao foi possivel conectar com a API.");
    } finally {
      window.clearTimeout(timeoutId);
    }
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      if (response.status === 401 && path !== "/api/auth/login" && auth.token === accessToken) {
        auth.lock("Sua sessao expirou. Informe a senha novamente.");
      }
      const detail = body?.detail || "Nao foi possivel concluir a operacao.";
      throw new Error(String(detail));
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
  rag(payload) {
    return this.request("/api/rag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      timeoutMs: 180000,
      body: JSON.stringify(payload),
    });
  },
  ragas(payload) {
    return this.request("/api/rag/ragas", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      timeoutMs: RAGAS_TIMEOUT_MS,
      body: JSON.stringify(payload),
    });
  },
};

const escapeHtml = (value = "") => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

const shortId = (id) => id.length > 13 ? `${id.slice(0, 8)}...${id.slice(-4)}` : id;
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

function renderMetricShell() {
  const names = ["Faithfulness", "Answer relevancy", "Context precision", "Context recall", "Factual correctness"];
  byId("ragas-metrics").innerHTML = names.map((name) => `
    <div class="metric"><span>${name}</span><strong>--</strong></div>
  `).join("");
}

function renderRagas(report) {
  const metrics = report.metrics || [];
  if (!metrics.length) {
    renderMetricShell();
    byId("ragas-note").textContent = report.message || "Metricas nao calculadas.";
    return;
  }
  byId("ragas-metrics").innerHTML = metrics.map((metric) => `
    <div class="metric" title="${escapeHtml(metric.reason || "")}">
      <span>${escapeHtml(metric.name)}</span>
      <strong>${metric.value === null || metric.value === undefined ? "--" : metric.value.toFixed(2)}</strong>
    </div>
  `).join("");
  byId("ragas-note").textContent = report.evaluated
    ? report.message || "Notas de 0 a 1 calculadas pela biblioteca oficial ragas."
    : report.message || "Metricas nao calculadas.";
}

function updateRagasAction() {
  const button = byId("calculate-ragas");
  button.disabled = !state.lastRagResult?.answer;
  const canExport = Boolean(state.lastRagPayload && state.lastRagResult);
  byId("export-rag-json").disabled = !canExport;
  byId("export-rag-csv").disabled = !canExport;
}

function methodLabel(method, usedReranker) {
  const base = method === "dense" ? "EMBEDDING" : method === "bm25" ? "BM25" : "HIBRIDO";
  return `${base}${usedReranker ? " · RERANK" : ""}`;
}

function metricKey(name = "") {
  return String(name)
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "");
}

function ragasMetricValues(report = {}) {
  const values = {};
  (report.metrics || []).forEach((metric) => {
    values[`ragas_${metricKey(metric.name)}`] = metric.value;
  });
  return values;
}

function buildRagExport() {
  if (!state.lastRagPayload || !state.lastRagResult) return null;
  return {
    schema_version: "1.0",
    exported_at: new Date().toISOString(),
    app: "chat",
    export_type: "rag_answer",
    question: state.lastRagPayload.query,
    answer: state.lastRagResult.answer,
    answer_available: true,
    query: state.lastRagPayload.query,
    reference_answer: byId("reference-answer").value.trim(),
    request: state.lastRagPayload,
    response: state.lastRagResult,
  };
}

function ragExportRows(exportData) {
  const response = exportData.response;
  const metricValues = ragasMetricValues(response.ragas);
  const base = {
    app: exportData.app,
    export_type: exportData.export_type,
    schema_version: exportData.schema_version,
    exported_at: exportData.exported_at,
    collection_name: response.collection_name,
    question: exportData.question,
    answer_available: exportData.answer_available,
    query: exportData.query,
    reference_answer: exportData.reference_answer,
    answer: response.answer,
    method: response.method,
    top_k: response.top_k,
    candidate_k: response.candidate_k,
    used_reranker: response.used_reranker,
    latency_ms: response.latency_ms,
    source_count: response.sources.length,
    ragas_evaluated: response.ragas?.evaluated,
    ragas_message: response.ragas?.message,
    request_json: exportData.request,
    ragas_json: response.ragas,
    response_json: response,
    ...metricValues,
  };
  const sources = response.sources.length ? response.sources : [{}];
  return sources.map((source) => ({
    ...base,
    rank: source.rank,
    retrieval_rank: source.retrieval_rank,
    chunk_id: source.chunk_id,
    document_id: source.document_id,
    document_name: source.document_name,
    page_number: source.page_number,
    ordinal: source.ordinal,
    score: source.score,
    dense_score: source.dense_score,
    bm25_score: source.bm25_score,
    rerank_score: source.rerank_score,
    content: source.content,
    source_json: source.chunk_id ? source : "",
  }));
}

function renderSources(sources) {
  const container = byId("sources");
  byId("sources-count").textContent = sources.length ? `(${sources.length})` : "";
  if (!sources.length) {
    container.className = "sources empty-state";
    container.textContent = "Nenhum chunk foi usado.";
    return;
  }
  container.className = "sources";
  container.innerHTML = sources.map((source) => {
    const details = [
      source.dense_score !== null && source.dense_score !== undefined ? `dense ${source.dense_score.toFixed(3)}` : null,
      source.bm25_score !== null && source.bm25_score !== undefined ? `bm25 ${source.bm25_score.toFixed(3)}` : null,
      source.rerank_score !== null && source.rerank_score !== undefined ? `rerank ${source.rerank_score.toFixed(3)}` : null,
      `rank original ${source.retrieval_rank}`,
    ].filter(Boolean).join(" · ");
    return `
      <article class="source">
        <div class="source-top">
          <div>
            <span class="source-title">[${source.rank}] ${escapeHtml(source.document_name)} · pagina ${source.page_number}</span>
            <code title="${escapeHtml(source.chunk_id)}">${shortId(source.chunk_id)}</code>
          </div>
          <span class="score">${source.score.toFixed(4)}</span>
        </div>
        <p>${escapeHtml(source.content)}</p>
        <div class="score-detail">${details}</div>
      </article>
    `;
  }).join("");
}

function renderResult(data) {
  byId("result-method").textContent = methodLabel(data.method, data.used_reranker);
  byId("run-summary").textContent = `${data.sources.length} fonte(s) · ${data.latency_ms} ms · ${data.collection_name}`;
  byId("answer").className = "answer";
  byId("answer").textContent = data.answer;
  renderSources(data.sources);
  renderRagas(data.ragas);
  state.lastRagResult = data;
  updateRagasAction();
}

function bindCollections() {
  byId("collection-select").addEventListener("change", (event) => {
    state.collectionName = event.target.value;
    setNotice("collection-message");
  });
  byId("refresh-collections").addEventListener("click", () => loadCollections(state.collectionName));
}

function bindRag() {
  byId("rag-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const submit = byId("rag-submit");
    submit.disabled = true;
    submit.textContent = "Consultando...";
    state.lastRagPayload = null;
    state.lastRagResult = null;
    updateRagasAction();
    setNotice("rag-message");
    byId("run-summary").textContent = "Recuperando chunks e gerando resposta...";
    renderRagas({ evaluated: false, message: "RAGAS sera calculado depois, se solicitado." });
    try {
      const payload = {
        query: byId("query").value.trim(),
        collection_name: state.collectionName,
        method: byId("method").value,
        top_k: Number(byId("top-k").value),
        candidate_k: Number(byId("candidate-k").value),
        use_reranker: byId("use-reranker").checked,
        evaluate_ragas: false,
        reference_answer: byId("reference-answer").value.trim(),
      };
      state.lastRagPayload = payload;
      const data = await api.rag(payload);
      renderResult(data);
    } catch (error) {
      setNotice("rag-message", error.message, true);
      byId("run-summary").textContent = "Falha na consulta.";
    } finally {
      submit.disabled = false;
      submit.textContent = "Perguntar";
    }
  });
}

function bindRagas() {
  byId("calculate-ragas").addEventListener("click", async () => {
    if (!state.lastRagPayload || !state.lastRagResult) return;
    const button = byId("calculate-ragas");
    button.disabled = true;
    button.textContent = "Calculando...";
    setNotice("rag-message");
    renderRagas({ evaluated: false, message: "Calculando metricas RAGAS..." });
    try {
      const report = await api.ragas({
        query: state.lastRagPayload.query,
        answer: state.lastRagResult.answer,
        sources: state.lastRagResult.sources,
        reference_answer: byId("reference-answer").value.trim(),
      });
      state.lastRagResult.ragas = report;
      renderRagas(report);
    } catch (error) {
      setNotice("rag-message", error.message, true);
      renderRagas({ evaluated: false, message: "Nao foi possivel calcular RAGAS." });
    } finally {
      button.disabled = false;
      button.textContent = "Calcular";
      updateRagasAction();
    }
  });
}

function bindExport() {
  byId("export-rag-json").addEventListener("click", () => {
    const exportData = buildRagExport();
    if (!exportData) return;
    downloadFile(
      `chat-rag-${fileTimestamp()}.json`,
      JSON.stringify(exportData, null, 2),
      "application/json;charset=utf-8",
    );
  });

  byId("export-rag-csv").addEventListener("click", () => {
    const exportData = buildRagExport();
    if (!exportData) return;
    const columns = [
      "app",
      "export_type",
      "schema_version",
      "exported_at",
      "collection_name",
      "question",
      "query",
      "reference_answer",
      "answer",
      "answer_available",
      "method",
      "top_k",
      "candidate_k",
      "used_reranker",
      "latency_ms",
      "source_count",
      "rank",
      "retrieval_rank",
      "chunk_id",
      "document_id",
      "document_name",
      "page_number",
      "ordinal",
      "score",
      "dense_score",
      "bm25_score",
      "rerank_score",
      "ragas_evaluated",
      "ragas_faithfulness",
      "ragas_answer_relevancy",
      "ragas_context_precision",
      "ragas_context_recall",
      "ragas_factual_correctness",
      "ragas_context_utilization",
      "ragas_message",
      "content",
      "request_json",
      "ragas_json",
      "response_json",
      "source_json",
    ];
    downloadFile(
      `chat-rag-${fileTimestamp()}.csv`,
      `${toCsv(ragExportRows(exportData), columns)}\n`,
      "text/csv;charset=utf-8",
    );
  });
}

function bindAuthentication() {
  byId("auth-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const submit = byId("auth-submit");
    submit.disabled = true;
    submit.textContent = "Validando...";
    setNotice("auth-message");
    try {
      const response = await api.login(byId("auth-password").value);
      auth.token = response.access_token;
      auth.unlock();
      await loadCollections();
    } catch (error) {
      setNotice("auth-message", error.message, true);
    } finally {
      submit.disabled = false;
      submit.textContent = "Acessar chat";
    }
  });
  byId("logout-button").addEventListener("click", () => auth.lock());
}

async function restoreSession() {
  const accessToken = auth.token;
  renderMetricShell();
  if (!accessToken) {
    auth.lock();
    return;
  }
  try {
    await api.session();
    if (auth.token !== accessToken) return;
    auth.unlock();
    await loadCollections();
  } catch {
    if (auth.token === accessToken) auth.lock();
  }
}

bindCollections();
bindRag();
bindRagas();
bindExport();
bindAuthentication();
restoreSession();

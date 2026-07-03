const byId = (id) => document.getElementById(id);

const state = {
  collectionStorageKey: "rag-chat-selected-collection",
  collections: [],
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
    const headers = new Headers(options.headers || {});
    const accessToken = auth.token;
    if (accessToken && path !== "/api/auth/login") {
      headers.set("Authorization", `Bearer ${accessToken}`);
    }
    const response = await fetch(path, { ...options, headers });
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
  const names = ["Faithfulness", "Answer relevancy", "Context precision", "Context recall", "Answer correctness"];
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
    ? "Notas de 0 a 1 por juiz LLM no formato das metricas RAGAS."
    : report.message || "Metricas nao calculadas.";
}

function methodLabel(method, usedReranker) {
  const base = method === "dense" ? "EMBEDDING" : method === "bm25" ? "BM25" : "HIBRIDO";
  return `${base}${usedReranker ? " · RERANK" : ""}`;
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
    setNotice("rag-message");
    byId("run-summary").textContent = "Recuperando chunks e gerando resposta...";
    try {
      const data = await api.rag({
        query: byId("query").value.trim(),
        collection_name: state.collectionName,
        method: byId("method").value,
        top_k: Number(byId("top-k").value),
        candidate_k: Number(byId("candidate-k").value),
        use_reranker: byId("use-reranker").checked,
        evaluate_ragas: byId("evaluate-ragas").checked,
        reference_answer: byId("reference-answer").value.trim(),
      });
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
bindAuthentication();
restoreSession();

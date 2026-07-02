const byId = (id) => document.getElementById(id);

const state = {
  collectionStorageKey: "rag-ingest-selected-collection",
  collections: [],
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
    const headers = new Headers(options.headers || {});
    const accessToken = auth.token;
    if (accessToken && path !== "/api/auth/login") {
      headers.set("Authorization", `Bearer ${accessToken}`);
    }
    const response = await fetch(path, { ...options, headers });
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
  upload(data) { return this.request("/api/documents/upload", { method: "POST", body: data }); },
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
    data.append("collection_name", state.collectionName);
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
      await loadCollections(state.collectionName);
      await Promise.all([loadDocuments(), loadPoints()]);
    } catch (error) {
      setNotice("upload-message", error.message, true);
    } finally {
      button.textContent = "Ingerir arquivos";
      updateFiles();
    }
  });
}

function bindCollections() {
  byId("collection-select").addEventListener("change", async (event) => {
    state.collectionName = event.target.value;
    setNotice("collection-message");
    byId("search-results").className = "search-results empty-state";
    byId("search-results").textContent = "Faça uma pergunta para ver os chunks recuperados.";
    byId("result-method").textContent = "Aguardando busca";
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
  byId("search-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const submit = event.submitter;
    submit.disabled = true;
    submit.textContent = "Buscando…";
    setNotice("search-message");
    try {
      const data = await api.search({
        query: byId("query").value.trim(),
        collection_name: state.collectionName,
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

async function loadDashboard() {
  renderMetrics({ evaluated: false, message: "Informe chunks relevantes para habilitar a avaliação supervisionada." }, 5);
  await loadCollections();
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
bindAuthentication();
byId("refresh-documents").addEventListener("click", loadDocuments);
byId("refresh-points").addEventListener("click", loadPoints);
restoreSession();

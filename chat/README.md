# RAG modular - chat

Modulo de perguntas sobre as collections criadas no `ingest/`.

## Subir localmente

Deixe o `ingest` rodando primeiro, depois:

```bash
cd chat
docker compose up --build
```

Abra `http://localhost:8081`. A API do chat fica em `http://localhost:8011/docs`.

## Recursos

- Escolha da collection usada como base.
- Busca por embedding dense, BM25 sparse ou hibrida.
- Reranker opcional sobre um conjunto maior de candidatos.
- Fluxo RAG orquestrado com LangGraph.
- Resposta gerada por API compativel com OpenAI, citando as fontes recuperadas.
- Painel RAGAS oficial com Faithfulness, Answer relevancy, Context precision, Context recall e Factual correctness.

`Context precision`, `Context recall` e `Factual correctness` usam melhor a resposta de referencia no campo opcional da tela. Sem referencia, o painel usa `ContextUtilization` como proxy para `Context precision` e deixa as metricas que exigem referencia sem nota.

## Benchmark RAGAS oficial por ground truth

Para experimento reproduzivel, use `chat/api/benchmarks/ragas/`. O site ja calcula RAGAS oficial pergunta a pergunta; o benchmark recebe um JSONL com `query` e `reference_answer`, roda a API do chat e salva `results.jsonl`, `responses.csv` e `summary.json` em uma pasta com timestamp. Depois, o script `evaluate_official.py` recalcula as metricas oficiais em lote e salva CSV/JSON para o artigo.

Exemplo:

```bash
cd chat/api
CHAT_APP_PASSWORD=alterar-esta-senha python benchmarks/ragas/run_groundtruth.py \
  --cases benchmarks/ragas/ground_truth.example.jsonl \
  --base-url http://localhost:8011 \
  --collection rag_chunks
```

Instale e rode o avaliador oficial:

```bash
pip install -r benchmarks/ragas/requirements-ragas.txt
RAGAS_LLM_API_KEY=sua-chave python benchmarks/ragas/evaluate_official.py
```

## Dependencias externas

O chat nao indexa documentos diretamente. Ele chama a API do `ingest` configurada por `INGEST_API_URL` e usa `INGEST_APP_PASSWORD` para obter um token interno.

A geracao e o RAGAS oficial usam:

```text
LLM_API_KEY
RAGAS_MODEL
RAGAS_EMBEDDING_MODEL
```

`RAGAS_MODEL` e opcional; se ficar vazio, usa `LLM_MODEL`. O embedding padrao do RAGAS oficial e `sentence-transformers/all-MiniLM-L6-v2`.

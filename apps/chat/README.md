# RAG modular - chat

Modulo de perguntas sobre as collections criadas no `apps/ingest/`.

## Subir localmente

Deixe o `ingest` rodando primeiro, depois:

```bash
cd apps/chat
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

`Context precision`, `Context recall` e `Factual correctness` usam a resposta de referencia no campo opcional da tela. Sem referencia, o painel mostra `Context utilization` como uma metrica separada e deixa as metricas que exigem referencia sem nota.

Para manter a avaliacao consistente, o chat registra os IDs dos chunks efetivamente enviados ao gerador. `Faithfulness` usa apenas esses chunks; as metricas de recuperacao usam todos os chunks retornados, sem truncamento adicional. O relatorio tambem registra a versao do RAGAS e os modelos do avaliador e de embeddings.

## Benchmark RAGAS oficial por ground truth

Para experimento reproduzivel, use `apps/chat/api/benchmarks/ragas/`. O site ja calcula RAGAS oficial pergunta a pergunta; o benchmark recebe um JSONL com `query` e `reference_answer`, roda a API do chat e salva `results.jsonl`, `responses.csv` e `summary.json` em uma pasta com timestamp. Depois, o script `evaluate_official.py` recalcula as metricas oficiais em lote e salva CSV/JSON para o artigo.

Exemplo:

```bash
cd apps/chat/api
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
```

`LLM_BASE_URL` e `LLM_MODEL` continuam no `.env` para voce escolher o provedor/modelo. O RAGAS oficial usa o mesmo provedor e modelo por padrao, mas aceita um juiz independente e fixo por meio de `RAGAS_MODEL`, `RAGAS_LLM_BASE_URL` e `RAGAS_LLM_API_KEY`. Para experimentos do artigo, prefira separar o juiz do modelo gerador. O embedding local padrao e `BAAI/bge-m3`.

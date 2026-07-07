# Comandos de benchmarks

Este arquivo concentra os comandos para rodar as tres avaliacoes do projeto:

- SciQ: baseline automatizada de recuperacao no `ingest`.
- Ground truth manual do ingest: perguntas suas + `chunk_ids` relevantes.
- Ground truth RAGAS oficial do chat: perguntas suas + resposta esperada.

## Antes de rodar

Suba o `ingest` quando o benchmark chamar a API local:

```bash
cd ingest
docker compose up --build
```

Suba tambem o `chat` quando for rodar RAGAS:

```bash
cd chat
docker compose up --build
```

Use as senhas dos seus `.env` nos comandos abaixo. Para VPS, troque `--base-url` pelas URLs publicas, por exemplo `https://ingest.136-248-79-252.sslip.io` ou `https://chat.136-248-79-252.sslip.io`.

## 1. Pipeline base com SciQ

Use quando quiser uma baseline automatizada com dataset publico. O SciQ prepara corpus, queries e qrels; indexa no Qdrant; roda `bm25`, `dense` e `hybrid`; e calcula as metricas.

Instalar dependencias:

```bash
cd ingest/api
pip install -r requirements.txt
pip install -r benchmarks/sciq/requirements-benchmark.txt
```

Rodar tudo:

```bash
python benchmarks/sciq/run_all.py --collection sciq_baseline --recreate
```

Smoke test rapido:

```bash
python benchmarks/sciq/run_all.py \
  --collection sciq_baseline \
  --recreate \
  --methods hybrid \
  --limit-queries 50
```

Se estiver rodando de fora do compose local, normalmente use:

```bash
python benchmarks/sciq/run_all.py \
  --collection sciq_baseline \
  --recreate \
  --qdrant-url http://localhost:6335
```

Se estiver dentro do container da API do ingest, use:

```bash
python benchmarks/sciq/run_all.py \
  --collection sciq_baseline \
  --recreate \
  --qdrant-url http://qdrant:6333
```

Saida:

```text
ingest/api/benchmarks/sciq/data/runs/YYYYMMDD-HHMMSS/
├── retrieval/{method}_{split}.jsonl
├── results/{method}_{split}_metrics.json
├── results/{method}_{split}_metrics.csv
└── summary_{split}.json
```

## 2. Ground truth manual do ingest

Use quando quiser avaliar suas proprias collections. Aqui voce monta uma bateria manual com pergunta e lista de chunks relevantes. O script chama `/api/search` e compara os resultados com `relevant_chunk_ids`.

Arquivo de exemplo:

```json
{"id":"q1","collection_name":"rag_chunks","query":"Qual ponto do documento responde esta pergunta?","relevant_chunk_ids":["chunk_id_1","chunk_id_2"]}
```

Crie seu arquivo, por exemplo:

```text
ingest/api/benchmarks/groundtruth/meu_ground_truth.jsonl
```

Rodar `bm25`, `dense` e `hybrid`:

```bash
cd ingest/api
INGEST_APP_PASSWORD=alterar-esta-senha python benchmarks/groundtruth/run_groundtruth.py \
  --cases benchmarks/groundtruth/meu_ground_truth.jsonl \
  --base-url http://localhost:8010 \
  --collection rag_chunks
```

Rodar apenas um metodo:

```bash
INGEST_APP_PASSWORD=alterar-esta-senha python benchmarks/groundtruth/run_groundtruth.py \
  --cases benchmarks/groundtruth/meu_ground_truth.jsonl \
  --base-url http://localhost:8010 \
  --collection rag_chunks \
  --methods hybrid
```

Rodar contra a VPS:

```bash
INGEST_APP_PASSWORD=alterar-esta-senha python benchmarks/groundtruth/run_groundtruth.py \
  --cases benchmarks/groundtruth/meu_ground_truth.jsonl \
  --base-url https://ingest.136-248-79-252.sslip.io \
  --collection rag_chunks
```

Saida:

```text
ingest/api/benchmarks/groundtruth/data/runs/YYYYMMDD-HHMMSS/
├── results.jsonl
├── metrics.csv
└── summary.json
```

## 3. Ground truth RAGAS oficial do chat

Use quando quiser avaliar a resposta gerada pelo RAG. Aqui o ground truth e uma resposta esperada, nao uma lista de chunks. O fluxo academico tem duas etapas: primeiro coletar respostas e fontes do chat; depois calcular as metricas com a biblioteca oficial `ragas`.

Arquivo de exemplo:

```json
{"id":"q1","collection_name":"rag_chunks","query":"Pergunta para o RAG","reference_answer":"Resposta esperada baseada nos documentos."}
```

Crie seu arquivo, por exemplo:

```text
chat/api/benchmarks/ragas/meu_ground_truth.jsonl
```

Instalar dependencias oficiais:

```bash
cd chat/api
pip install -r benchmarks/ragas/requirements-ragas.txt
```

Coletar respostas local:

```bash
cd chat/api
CHAT_APP_PASSWORD=alterar-esta-senha python benchmarks/ragas/run_groundtruth.py \
  --cases benchmarks/ragas/meu_ground_truth.jsonl \
  --base-url http://localhost:8011 \
  --collection rag_chunks
```

Coletar respostas sem reranker:

```bash
CHAT_APP_PASSWORD=alterar-esta-senha python benchmarks/ragas/run_groundtruth.py \
  --cases benchmarks/ragas/meu_ground_truth.jsonl \
  --base-url http://localhost:8011 \
  --collection rag_chunks \
  --no-reranker
```

Coletar respostas contra a VPS:

```bash
CHAT_APP_PASSWORD=alterar-esta-senha python benchmarks/ragas/run_groundtruth.py \
  --cases benchmarks/ragas/meu_ground_truth.jsonl \
  --base-url https://chat.136-248-79-252.sslip.io \
  --collection rag_chunks
```

Calcular RAGAS oficial na ultima rodada:

```bash
cd chat/api
RAGAS_LLM_API_KEY=sua-chave python benchmarks/ragas/evaluate_official.py \
  --llm-base-url https://openrouter.ai/api/v1 \
  --llm-model deepseek/deepseek-v4-flash
```

Calcular RAGAS oficial em uma rodada especifica:

```bash
RAGAS_LLM_API_KEY=sua-chave python benchmarks/ragas/evaluate_official.py \
  --results benchmarks/ragas/data/runs/YYYYMMDD-HHMMSS/results.jsonl \
  --llm-base-url https://openrouter.ai/api/v1 \
  --llm-model deepseek/deepseek-v4-flash
```

Saida:

```text
chat/api/benchmarks/ragas/data/runs/YYYYMMDD-HHMMSS/
├── results.jsonl
├── responses.csv
├── summary.json
└── ragas-official/YYYYMMDD-HHMMSS/
    ├── official_ragas_results.jsonl
    ├── official_ragas_metrics.csv
    └── official_ragas_summary.json
```

## 4. Exportacao pelo frontend

As telas tambem permitem baixar a execucao atual em JSON ou CSV:

- `ingest`: depois de rodar uma busca, use `Exportar JSON` ou `Exportar CSV` no card de metricas de ranking. Ele registra a pergunta, parametros, metricas e chunks recuperados; o campo `answer` fica vazio porque o ingest nao gera resposta.
- `chat`: depois de rodar uma pergunta, use `JSON` ou `CSV` no card RAGAS. Ele registra pergunta, resposta gerada, resposta de referencia, parametros, fontes recuperadas e metricas. Se clicar em `Calcular` antes, as metricas RAGAS entram no arquivo exportado.

Os CSVs sao propositalmente amplos: incluem colunas normalizadas e colunas `*_json` com os objetos brutos da execucao. Use esses arquivos para evidencia visual, inspecao rapida, prints da aplicacao e analises exploratorias. Para a tabela principal do artigo, prefira as rodadas dos scripts acima, porque elas fixam os casos, metodos, timestamps e saidas em pastas versionaveis.

## Protocolo de replicabilidade

1. Fixe o corpus/collection e registre estrategia de chunking, modelo dense, sparse, reranker, LLM e parametros (`top_k`, `candidate_k`).
2. Monte ground truths versionados:
   - `ingest`: `query` + `relevant_chunk_ids`.
   - `chat`: `query` + `reference_answer`.
3. Rode todos os metodos nos mesmos casos (`bm25`, `dense`, `hybrid`, com/sem reranker quando aplicavel).
4. Preserve os artefatos brutos (`results.jsonl`), tabelas (`*.csv`) e sumarios (`*.json`) gerados em `benchmarks/**/data/runs/YYYYMMDD-HHMMSS/`.
5. Faca as analises estatisticas sobre dados pareados por `case_id/query` e metodo:
   - Friedman para comparar mais de dois metodos.
   - Wilcoxon pareado para pos-teste entre pares.
   - Holm-Bonferroni para corrigir multiplas comparacoes.
   - Cliff's Delta para tamanho de efeito.
   - Spearman/Kendall para correlacionar metricas de RI (`precision`, `recall`, `map`, `ndcg`, `mrr`) com metricas RAG/RAGAS.
6. Gere os graficos a partir dos CSVs consolidados: tabela principal, heatmap de correlacao, boxplot por metodo, barras de medias/IC, scatter RI x RAG, pipeline visual e prints das telas.

## Qual usar?

| Caso | Use |
| --- | --- |
| Baseline academica automatizada de recuperacao | SciQ |
| Comparar `bm25`, `dense` e `hybrid` nos seus documentos | Ground truth manual do ingest |
| Avaliar se a resposta final do chat esta correta e fiel | Ground truth RAGAS oficial do chat |

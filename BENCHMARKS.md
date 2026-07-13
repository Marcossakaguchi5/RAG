# Comandos de benchmarks

Este arquivo concentra os comandos dos quatro estudos, que compartilham o mesmo nucleo
de recuperacao:

- SciQ: baseline automatizada de recuperacao no `ingest`.
- QASPER: recuperação de evidências textuais em artigos acadêmicos longos, com
  múltiplos parágrafos relevantes por pergunta.
- PDF: perguntas, respostas e evidencias canonicas materializadas para os chunks.
- RAG: extensao exploratoria com as mesmas perguntas e respostas do estudo PDF.

O foco academico e RI. O RAG so deve ser executado depois que o ground truth de PDF
estiver congelado. O desenho completo, as RQs e os controles estatisticos estao em
[PLANO_EXPERIMENTAL.md](PLANO_EXPERIMENTAL.md).
O roteiro recomendado, já com ordem, serviços, IDs e saídas, está em
[EXECUCAO_COMPLETA.md](EXECUCAO_COMPLETA.md).

O QASPER é executado diretamente em
[`ingest/api/benchmarks/qasper/README.md`](ingest/api/benchmarks/qasper/README.md).
Seu primeiro recorte é somente de perguntas respondíveis com evidência textual;
tabelas, figuras e perguntas não respondíveis pertencem a experimentos separados.

Para criar uma rodada nova sem sobrescrever a baseline existente, prefira o
orquestrador:

```bash
python3 experiments/run_experiment.py sciq \
  --qdrant-url http://localhost:6335 \
  --dataset-revision 2c94ad3e1aafab77146f384e23536f97a4849815 \
  --embedding-model-revision 5617a9f61b028005a4858fdac845db406aefb181
python3 experiments/run_experiment.py pdf-ir --help
python3 experiments/run_experiment.py rag --help
```

Cada execução acadêmica usa um diretório timestampado em
`experiments/runs/novas/<estudo>/` e grava um `manifest.json`. As rodadas anteriores
ficam congeladas em `experiments/runs/antigos/`. Os comandos detalhados abaixo
continuam disponíveis para diagnóstico.

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

Rodar tudo e ja exportar os casos para o runner manual do ingest (`query -> relevant_chunk_ids`):

```bash
python benchmarks/sciq/run_all.py \
  --collection sciq_baseline \
  --recreate \
  --export-groundtruth
```

Rodar tudo, exportar ground truth e gerar graficos:

```bash
python benchmarks/sciq/run_all.py \
  --collection sciq_baseline \
  --recreate \
  --export-groundtruth \
  --plot-results
```

Por padrão, o SciQ usa `recursive_text`. Para comparar chunking textual, rode coleções separadas:

```bash
python benchmarks/sciq/run_all.py \
  --collection sciq_recursive \
  --recreate \
  --chunking-strategy recursive_text \
  --export-groundtruth \
  --plot-results

python benchmarks/sciq/run_all.py \
  --collection sciq_fixed_token \
  --recreate \
  --chunking-strategy fixed_token \
  --export-groundtruth \
  --plot-results
```

As estratégias `docling_*` são para PDFs na aplicação; no SciQ, o corpus já vem em texto.

Preparar/ingerir e exportar o ground truth sem rodar retrieval/evaluation nessa etapa:

```bash
python benchmarks/sciq/run_all.py \
  --collection sciq_baseline \
  --recreate \
  --export-groundtruth \
  --skip-retrieval
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
├── statistics/metrics_with_ci.{csv,json}
├── statistics/paired_differences.{csv,json}
└── summary_{split}.json
```

Os artefatos `statistics/` são gerados pelo orquestrador acadêmico. Eles incluem
IC95% bootstrap por método e contrastes pareados, com McNemar exato e ajuste de Holm
para Hit@k.

Converter SciQ para o runner manual do ingest, gerando automaticamente `query -> relevant_chunk_ids`:

```bash
cd ingest/api
python benchmarks/sciq/export_groundtruth.py \
  --collection sciq_baseline \
  --split test \
  --output benchmarks/groundtruth/sciq_ground_truth.jsonl

INGEST_APP_PASSWORD=alterar-esta-senha python benchmarks/groundtruth/run_groundtruth.py \
  --cases benchmarks/groundtruth/sciq_ground_truth.jsonl \
  --base-url http://localhost:8010 \
  --collection sciq_baseline
```

Gerar graficos de uma rodada ja existente:

```bash
cd ingest/api
python benchmarks/sciq/plot_results.py \
  --run-dir benchmarks/sciq/data/runs/YYYYMMDD-HHMMSS
```

Saida dos graficos:

```text
plots/
├── metrics_at_10.svg
├── metric_curves.svg
├── latency_by_method.svg
├── first_relevant_rank.svg
├── metrics_consolidated.csv
└── report.html
```

## 2. Pipeline de RI com PDFs

No estudo academico, nao anote UUIDs do Qdrant como fonte primaria. Guarde pergunta,
resposta esperada e uma citacao do PDF. Depois materialize automaticamente a citacao
para os chunks de cada configuracao. Assim, trocar o chunking ou reingerir o PDF nao
invalida a anotacao original.

Caso mestre:

```json
{"id":"q1","split":"test","category":"conceitual","query":"Pergunta sobre o material","reference_answer":"Resposta sustentada pelo PDF.","evidence":[{"document_name":"material.pdf","page":12,"quote":"Citacao literal que sustenta a resposta.","relevance":2}]}
```

O esquema, a politica de normalizacao e os testes ficam em
`experiments/groundtruth/`.

### 2.0 Auditar citações no PDF-fonte

Antes de criar os chunks, valide hash, página e correspondência literal:

```bash
python3 experiments/groundtruth/audit_pdf.py \
  --cases experiments/cases/freire_pilot.draft.jsonl \
  --pdf importancia_ato_ler.pdf \
  --report experiments/data/freire-pilot/source-audit.json
```

O conjunto piloto atual possui 32 casos e 37 evidências, todos ainda com estado
`draft/silver`; ele precisa de revisão humana antes de constituir o teste final.

### 2.1 Exportar os chunks da collection

Na raiz do repositorio:

```bash
python3 ingest/api/benchmarks/groundtruth/export_chunks.py \
  --collection freire_recursive \
  --qdrant-url http://localhost:6335 \
  --pdf importancia_ato_ler.pdf \
  --output experiments/data/freire_recursive/chunks.jsonl
```

### 2.2 Materializar evidencias para chunks

```bash
python3 experiments/groundtruth/materialize.py materialize \
  --cases experiments/data/freire/cases.jsonl \
  --chunks experiments/data/freire_recursive/chunks.jsonl \
  --ingest-out experiments/data/freire_recursive/ingest-groundtruth.jsonl \
  --ragas-out experiments/data/freire_recursive/ragas-groundtruth.jsonl \
  --report-out experiments/data/freire_recursive/matching-report.json
```

O comando falha se alguma evidencia nao for localizada. O relatorio registra
cobertura, melhores candidatos e hashes dos insumos.

O runner usa `relevant_chunk_ids` para Hit/Precision/Recall/MAP/MRR e
`relevance_by_chunk` para nDCG graduada. Qualquer caso com erro invalida a rodada por
padrao, embora os artefatos de diagnostico sejam preservados.

### 2.3 Rodar BM25, denso e hibrido

```bash
cd ingest/api
INGEST_APP_PASSWORD=alterar-esta-senha python benchmarks/groundtruth/run_groundtruth.py \
  --cases ../../experiments/data/freire_recursive/ingest-groundtruth.jsonl \
  --base-url http://localhost:8010 \
  --collection freire_recursive
```

Para a comparacao principal, mantenha um chunking fixo. Execute a ablation de
chunking separadamente, repetindo exportacao e materializacao para outra collection.

Rodar apenas um metodo continua disponivel para diagnostico:

```bash
INGEST_APP_PASSWORD=alterar-esta-senha python benchmarks/groundtruth/run_groundtruth.py \
  --cases ../../experiments/data/freire_recursive/ingest-groundtruth.jsonl \
  --base-url http://localhost:8010 \
  --collection freire_recursive \
  --methods hybrid
```

Rodar contra a VPS:

```bash
INGEST_APP_PASSWORD=alterar-esta-senha python benchmarks/groundtruth/run_groundtruth.py \
  --cases ../../experiments/data/freire_recursive/ingest-groundtruth.jsonl \
  --base-url https://ingest.136-248-79-252.sslip.io \
  --collection freire_recursive
```

Saida:

```text
ingest/api/benchmarks/groundtruth/data/runs/YYYYMMDD-HHMMSS/
├── results.jsonl
├── metrics.csv
└── summary.json
```

Quando executado pelo orquestrador, o diretorio superior da rodada tambem recebe:

```text
plots/
├── metrics_with_ci.csv
├── metrics_with_ci.json
├── metrics_with_ci.svg
├── paired_differences.csv
├── paired_differences.json
├── paired_chunking_differences.csv
├── paired_chunking_differences.json
└── report.html
```

## 3. Ground truth RAGAS oficial do chat

Use somente como extensao do estudo PDF. A projecao RAGAS ja e gerada pelo mesmo
materializador, evitando um segundo ground truth independente. O fluxo tem duas
etapas: primeiro coletar respostas, fontes recuperadas e IDs dos chunks efetivamente
enviados ao gerador; depois calcular as metricas offline. `Faithfulness` usa os
contextos enviados ao gerador, enquanto `context_precision` e `context_recall` usam
todos os contextos recuperados.

Arquivo de exemplo:

```json
{"id":"q1","collection_name":"rag_chunks","query":"Pergunta para o RAG","reference_answer":"Resposta esperada baseada nos documentos."}
```

Use a saida gerada na Secao 2, por exemplo:

```text
experiments/data/freire_recursive/ragas-groundtruth.jsonl
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
  --cases ../../experiments/data/freire_recursive/ragas-groundtruth.jsonl \
  --base-url http://localhost:8011 \
  --collection freire_recursive
```

Coletar respostas sem reranker:

```bash
CHAT_APP_PASSWORD=alterar-esta-senha python benchmarks/ragas/run_groundtruth.py \
  --cases ../../experiments/data/freire_recursive/ragas-groundtruth.jsonl \
  --base-url http://localhost:8011 \
  --collection freire_recursive \
  --no-reranker
```

Coletar respostas contra a VPS:

```bash
CHAT_APP_PASSWORD=alterar-esta-senha python benchmarks/ragas/run_groundtruth.py \
  --cases ../../experiments/data/freire_recursive/ragas-groundtruth.jsonl \
  --base-url https://chat.136-248-79-252.sslip.io \
  --collection freire_recursive
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
2. Monte um unico ground truth mestre: `query`, `reference_answer` e evidencias
   canonicas; derive dele os `relevant_chunk_ids` e os casos RAGAS.
3. Rode todos os metodos nos mesmos casos (`bm25`, `dense`, `hybrid`, com/sem reranker quando aplicavel).
4. Preserve os artefatos brutos (`results.jsonl`), tabelas (`*.csv`) e sumarios (`*.json`) gerados em `benchmarks/**/data/runs/YYYYMMDD-HHMMSS/`.
5. Faca as analises estatisticas sobre dados pareados por `case_id/query` e metodo:
   - bootstrap pareado e IC95% para diferencas de MRR/nDCG.
   - McNemar para Hit@k entre pares.
   - teste de randomizacao ou Wilcoxon pareado quando aplicavel.
   - procedimento de Holm para corrigir multiplas comparacoes.
   - tamanho de efeito adequado ao desenho pareado.
   - Spearman/Kendall para correlacionar metricas de RI (`precision`, `recall`, `map`, `ndcg`, `mrr`) com metricas RAG/RAGAS.
6. Gere os graficos a partir dos CSVs consolidados: tabela principal, heatmap de correlacao, boxplot por metodo, barras de medias/IC, scatter RI x RAG, pipeline visual e prints das telas.

## Qual usar?

| Caso | Use |
| --- | --- |
| Baseline academica automatizada de recuperacao | SciQ |
| Comparar `bm25`, `dense` e `hybrid` nos PDFs | Ground truth canonico + materializacao |
| Avaliar a resposta final sem duplicar o gabarito | Projecao RAGAS do mesmo ground truth |

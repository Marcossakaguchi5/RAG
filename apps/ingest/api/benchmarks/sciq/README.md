# SciQ Retrieval Benchmark

Benchmark isolado para medir a baseline de recuperação do módulo `ingest`.
Ele usa o SciQ como conjunto supervisionado:

- `support` vira documento indexado.
- `question` vira consulta.
- `query_id -> support_doc_id` vira qrels.
- `correct_answer` fica salvo para uma etapa futura de geração, mas não entra no índice.

Como o corpus SciQ já vem em texto, ele não passa pelo Docling. A etapa de ingestão usa o mesmo chunker dinâmico do módulo `ingest`, então supports longos podem virar múltiplos chunks. A avaliação continua em nível de documento/support: qualquer chunk recuperado do support correto conta como acerto para o `doc_id` relevante.

## Preparar ambiente

Execute os comandos a partir de `apps/ingest/api`:

```bash
pip install -r requirements.txt
pip install -r benchmarks/sciq/requirements-benchmark.txt
```

Por padrão, os scripts do benchmark usam:

```bash
QDRANT_URL=http://localhost:6335
SPARSE_LANGUAGE=english
FASTEMBED_CACHE_DIR=benchmarks/sciq/data/model_cache/fastembed
```

Se estiver dentro do container da API, passe `--qdrant-url http://qdrant:6333`.

## Fluxo

Tudo em um comando:

```bash
python benchmarks/sciq/run_all.py --collection sciq_baseline --recreate
```

Se o Qdrant estiver em outro endereço:

```bash
python benchmarks/sciq/run_all.py --collection sciq_baseline --recreate --qdrant-url http://localhost:6335
```

Para um smoke test rápido:

```bash
python benchmarks/sciq/run_all.py --collection sciq_baseline --recreate --methods hybrid --limit-queries 50
```

Para preparar, ingerir, avaliar e também gerar o arquivo `query -> relevant_chunk_ids` para o runner manual:

```bash
python benchmarks/sciq/run_all.py \
  --collection sciq_baseline \
  --recreate \
  --export-groundtruth
```

Para fazer a rodada completa e gerar graficos SVG/HTML ao final:

```bash
python benchmarks/sciq/run_all.py \
  --collection sciq_baseline \
  --recreate \
  --export-groundtruth \
  --plot-results
```

Por padrão o SciQ usa `recursive_text`. Como o SciQ é corpus textual, as estratégias suportadas neste benchmark são:

- `recursive_text`
- `fixed_token`

As estratégias `docling_*` continuam disponíveis para ingestão de PDFs pela aplicação, mas não se aplicam diretamente ao SciQ porque o dataset já vem como texto e não possui `DoclingDocument`.

Para comparar estratégias de chunking, rode coleções separadas:

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

Para apenas preparar/ingerir e exportar o ground truth, sem rodar retrieval/evaluation nessa etapa:

```bash
python benchmarks/sciq/run_all.py \
  --collection sciq_baseline \
  --recreate \
  --export-groundtruth \
  --skip-retrieval
```

Ou, passo a passo:

```bash
python benchmarks/sciq/prepare_sciq.py
python benchmarks/sciq/ingest_corpus.py --collection sciq_baseline --recreate
python benchmarks/sciq/run_retrieval.py --collection sciq_baseline --method bm25 --split test --top-k 10
python benchmarks/sciq/run_retrieval.py --collection sciq_baseline --method dense --split test --top-k 10
python benchmarks/sciq/run_retrieval.py --collection sciq_baseline --method hybrid --split test --top-k 10
python benchmarks/sciq/evaluate_retrieval.py --run ../../benchmark/runs/antigos/sciq/20260702-legacy/retrieval/bm25_test.jsonl --split test
python benchmarks/sciq/evaluate_retrieval.py --run ../../benchmark/runs/antigos/sciq/20260702-legacy/retrieval/dense_test.jsonl --split test
python benchmarks/sciq/evaluate_retrieval.py --run ../../benchmark/runs/antigos/sciq/20260702-legacy/retrieval/hybrid_test.jsonl --split test
```

## Exportar para ground truth do ingest

Se quiser reaproveitar o SciQ no runner manual de ground truth do `ingest`, gere os casos com `query` e `relevant_chunk_ids` automaticamente:

```bash
python benchmarks/sciq/export_groundtruth.py \
  --collection sciq_baseline \
  --split test \
  --output benchmarks/groundtruth/sciq_ground_truth.jsonl
```

O mesmo arquivo também pode ser gerado durante o `run_all.py` com `--export-groundtruth`.

Depois rode o benchmark manual pela API do ingest:

```bash
INGEST_APP_PASSWORD=alterar-esta-senha python benchmarks/groundtruth/run_groundtruth.py \
  --cases benchmarks/groundtruth/sciq_ground_truth.jsonl \
  --base-url http://localhost:8010 \
  --collection sciq_baseline
```

Para um teste curto:

```bash
python benchmarks/sciq/export_groundtruth.py \
  --collection sciq_baseline \
  --split test \
  --limit-queries 50 \
  --top-k 10 \
  --output benchmarks/groundtruth/sciq_ground_truth_50.jsonl
```

O exportador usa os mesmos IDs determinísticos do `ingest_corpus.py`, por exemplo `sciq_doc_..._chunk_0000`. Para a estratégia padrão `recursive_text`, ele reconstrói os IDs sem carregar Docling/PDF; se você alterar `CHUNK_MIN_WORDS`, `CHUNK_SIZE_WORDS` ou `CHUNK_OVERLAP_WORDS` na ingestão SciQ, passe os mesmos valores no exportador.

## Graficos

Para gerar graficos de uma rodada ja existente:

```bash
python benchmarks/sciq/plot_results.py \
  --run-dir ../../benchmark/runs/novas/sciq/ID_DA_RODADA
```

Se `--run-dir` nao for informado, o script usa a rodada timestampada mais recente. As saidas ficam em `plots/` dentro da propria rodada:

```text
plots/
├── metrics_at_10.svg
├── metric_curves.svg
├── latency_by_method.svg
├── first_relevant_rank.svg
├── metrics_consolidated.csv
└── report.html
```

## Saídas

Os arquivos preparados ficam em `benchmarks/sciq/data/processed/`:

- `processed/corpus.jsonl`
- `processed/queries.jsonl`
- `processed/qrels.jsonl`

Cada execução completa de `run_all.py` cria uma pasta nova com timestamp:

```text
benchmark/runs/novas/sciq/ID_DA_RODADA/
├── retrieval/{method}_{split}.jsonl
├── results/{method}_{split}_metrics.json
├── results/{method}_{split}_metrics.csv
└── summary_{split}.json
```

Para resultados acadêmicos, use `benchmark/run_experiment.py`; ele define
automaticamente `benchmark/runs/novas/sciq/<run-id>`. A opção `--run-dir` do script
interno permanece apenas para diagnóstico.

As métricas calculadas são `hit_rate`, `precision`, `recall`, `MAP`, `NDCG` e `MRR` para cada `k`.

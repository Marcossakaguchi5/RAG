# SciQ Retrieval Benchmark

Benchmark isolado para medir a baseline de recuperação do módulo `ingest`.
Ele usa o SciQ como conjunto supervisionado:

- `support` vira documento indexado.
- `question` vira consulta.
- `query_id -> support_doc_id` vira qrels.
- `correct_answer` fica salvo para uma etapa futura de geração, mas não entra no índice.

## Preparar ambiente

Execute os comandos a partir de `ingest/api`:

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

Ou, passo a passo:

```bash
python benchmarks/sciq/prepare_sciq.py
python benchmarks/sciq/ingest_corpus.py --collection sciq_baseline --recreate
python benchmarks/sciq/run_retrieval.py --collection sciq_baseline --method bm25 --split test --top-k 10
python benchmarks/sciq/run_retrieval.py --collection sciq_baseline --method dense --split test --top-k 10
python benchmarks/sciq/run_retrieval.py --collection sciq_baseline --method hybrid --split test --top-k 10
python benchmarks/sciq/evaluate_retrieval.py --run benchmarks/sciq/data/runs/bm25_test.jsonl --split test
python benchmarks/sciq/evaluate_retrieval.py --run benchmarks/sciq/data/runs/dense_test.jsonl --split test
python benchmarks/sciq/evaluate_retrieval.py --run benchmarks/sciq/data/runs/hybrid_test.jsonl --split test
```

## Saídas

Os arquivos gerados ficam em `benchmarks/sciq/data/`:

- `processed/corpus.jsonl`
- `processed/queries.jsonl`
- `processed/qrels.jsonl`
- `runs/{method}_{split}.jsonl`
- `results/{method}_{split}_metrics.json`
- `results/{method}_{split}_metrics.csv`
- `results/summary_{split}.json`

As métricas calculadas são `hit_rate`, `precision`, `recall`, `MAP`, `NDCG` e `MRR` para cada `k`.

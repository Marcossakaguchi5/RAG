# Benchmark manual de recuperacao

Este runner avalia o modulo `ingest` com um ground truth criado manualmente. A ideia e simples: para cada pergunta, voce informa a lista de `chunk_ids` que deveriam aparecer. O script chama `/api/search`, calcula as metricas que a API ja retorna e salva uma rodada com timestamp.

Ele e diferente do benchmark SciQ:

- SciQ usa um dataset pronto e avalia em nivel de documento/support.
- Este benchmark usa sua propria collection e avalia exatamente os `chunk_ids` que voce marcou como relevantes.

## Formato do ground truth

Crie um arquivo `.jsonl` com uma pergunta por linha:

```json
{"id":"q1","collection_name":"minha_collection","query":"Pergunta de teste","relevant_chunk_ids":["chunk_a","chunk_b"]}
```

Campos aceitos:

- `id`: identificador opcional da pergunta.
- `query` ou `question`: pergunta.
- `relevant_chunk_ids` ou `relevant_ids`: lista de chunks relevantes.
- `collection_name`: collection usada naquela pergunta. Se nao vier, usa `--collection`.
- `top_k` e `method`: opcionais por pergunta. Se `method` vier no caso, ele sobrescreve `--methods` para aquela pergunta.

## Rodar

Com o ingest rodando:

```bash
cd ingest/api
INGEST_APP_PASSWORD=alterar-esta-senha python benchmarks/groundtruth/run_groundtruth.py \
  --cases benchmarks/groundtruth/ground_truth.example.jsonl \
  --base-url http://localhost:8010 \
  --collection rag_chunks
```

Por padrao, roda `bm25`, `dense` e `hybrid`. Para limitar:

```bash
python benchmarks/groundtruth/run_groundtruth.py --methods hybrid --cases meu_ground_truth.jsonl
```

Tambem da para passar a senha por `--password`.

## Saidas

Cada execucao cria:

```text
benchmarks/groundtruth/data/runs/YYYYMMDD-HHMMSS/
├── results.jsonl
├── metrics.csv
└── summary.json
```

- `results.jsonl`: resposta completa por pergunta/metodo, chunks retornados e metricas.
- `metrics.csv`: uma linha por pergunta/metodo, facil de abrir em planilha.
- `summary.json`: medias das metricas por metodo e contagem de erros.

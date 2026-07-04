# Benchmark RAGAS oficial com ground truth

Este benchmark usa duas etapas:

1. `run_groundtruth.py` chama a API do `chat`, deixa o LangGraph executar recuperação, reranking e geração, e salva pergunta, resposta e fontes em uma pasta com timestamp.
2. `evaluate_official.py` lê o `results.jsonl` e calcula as métricas pela biblioteca oficial `ragas`.

O painel RAGAS do site tambem usa a biblioteca oficial `ragas`; esta pipeline existe para rodar uma bateria inteira e salvar CSV/JSON reproduziveis para o artigo.

## Formato do ground truth

Crie um arquivo `.json` com uma lista de perguntas ou um `.jsonl` com uma pergunta por linha:

```json
[
  {"id":"q1","collection_name":"minha_collection","query":"Pergunta para o RAG","reference_answer":"Resposta esperada"}
]
```

Campos aceitos:

- `id`: identificador opcional da pergunta.
- `query` ou `question`: pergunta.
- `reference_answer` ou `expected_answer`: resposta esperada usada pelo RAGAS oficial.
- `collection_name`: collection usada naquela pergunta. Se não vier, usa `--collection`.
- `method`, `top_k`, `candidate_k` e `use_reranker`: opcionais por pergunta.

## Instalar dependencias oficiais

Em um ambiente Python do `chat/api`:

```bash
pip install -r benchmarks/ragas/requirements-ragas.txt
```

## 1. Coletar respostas

Com o chat rodando:

```bash
cd chat/api
CHAT_APP_PASSWORD=alterar-esta-senha python benchmarks/ragas/run_groundtruth.py \
  --cases benchmarks/ragas/ground_truth.example.jsonl \
  --base-url http://localhost:8011 \
  --collection rag_chunks
```

Também dá para passar a senha por `--password`.

Por padrão, o coletor não executa o avaliador RAGAS durante a chamada da API para evitar duplicar custo e tempo. Se quiser salvar tambem o relatório retornado pelo site, use `--site-ragas`; ele usa a mesma biblioteca oficial.

## 2. Calcular RAGAS oficial

Use o `results.jsonl` da rodada que acabou de ser criada:

```bash
cd chat/api
RAGAS_LLM_API_KEY=sua-chave python benchmarks/ragas/evaluate_official.py \
  --results benchmarks/ragas/data/runs/YYYYMMDD-HHMMSS/results.jsonl \
  --llm-base-url https://openrouter.ai/api/v1 \
  --llm-model deepseek/deepseek-v4-flash
```

Se não passar `--results`, o script usa automaticamente a última rodada encontrada em `benchmarks/ragas/data/runs/`.

Métricas oficiais padrão:

- `faithfulness`
- `context_precision`
- `context_recall`
- `factual_correctness`
- `answer_relevancy`

A métrica `answer_relevancy` usa embedding local `sentence-transformers/all-MiniLM-L6-v2` por padrão. Para trocar:

```bash
RAGAS_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
RAGAS_LLM_API_KEY=sua-chave \
python benchmarks/ragas/evaluate_official.py
```

## Saidas

Cada execução cria:

```text
benchmarks/ragas/data/runs/YYYYMMDD-HHMMSS/
├── results.jsonl
├── responses.csv
├── summary.json
└── ragas-official/YYYYMMDD-HHMMSS/
    ├── official_ragas_results.jsonl
    ├── official_ragas_metrics.csv
    └── official_ragas_summary.json
```

- `results.jsonl`: resposta completa da API por pergunta e fontes usadas.
- `responses.csv`: uma linha por pergunta, sem métricas oficiais.
- `official_ragas_metrics.csv`: métricas oficiais por pergunta.
- `official_ragas_summary.json`: médias oficiais e contagem de erros.

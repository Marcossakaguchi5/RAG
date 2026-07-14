# QASPER Retrieval Benchmark

Segundo benchmark de recuperação acadêmica para o módulo `ingest`. Ele transforma
cada parágrafo textual de um artigo QASPER em um documento canônico e avalia a
recuperação das evidências anotadas para cada pergunta.

O recorte padrão é deliberadamente textual e respondível:

- descarta perguntas com anotação não respondível ou discordante;
- descarta perguntas cuja evidência anotada inclui tabela ou figura (`FLOAT SELECTED`);
- usa somente os parágrafos de evidência explicitamente anotados como qrels, com
  `relevance = 2`;
- mantém as respostas anotadas em `reference_answer` e `reference_answers`, mas
  não as indexa.

Isso evita marcar todo o artigo como relevante e torna Recall/nDCG sensíveis à
recuperação de múltiplas evidências distribuídas no texto. A extensão multimodal
deve ser uma rodada separada, depois de definir extração de tabelas, figuras e captions.

## Preparar e executar

Execute em `apps/ingest/api`:

```bash
pip install -r requirements.txt
pip install -r benchmarks/qasper/requirements-benchmark.txt

python benchmarks/qasper/run_all.py \
  --collection qasper_text_baseline \
  --recreate \
  --split test
```

O QASPER público disponibiliza `train`, `validation` e `test`; portanto `test` é o
split de avaliação padrão. Para uma verificação rápida:

```bash
python benchmarks/qasper/run_all.py \
  --collection qasper_smoke \
  --recreate \
  --methods hybrid \
  --limit-queries 50
```

O preparador lê diretamente os arquivos Parquet oficiais em uma revisão fixada do
dataset. Isso é necessário porque versões recentes de `datasets` não executam mais
o script legado `qasper.py`.

Para exportar os casos no formato do runner manual:

```bash
python benchmarks/qasper/run_all.py \
  --collection qasper_text_baseline \
  --recreate \
  --export-groundtruth
```

Os três artefatos canônicos ficam em `data/processed/`:

```text
corpus.jsonl   # um parágrafo por doc_id, com paper_id e seção
queries.jsonl  # pergunta, respostas de referência e tipos de resposta
qrels.jsonl    # todos os parágrafos de evidência anotados (relevance=2)
```

O chunker ainda pode dividir parágrafos excepcionalmente longos. A recuperação é
normalizada novamente em `doc_id`, portanto encontrar qualquer chunk de um
parágrafo de evidência conta como recuperar esse parágrafo.

`--include-float-evidence` e `--include-unanswerable` existem apenas para
inspeção dos dados: eles não tornam essas perguntas adequadas ao benchmark
textual padrão e não devem ser usados para comparar o primeiro experimento.

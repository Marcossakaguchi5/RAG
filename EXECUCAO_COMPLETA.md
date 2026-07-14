# Execução completa da pipeline acadêmica

Execute os comandos a partir da raiz do repositório. Use um terminal único para que
as variáveis exportadas permaneçam disponíveis.

## 1. Preparar o Python local

```bash
python3 -m venv .venv-academic
source .venv-academic/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ingest/api/requirements.txt
python -m pip install -r ingest/api/benchmarks/sciq/requirements-benchmark.txt
```

A primeira execução pode baixar vários gigabytes de modelos. Não interrompa o
download do BGE-M3 ou do Docling.

## 2. Subir ingest, Qdrant e dependências

Escolha uma senha e mantenha o commit do embedding igual ao usado no benchmark:

```bash
export INGEST_APP_PASSWORD='troque-por-uma-senha-forte'
export EMBEDDING_MODEL_REVISION='5617a9f61b028005a4858fdac845db406aefb181'
export MYSQL_PORT=3308
export MINIO_API_PORT=9010
export MINIO_CONSOLE_PORT=9011

docker compose -f ingest/docker-compose.yml up -d --build
docker compose -f ingest/docker-compose.yml ps
curl --fail http://localhost:8010/health
curl --fail http://localhost:6335/
```

As portas alternativas de MySQL e MinIO evitam conflito com outros containers já
ativos neste computador. Elas não alteram o experimento, pois API e dependências se
comunicam pela rede interna do Compose.

## 3. Rodar a baseline SciQ atualizada

```bash
python experiments/run_experiment.py sciq \
  --run-id sciq-atualizada-01 \
  --qdrant-url http://localhost:6335 \
  --dataset-revision 2c94ad3e1aafab77146f384e23536f97a4849815 \
  --embedding-model-revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --methods bm25,dense,hybrid \
  --split test \
  --top-k 10 \
  --k 1,3,5,10 \
  --bootstrap-repetitions 2000
```

Saída: `experiments/runs/novas/sciq/sciq-atualizada-01/`.

O comando prepara o SciQ, recria uma coleção limpa, indexa o corpus, executa 878
consultas para os três métodos, calcula métricas, IC95%, McNemar/Holm e gera gráficos.

## 4. Rodar o experimento principal PDF/RI

Os 50 casos de `freire_50.gold.jsonl` foram revisados e aprovados por anotador
humano. Por isso, esta rodada não usa `--allow-draft-cases`. O extrator
`pdftotext` é indicado explicitamente para preservar a correspondência das citações
com o texto deste PDF.

```bash
python experiments/run_experiment.py pdf-ir \
  --run-id pdf-principal-01 \
  --pdf importancia_ato_ler.pdf \
  --cases experiments/cases/freire_50.gold.jsonl \
  --collection freire_recursive_gold_50_01 \
  --chunking-strategies recursive_text \
  --methods bm25,dense,hybrid \
  --top-k 10 \
  --source-audit-extractor pdftotext \
  --bootstrap-repetitions 2000
```

Saída: `experiments/runs/novas/pdf-ir/pdf-principal-01/`.

O comando audita o PDF, envia e segmenta o documento, exporta os chunks, converte as
69 evidências em IDs relevantes, executa 50 perguntas × 3 recuperadores e produz
métricas e gráficos. Se uma citação não couber integralmente em nenhum chunk, a
execução para e deixa um relatório de diagnóstico, em vez de calcular métricas com
qrels incompletos.

## 5. Rodar a ablação de chunking

Esta etapa mantém o recuperador híbrido fixo e varia apenas a segmentação. Assim ela
não é confundida com a comparação principal entre recuperadores.

Use o mesmo conjunto gold aprovado da etapa 4. Cada estratégia precisa materializar
integralmente as 69 evidências; uma evidência pode usar até três chunks estruturais
ordenados da mesma página quando eles recompõem exatamente a citação. A execução
para com diagnóstico se ainda houver tokens não mapeados, em vez de reduzir a
cobertura exigida.

```bash
python experiments/run_experiment.py pdf-ir \
  --run-id pdf-chunking-gold-01 \
  --pdf importancia_ato_ler.pdf \
  --cases experiments/cases/freire_50.gold.jsonl \
  --collection-prefix freire_chunking_gold_50_01 \
  --chunking-strategies fixed_token,recursive_text,docling_hierarchical,docling_hybrid \
  --methods hybrid \
  --top-k 10 \
  --source-audit-extractor pdftotext \
  --bootstrap-repetitions 2000
```

Saída: `experiments/runs/novas/pdf-ir/pdf-chunking-gold-01/`. Além dos agregados de cada
estratégia, `plots/paired_chunking_differences.*` contém os contrastes pareados entre
chunkings mantendo o método híbrido fixo.

## 6. Subir o chat

Configure uma chave do provedor compatível com OpenAI. A senha do ingest deve ser a
mesma usada na etapa 2.

```bash
export CHAT_APP_PASSWORD='troque-por-outra-senha-forte'
export LLM_API_KEY='sua-chave-do-provedor'
export RAGAS_LLM_API_KEY="$LLM_API_KEY"

docker compose -f chat/docker-compose.yml up -d --build
docker compose -f chat/docker-compose.yml ps
curl --fail http://localhost:8011/health
```

## 7. Rodar RAG e, opcionalmente, RAGAS

Primeiro instale o avaliador no mesmo ambiente Python usado pelo orquestrador:

```bash
python -m pip install -r chat/api/requirements.txt
```

Rodada completa com geração e avaliação RAGAS oficial:

```bash
python experiments/run_experiment.py rag \
  --run-id rag-principal-01 \
  --cases experiments/runs/novas/pdf-ir/pdf-principal-01/recursive_text/ragas-groundtruth.jsonl \
  --collection freire_recursive_gold_50_01 \
  --methods bm25,dense,hybrid \
  --top-k 5 \
  --evaluate-ragas
```

Saída: `experiments/runs/novas/rag/rag-principal-01/`. São até 150 respostas geradas
(50 perguntas × 3 métodos). O RAGAS faz chamadas adicionais ao modelo juiz e pode
gerar custo no provedor. Para coletar somente as respostas, retire `--evaluate-ragas`.

## 8. Cuidados com a rodada final

O arquivo `freire_50.gold.jsonl` já registra os 50 casos como aprovados. Para manter
a rastreabilidade do resultado principal:

1. não alterar o arquivo gold depois de iniciar a rodada principal;
2. registrar no artigo o hash do arquivo de casos presente no `manifest.json`;
3. usar novos `run_id` e collection se uma nova versão dos casos for necessária;
4. não ajustar parâmetros após observar os resultados desta coleção.

Se um `run_id` já existir, use `-02`, `-03` etc. O orquestrador não sobrescreve uma
rodada anterior.

## 9. O que será obtido

```text
experiments/runs/
├── antigos/
│   └── sciq/{20260702-legacy,20260707-183922}/
└── novas/
    ├── sciq/sciq-atualizada-01/
    │   ├── manifest.json
    │   ├── summary_test.json
    │   ├── retrieval/*.jsonl
    │   ├── results/*.{json,csv}
    │   ├── plots/*.{svg,csv,html}
    │   └── statistics/*.{json,csv}
    ├── pdf-ir/pdf-principal-01/
    │   ├── manifest.json
    │   ├── source-audit.json
    │   ├── recursive_text/
    │   │   ├── chunks.jsonl
    │   │   ├── ingest-groundtruth.jsonl
    │   │   ├── ragas-groundtruth.jsonl
    │   │   ├── matching-report.json
    │   │   └── retrieval/{results.jsonl,metrics.csv,summary.json}
    │   └── plots/*.{json,csv,svg,html}
    └── rag/rag-principal-01/
        ├── manifest.json
        └── {bm25,dense,hybrid}/
            ├── results.jsonl
            ├── responses.csv
            ├── summary.json
            └── ragas-official/*.{jsonl,json,csv}
```

Esses comandos não modificam automaticamente a tabela do artigo. Eles produzem os
artefatos rastreáveis dos quais a nova tabela, os gráficos finais e a discussão devem
ser derivados.

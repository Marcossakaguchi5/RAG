# Pipeline acadêmica

Esta pasta separa os experimentos reproduzíveis das interfaces web. O código de
extração, embeddings, Qdrant e busca continua sendo reutilizado; a interface não é a
fonte das métricas ou tabelas do artigo.

## Desenho simplificado

```text
SciQ ───────────────┐
                    ├─> BM25 / denso / híbrido -> métricas RI -> estatística -> gráficos
PDF + evidências ───┘
                              |
                              └─> RAG/RAGAS opcional
```

São dois estudos principais de RI e uma extensão:

1. **SciQ:** dataset, consultas e qrels prontos; baseline controlada.
2. **PDF:** perguntas e evidências canônicas; os IDs de chunk são derivados para cada
   configuração de segmentação.
3. **RAG:** reutiliza os casos aprovados do PDF e não cria outro gabarito.

Para evitar uma matriz difícil de interpretar, a comparação principal varia o
recuperador com chunking fixo. O efeito do chunking é uma ablação separada.

## Onde executar

- Ponto de entrada único para novas rodadas: `benchmark/run_experiment.py`.
- SciQ e gráficos: `apps/ingest/api/benchmarks/sciq/run_all.py`.
- Exportação dos chunks do Qdrant:
  `apps/ingest/api/benchmarks/groundtruth/export_chunks.py`.
- Esquema mestre e materialização: [groundtruth/README.md](groundtruth/README.md).
- RI em PDFs: `apps/ingest/api/benchmarks/groundtruth/run_groundtruth.py`.
- Coleta e RAGAS offline: `apps/chat/api/benchmarks/ragas/`.
- Comandos completos: [BENCHMARKS.md](../BENCHMARKS.md).
- Decisões metodológicas e critérios de saída:
  [PLANO_EXPERIMENTAL.md](../PLANO_EXPERIMENTAL.md).

## Estado atual

- SciQ: pipeline completa e rodada preliminar disponível; a avaliação de subconjuntos
  foi corrigida.
- PDF: esquema canônico, auditor do PDF, exportador de chunks e materializador
  implementados; falta a revisão humana do conjunto real. Há 32 casos
  `development` redigidos e ancorados em 37 evidências em
  [cases/freire_pilot.draft.jsonl](cases/freire_pilot.draft.jsonl) apenas para validar
  o fluxo técnico e conduzir o piloto. Eles permanecem `draft/silver`.
- RAG: coletor e avaliador existem, mas ainda não há rodada acadêmica real.

O manifesto da rodada SciQ que corresponde às figuras atuais está em
[runs/antigos/sciq/20260707-183922/manifest.json](runs/antigos/sciq/20260707-183922/manifest.json).
Ele é explicitamente preliminar porque a execução original não preservou todas as
versões de modelos, dependências e hardware.

## Rodar novamente sem sobrescrever a baseline

Prepare um ambiente limpo; o arquivo extra do SciQ não repete nem atualiza
silenciosamente as dependências fixadas pela aplicação:

```bash
python3 -m venv .venv-academic
source .venv-academic/bin/activate
pip install -r apps/ingest/api/requirements.txt
pip install -r apps/ingest/api/benchmarks/sciq/requirements-benchmark.txt
```

Nova rodada SciQ completa, com pasta, collection e manifesto timestampados:

```bash
python benchmark/run_experiment.py sciq \
  --qdrant-url http://localhost:6335 \
  --dataset-revision 2c94ad3e1aafab77146f384e23536f97a4849815 \
  --embedding-model-revision 5617a9f61b028005a4858fdac845db406aefb181
```

Além dos rankings, métricas e gráficos, a pasta `statistics/` recebe IC95% bootstrap
e diferenças pareadas com McNemar/Holm. O número de repetições pode ser fixado com
`--bootstrap-repetitions` e fica registrado no manifesto.

O commit SciQ acima corresponde à revisão já presente no cache local e agora também é
o padrão do orquestrador; o commit seguinte fixa o BGE-M3. O manifesto registra
ambiente, estado do Git, parâmetros, fingerprints do dataset e hashes dos artefatos.

Nova rodada PDF/RI, incluindo auditoria do PDF, upload, chunking, exportação,
materialização e busca:

```bash
export INGEST_APP_PASSWORD='sua-senha'
python benchmark/groundtruth/audit_pdf.py \
  --cases benchmark/cases/freire_pilot.draft.jsonl \
  --pdf benchmark/sources/importancia_ato_ler.pdf

python benchmark/run_experiment.py pdf-ir \
  --pdf benchmark/sources/importancia_ato_ler.pdf \
  --cases benchmark/cases/freire_pilot.draft.jsonl \
  --allow-draft-cases \
  --chunking-strategies recursive_text \
  --methods bm25,dense,hybrid \
  --top-k 10
```

O primeiro comando permite inspecionar a auditoria isoladamente; o orquestrador a
repete e arquiva `source-audit.json` antes de gastar tempo com a ingestão.

Ao final, a rodada contém os resultados por consulta e, em `plots/`, CSV/JSON com
médias e IC95% bootstrap, diferenças pareadas, McNemar exato com ajuste de Holm para
Hit@k, gráfico SVG e relatório HTML. Quando há mais de um chunking, também produz
`paired_chunking_differences.*`. Use `--no-plot-results` apenas em diagnósticos.

Os 32 casos fornecidos são uma anotação `silver` de desenvolvimento. Por isso o
comando piloto exige o reconhecimento explícito `--allow-draft-cases`. Sem essa
opção, o orquestrador aceita apenas casos `approved/adjudicated`.

Depois de aprovar os casos, a extensão RAG pode executar os mesmos métodos sem a
heurística de reranking:

```bash
export CHAT_APP_PASSWORD='sua-senha'
python benchmark/run_experiment.py rag \
  --cases benchmark/runs/novas/pdf-ir/ID_DA_RODADA/recursive_text/ragas-groundtruth.jsonl \
  --collection COLLECTION_REGISTRADA_NO_MANIFESTO \
  --methods bm25,dense,hybrid
```

Adicione `--evaluate-ragas` somente quando o ambiente do juiz e a chave
`RAGAS_LLM_API_KEY` estiverem configurados.

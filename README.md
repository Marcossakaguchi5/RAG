# RAG modular

Projeto de Recuperação Aumentada por Geração (RAG) dividido em dois módulos independentes:

- `ingest/`: recebe PDFs, extrai texto, cria chunks, gera índices dense e sparse, e disponibiliza busca semântica/BM25/híbrida.
- `chat/`: consome as bases criadas pelo `ingest`, recupera contexto, opcionalmente aplica reranker, gera respostas com uma API compatível com OpenAI e calcula métricas RAGAS.

## Arquitetura

```text
PDFs
  |
  v
ingest/
  FastAPI + MySQL + MinIO + Qdrant
  - PDF original no MinIO
  - metadados e chunks no MySQL
  - vetores dense/sparse no Qdrant
  |
  v
chat/
  FastAPI + interface web
  - consulta collections do ingest
  - recupera fontes
  - gera resposta via LLM
  - avalia com RAGAS
```

## Estrutura do repositório

```text
.
|-- ingest/                 # Pipeline de ingestão, indexação e recuperação
|   |-- api/                # API FastAPI
|   |-- web/                # Interface web estática
|   |-- docker-compose.yml
|   `-- README.md
|-- chat/                   # Interface e API de perguntas sobre as collections
|   |-- api/                # API FastAPI do chat
|   |-- web/                # Interface web estática
|   |-- docker-compose.yml
|   `-- README.md
|-- shared/                 # Espaço para arquivos compartilhados
`-- LICENSE
```

## Pré-requisitos

- Docker
- Docker Compose
- Acesso à internet na primeira execução do `ingest`, para baixar os modelos de embedding e sparse retrieval
- Uma chave de LLM compatível com a API da OpenAI/OpenRouter para usar o `chat`

## Subir o ambiente local

### 1. Ingestão

```bash
cd ingest
cp .env.example .env
docker compose up --build
```

Serviços principais:

| Serviço | URL padrão |
| --- | --- |
| Interface do ingest | http://localhost:8080 |
| API do ingest | http://localhost:8010/docs |
| MinIO Console | http://localhost:9001 |
| Qdrant | http://localhost:6335 |

O painel e as rotas `/api/*` usam a senha `INGEST_APP_PASSWORD`. Troque o valor padrão no `.env` antes de expor o serviço.

### 2. Chat

Com o `ingest` rodando, configure o ambiente do chat e suba o módulo:

```bash
cd ../chat
docker compose up --build
```

Serviços principais:

| Serviço | URL padrão |
| --- | --- |
| Interface do chat | http://localhost:8081 |
| API do chat | http://localhost:8011/docs |

O `chat` usa `INGEST_API_URL` para chamar a API de ingestão e `INGEST_APP_PASSWORD` para autenticar nessa API. Para geração de resposta e avaliação RAGAS, configure no `chat/.env`:

```text
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=
RAGAS_MODEL=
```

`RAGAS_MODEL` pode ficar vazio para reutilizar `LLM_MODEL`.

## Fluxo de uso

1. Acesse `http://localhost:8080`.
2. Faça login com a senha configurada em `INGEST_APP_PASSWORD`.
3. Crie ou selecione uma collection.
4. Envie PDFs para ingestão.
5. Teste a busca por `bm25`, `dense` ou `hybrid`.
6. Acesse `http://localhost:8081`.
7. Escolha a collection, faça uma pergunta e confira a resposta com as fontes recuperadas.

## Métodos de recuperação

- `bm25`: busca sparse usando BM25 com stemming em português.
- `dense`: busca vetorial por similaridade de cosseno.
- `hybrid`: combina sparse e dense com Reciprocal Rank Fusion (RRF).

No módulo `chat`, o reranker pode refinar os candidatos antes de montar o contexto para o LLM.

## Avaliação

O projeto inclui dois caminhos de avaliação:

- No `ingest`, métricas clássicas de recuperação como Precision@K, Recall@K, MAP, NDCG@K e MRR, desde que exista ground truth de chunks relevantes.
- No `chat`, relatório RAGAS com Faithfulness, Answer relevancy, Context precision, Context recall e Answer correctness.

Há também um benchmark SciQ em `ingest/api/benchmarks/sciq/`.

## Comandos úteis

Subir o módulo de ingestão:

```bash
cd ingest
docker compose up --build
```

Subir o módulo de chat:

```bash
cd chat
docker compose up --build
```

Parar os serviços de um módulo:

```bash
docker compose down
```

Parar e remover volumes do módulo atual:

```bash
docker compose down -v
```

Use `down -v` com cuidado: ele remove dados persistidos em volumes Docker, incluindo documentos, chunks, índices e caches do módulo correspondente.

## Documentação dos módulos

- [ingest/README.md](ingest/README.md)
- [chat/README.md](chat/README.md)

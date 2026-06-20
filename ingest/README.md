# RAG modular — ingestão

Este repositório separa deliberadamente `ingest/` do futuro `chat/`. O módulo atual recebe PDFs, extrai e divide o texto, armazena o original no MinIO, os metadados/chunks no MySQL e vetores dense+sparse no Qdrant.

## Subir localmente

```bash
cp .env.example .env
docker compose up --build
```

Abra [http://localhost:8080](http://localhost:8080). A documentação interativa da API fica em [http://localhost:8010/docs](http://localhost:8010/docs), o console MinIO em [http://localhost:9001](http://localhost:9001) e o Qdrant atende em `http://localhost:6335`.

O painel e todas as rotas `/api/*` exigem a senha de `INGEST_APP_PASSWORD`. Para desenvolvimento, ela está definida no `.env.example`; troque-a no `.env` (ou nas variáveis do stack do Portainer) antes de expor o serviço. A sessão do navegador dura 8 horas por padrão e pode ser ajustada com `INGEST_AUTH_TOKEN_TTL_SECONDS`. Apenas `/health` permanece público para monitoramento.

Na primeira subida, o `sentence-transformers/all-MiniLM-L6-v2` será baixado para o volume persistente `model_cache`. Isso requer acesso à internet e pode levar alguns minutos; nas próximas inicializações ele é reutilizado.

## Estrutura

```text
ingest/
├── api/
│   └── app/
│       ├── services/     # PDF, embeddings, armazenamento, índices e recuperação
│       ├── models/       # Entidades MySQL
│       ├── schemas.py    # Contratos HTTP
│       └── main.py       # Rotas FastAPI
└── web/                  # Interface estática servida por Nginx
```

| Componente | Responsabilidade |
| --- | --- |
| MinIO | arquivo PDF original |
| MySQL | documentos, chunks, páginas e metadados |
| Qdrant | vetores dense, sparse BM25, payload e fusão híbrida |
| FastAPI | pipeline de ingestão e API de recuperação |
| Nginx | interface e proxy `/api` |

## Recuperação

Há três métodos disponíveis na segunda aba:

- `bm25`: vetor sparse BM25 (`Qdrant/bm25`, com stemming em português) e índice invertido com IDF no Qdrant.
- `dense`: similaridade de cosseno no Qdrant usando `sentence-transformers/all-MiniLM-L6-v2`.
- `hybrid`: busca sparse+dense e Reciprocal Rank Fusion (RRF) nativa no Qdrant.

Para o primeiro corte, os clientes diretos são mais adequados que LangChain: o pipeline fica simples de depurar, as fronteiras entre armazenamentos são explícitas e a avaliação não fica escondida atrás de abstrações. LangChain pode entrar depois no `chat/` (chains, tools, memória) ou como adaptador, sem acoplar essa ingestão a ele.

## Métricas

Precision@K, Recall@K, MAP, NDCG@K e MRR exigem ground truth. Por isso, a tela permite colar IDs de chunks relevantes retornados na busca. Sem esses IDs, a API sinaliza corretamente que a avaliação não foi executada — não há uma métrica confiável para uma pergunta sem rótulo de relevância.

Para um benchmark real, o próximo passo natural é cadastrar um conjunto de pares `pergunta → chunk_ids relevantes` em uma tabela própria e rodar os três métodos sobre todo o conjunto, calculando a média das métricas por método.

## Limites desta primeira versão

- Aceita apenas PDFs com texto extraível. PDFs escaneados precisarão de uma etapa OCR.
- A coleção do Qdrant é um índice derivado do MySQL. Ao detectar a coleção densa legada, a API recria o formato dense+sparse e reindexa os chunks canônicos automaticamente na inicialização.
- Ainda não há exclusão de documentos nem fila assíncrona de ingestão. A autenticação atual é deliberadamente simples, baseada em uma senha fixa de ambiente, e deve evoluir para gestão de usuários antes de produção em escala.

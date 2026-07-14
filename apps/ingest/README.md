# RAG modular — ingestão

Este repositório separa deliberadamente `apps/ingest/` do futuro `apps/chat/`. O módulo atual recebe PDFs, extrai o conteúdo com Docling, divide o documento em chunks dinâmicos, armazena o original no MinIO, os metadados/chunks no MySQL e vetores dense+sparse no Qdrant.

## Subir localmente

```bash
cp .env.example .env
docker compose up --build
```

Abra [http://localhost:8080](http://localhost:8080). A documentação interativa da API fica em [http://localhost:8010/docs](http://localhost:8010/docs), o console MinIO em [http://localhost:9001](http://localhost:9001) e o Qdrant atende em `http://localhost:6335`.

O painel e todas as rotas `/api/*` exigem a senha de `INGEST_APP_PASSWORD`. Para desenvolvimento, ela está definida no `.env.example`; troque-a no `.env` (ou nas variáveis do stack do Portainer) antes de expor o serviço. Apenas `/health` permanece público para monitoramento.

Na primeira subida, o `BAAI/bge-m3`, o modelo sparse e os modelos do Docling podem ser baixados para o volume persistente `model_cache` via `HF_HOME=/models`. Isso requer acesso à internet e pode levar alguns minutos; nas próximas inicializações eles são reutilizados.

## Estrutura

```text
apps/ingest/
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

## Extração e chunking

Por padrão, a extração usa Docling para converter PDFs em uma representação textual/Markdown mais rica que a extração simples por página. A API tenta primeiro Docling sem OCR e, se não houver texto suficiente, Docling com OCR seletivo. O OCR vem habilitado no código, mas `DOCLING_OCR_FORCE_FULL_PAGE=false` evita forçar OCR na página inteira quando o Docling consegue decidir regiões/páginas que precisam.

Não defina `DOCLING_ARTIFACTS_PATH` a menos que ele aponte para uma pasta completa de artefatos/modelos do Docling. Para cache comum, basta manter `HF_HOME=/models`; apontar `DOCLING_ARTIFACTS_PATH` para uma pasta vazia faz o Docling falhar antes da extração.

Antes do upload, a interface permite escolher a estratégia de chunking:

- `fixed_token`: baseline por janela fixa de tokens, usando o tokenizer do modelo de embeddings quando disponível.
- `recursive_text`: baseline textual recursivo por parágrafos, frases e palavras, com overlap.
- `docling_hierarchical`: chunker hierárquico nativo do Docling, baseado na estrutura detectada do documento.
- `docling_hybrid`: chunker híbrido nativo do Docling, com refinamento orientado a tokens sobre o hierárquico.
- `docling_hybrid_parent_child`: usa chunks híbridos como pais e indexa janelas menores como filhos com contexto do pai.
- `docling_hybrid_contextual`: usa o `contextualize()` do HybridChunker para enriquecer o texto com metadados estruturais antes do embedding.

O valor padrão pode ser sobrescrito por `CHUNKING_STRATEGY`. Os limites `CHUNK_SIZE_WORDS`, `CHUNK_OVERLAP_WORDS`, `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `PARENT_CHILD_SIZE_WORDS` e `PARENT_CHILD_OVERLAP_WORDS` controlam os tamanhos usados pelos baselines e pelas estratégias avançadas.

O campo `page_number` continua existindo para compatibilidade com a interface e as citações, mas em chunks que atravessam páginas ele deve ser interpretado como referência aproximada.

O PDF original é salvo no MinIO apenas depois que a extração e a criação dos chunks passam. Se o upload retorna erro 400 durante extração/OCR, o arquivo não fica persistido no MinIO.

## Metadados dos pontos

Cada chunk indexado no Qdrant mantém o `content` e metadados úteis para inspeção, filtros e avaliação:

- identificação: `chunk_id`, `point_id`, `document_id`, `collection_name`;
- documento: `document_name`, `file_name`, `file_extension`, `content_type`, `object_name`, `minio_bucket`, `document_size_bytes`, `document_page_count`;
- posição: `page_number`, `page_start`, `page_end`, `ordinal`, `chunk_index`, `chunk_total`, `is_first_chunk`, `is_last_chunk`;
- tamanho: `word_count`, `char_count`;
- indexação: `schema_version`, `source_type`, `chunking_strategy`, `embedding_model`, `sparse_model`, `indexed_at`.

Campos como `collection_name`, `document_id`, `document_name`, `page_number`, `ordinal`, `word_count` e `indexed_at` também recebem índice de payload no Qdrant para facilitar filtros futuros.

## Recuperação

Há três métodos disponíveis na segunda aba:

- `bm25`: vetor sparse BM25 (`Qdrant/bm25`, com stemming em português) e índice invertido com IDF no Qdrant.
- `dense`: similaridade de cosseno no Qdrant usando `BAAI/bge-m3`.
- `hybrid`: busca sparse+dense e Reciprocal Rank Fusion (RRF) nativa no Qdrant.

Ao trocar o modelo de embedding, a API recria a coleção Qdrant derivada se a dimensão densa existente não bater com o modelo ativo, e reindexa os chunks canônicos a partir do MySQL na inicialização.

Para o primeiro corte, os clientes diretos são mais adequados que LangChain: o pipeline fica simples de depurar, as fronteiras entre armazenamentos são explícitas e a avaliação não fica escondida atrás de abstrações. LangChain pode entrar depois no `apps/chat/` (chains, tools, memória) ou como adaptador, sem acoplar essa ingestão a ele.

## Métricas

Precision@K, Recall@K, MAP, NDCG@K e MRR exigem ground truth. Por isso, a tela permite marcar os chunks relevantes diretamente nos resultados ou colar IDs manualmente. Depois de marcar, use `Recalcular métricas` para reenviar a mesma pergunta com os IDs relevantes. Sem esses IDs, a API sinaliza corretamente que a avaliação não foi executada — não há uma métrica confiável para uma pergunta sem rótulo de relevância.

Para um benchmark real, o próximo passo natural é cadastrar um conjunto de pares `pergunta → chunk_ids relevantes` em uma tabela própria e rodar os três métodos sobre todo o conjunto, calculando a média das métricas por método.

Para uma bateria manual por terminal, use `apps/ingest/api/benchmarks/groundtruth/`. Ela recebe um JSONL com `query` e `relevant_chunk_ids`, roda `bm25`, `dense` e `hybrid` contra a API do ingest e salva `results.jsonl`, `metrics.csv` e `summary.json` em uma pasta com timestamp.

## Limites desta primeira versão

- Aceita PDFs que o Docling consiga converter para texto. PDFs escaneados acionam OCR seletivo quando não há texto extraível, mas podem exigir download de modelos e mais tempo de processamento.
- A coleção do Qdrant é um índice derivado do MySQL. Ao detectar a coleção densa legada, a API recria o formato dense+sparse e reindexa os chunks canônicos automaticamente na inicialização.
- Ainda não há exclusão de documentos nem fila assíncrona de ingestão. A autenticação atual é deliberadamente simples, baseada em uma senha fixa de ambiente, e deve evoluir para gestão de usuários antes de produção em escala.

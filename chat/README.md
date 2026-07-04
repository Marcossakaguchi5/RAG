# RAG modular - chat

Modulo de perguntas sobre as collections criadas no `ingest/`.

## Subir localmente

Deixe o `ingest` rodando primeiro, depois:

```bash
cd chat
docker compose up --build
```

Abra `http://localhost:8081`. A API do chat fica em `http://localhost:8011/docs`.

## Recursos

- Escolha da collection usada como base.
- Busca por embedding dense, BM25 sparse ou hibrida.
- Reranker opcional sobre um conjunto maior de candidatos.
- Fluxo RAG orquestrado com LangGraph.
- Resposta gerada por API compativel com OpenAI, citando as fontes recuperadas.
- Painel RAGAS com Faithfulness, Answer relevancy, Context precision, Context recall e Answer correctness.

`Context recall` e `Answer correctness` ficam mais confiaveis quando voce informa uma resposta de referencia no campo opcional da tela.

## Dependencias externas

O chat nao indexa documentos diretamente. Ele chama a API do `ingest` configurada por `INGEST_API_URL` e usa `INGEST_APP_PASSWORD` para obter um token interno.

A geracao e o juiz RAGAS usam:

```text
LLM_API_KEY
```

`LLM_BASE_URL`, `LLM_MODEL`, temperatura, limite de tokens e contexto máximo têm defaults no código e só precisam entrar no ambiente se você quiser sobrescrever.

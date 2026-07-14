# Aplicações

Os serviços executáveis do projeto ficam separados por responsabilidade:

- [ingest/](ingest/README.md): recebe documentos, cria chunks e expõe a busca
  BM25, densa e híbrida.
- [chat/](chat/README.md): consulta as collections do ingest e executa o fluxo
  RAG, com geração e avaliação RAGAS.

Os comandos de execução e as variáveis de ambiente estão nos READMEs de cada
aplicação.

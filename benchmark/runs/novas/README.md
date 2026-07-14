# Novas rodadas

O orquestrador cria automaticamente:

```text
novas/
├── sciq/<run-id>/
├── pdf-ir/<run-id>/
└── rag/<run-id>/
```

Use um `run_id` novo a cada execução. O runner recusa sobrescrever diretórios não
vazios.

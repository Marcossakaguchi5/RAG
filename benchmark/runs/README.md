# Rodadas acadêmicas

- `antigos/`: execuções históricas congeladas. Não reutilize suas coleções como
  destino de uma nova rodada.
- `novas/`: destino automático de `benchmark/run_experiment.py` para SciQ, PDF/RI
  e RAG.

Os artefatos volumosos permanecem ignorados pelo Git. Arquivos `README.md` e
`manifest.json` podem ser versionados para preservar metodologia, parâmetros e hashes.

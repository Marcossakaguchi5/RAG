# Casos mestres

Arquivos com sufixo `.draft.jsonl` são pré-anotações e não podem ser usados como
ground truth final. Antes da rodada de teste, cada caso precisa de revisão da pergunta,
resposta, citação, página, categoria e suficiência da evidência. A versão aprovada deve
registrar os anotadores e mudar `review_status` para `approved` ou `adjudicated`.

`freire_pilot.draft.jsonl` contém 32 perguntas de desenvolvimento, 32 respostas de
referência e 37 evidências distribuídas por 26 páginas do PDF local
`importancia_ato_ler.pdf`. A redação foi feita pelo Codex a partir das evidências e a
proveniência está explicitamente marcada como `ai-authored-evidence-grounded`.
Todas as citações foram localizadas exatamente nas páginas declaradas, mas isso não
substitui o julgamento humano de naturalidade, suficiência e relevância.

Auditar novamente hash, página e citação:

```bash
python3 benchmark/groundtruth/audit_pdf.py \
  --cases benchmark/cases/freire_pilot.draft.jsonl \
  --pdf benchmark/sources/importancia_ato_ler.pdf \
  --report benchmark/data/freire-pilot/source-audit.json
```

Como a obra contém restrições editoriais, o arquivo e as citações não devem ser
tratados automaticamente como um dataset redistribuível. Para o benchmark
publicável, prefira materiais educacionais com licença aberta.

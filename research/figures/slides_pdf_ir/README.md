# Gráficos para os slides — PDF-IR

Arquivos SVG em formato 16:9, prontos para inserir nos slides (o formato vetorial
permanece nítido ao redimensionar):

- `01_ato_de_ler_49_paginas.svg`: desempenho dos três recuperadores no corpus *Ato de Ler*;
- `02_livro_ri_581_paginas.svg`: desempenho no livro de Recuperação da Informação;
- `03_comparacao_49_vs_581_paginas.svg`: efeito do aumento do corpus em cada recuperador.
- `04_sciq_878_consultas.svg`: desempenho dos recuperadores no conjunto de teste SciQ.
- `05_sciq_recall_por_k.svg`: ganho de Recall/Hit ao ampliar o número de resultados do SciQ.
- `06_livro_ri_primeira_evidencia.svg`: posição em que cada método encontra a primeira evidência relevante no livro de RI.
- `07_heatmap_geral_tres_datasets.svg`: visão geral das quatro métricas nos três datasets.
- `tabela_amostras_perguntas_respostas.md`: duas perguntas e respostas de referência por dataset, em tabelas separadas.

Os três gráficos usam MRR@10, MAP@10, nDCG@10 e Recall@10 das rodadas completas
`pdf-principal-02` e `iir-principal-03`: 50 consultas, segmentação `recursive_text`,
top-10 e IC95% bootstrap (2.000 repetições). Os PDFs possuem, respectivamente, 49 e 581 páginas;
os nomes dos gráficos usam essas contagens auditadas, em vez dos arredondamentos
“50” e “500”.

O gráfico SciQ usa a rodada `sciq-atualizada-01`: 878 consultas de teste, 12.135
supports indexados e top-10. Como cada consulta tem uma única evidência relevante,
MAP@10 e MRR@10 coincidem nesse conjunto.

Para regerar após uma nova rodada:

```bash
python3 benchmark/generate_slide_pdf_ir_charts.py
```

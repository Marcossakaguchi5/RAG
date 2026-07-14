# Plano experimental consolidado: RI como estudo principal e RAG como extensão

## 1. Decisão de escopo

O trabalho deve ser apresentado como uma avaliação de Recuperação da Informação em
corpora educacionais. A avaliação RAG é uma extensão exploratória, executada somente
depois que corpus, perguntas e julgamentos de relevância estiverem congelados.

Há duas contagens diferentes que não devem ser misturadas:

- **três estudos**: SciQ, corpus próprio em PDF e avaliação RAG;
- **três recuperadores implementados no benchmark de RI**: BM25, denso e híbrido
  por RRF.

O repositório menciona uma quarta condição, híbrido com reranqueamento. O componente
atual do `chat` é uma heurística de combinação de escore e sobreposição lexical, não
um modelo reranqueador treinado, e ainda não participa do benchmark SciQ. Portanto,
essa condição não deve aparecer como resultado principal até que seja implementada e
avaliada no mesmo protocolo dos demais recuperadores.

## 2. Questões de pesquisa revisadas

- **RQ1 — desempenho de RI:** como BM25, recuperação densa e recuperação híbrida
  diferem em efetividade e latência no SciQ e em um corpus educacional em português?
- **RQ2 — robustez:** as diferenças entre os recuperadores se mantêm entre corpora e
  entre tipos de pergunta (definicional, conceitual, procedimental, comparativa e
  multi-evidência)?
- **RQ3 — construção do ground truth:** qual é a concordância entre o mapeamento
  semiautomático de evidências para chunks e os julgamentos humanos?
- **RQ4-E — extensão exploratória:** métricas de RI por pergunta estão associadas à
  correção e à fidelidade das respostas geradas pelo RAG?

As RQ1–RQ3 constituem a contribuição da disciplina de RI. A RQ4-E não deve bloquear
o artigo principal e precisa ser identificada como exploratória enquanto não houver
uma rodada RAG completa.

## 3. Pipeline única e artefatos

```text
corpus bruto + licença
  -> fingerprint do corpus e configuração de extração
  -> casos mestres (pergunta, resposta e evidências canônicas)
  -> materialização evidência -> qrels/chunk_ids para cada chunking
  -> runs pareados BM25, denso e híbrido
  -> métricas por pergunta + agregados + inferência estatística
  -> análise de erros
  -> geração RAG opcional, com o mesmo case_id e a mesma condição de RI
  -> RAGAS offline + associação RI/RAG
```

Nenhuma tabela do artigo deve ser construída a partir da interface web. A interface é
útil para inspeção e anotação; os resultados publicáveis devem vir de runners em lote.

### 3.1 Caso mestre

O caso mestre é independente do Qdrant, de UUIDs e da estratégia de chunking:

```json
{
  "id": "freire-q001",
  "split": "test",
  "query": "Pergunta avaliada",
  "reference_answer": "Resposta de referência sustentada pelo material",
  "category": "conceitual",
  "evidence": [
    {
      "document_name": "documento.pdf",
      "document_sha256": "sha256-opcional-mas-recomendado",
      "page": 12,
      "quote": "Trecho literal curto que ancora o julgamento",
      "relevance": 2
    }
  ],
  "provenance": {
    "creation": "human|llm-assisted",
    "review_status": "draft|approved",
    "annotators": ["A1"]
  }
}
```

O `quote` é a âncora primária. A página funciona como metadado de auditoria porque,
nas estratégias `fixed_token` e `recursive_text` atuais, o número da página é estimado
pela posição do chunk. O hash do PDF deve ser registrado para impedir que uma nova
edição do arquivo seja tratada silenciosamente como o mesmo corpus.

### 3.2 Relevância graduada

Usar a seguinte rubrica antes de iniciar a anotação:

- **0 — não relevante:** não ajuda a responder à pergunta;
- **1 — parcialmente relevante:** contém contexto útil, mas insuficiente sozinho;
- **2 — altamente relevante:** contém evidência suficiente para sustentar a resposta.

Para métricas binárias, os graus 1 e 2 podem ser considerados relevantes em uma
análise e apenas o grau 2 em uma análise de sensibilidade. Para nDCG, preservar os
graus originais. A regra escolhida deve constar no manifesto da rodada.

## 4. Definição semiautomática dos pontos relevantes

É possível automatizar grande parte do processo, mas o conjunto de teste só deve ser
chamado de **gold** depois de revisão humana. Antes disso, ele é **silver**.

### 4.1 Rota recomendada: evidence-first

1. Extrair blocos do PDF mantendo documento, página e texto.
2. Selecionar uma evidência autocontida.
3. Gerar ou redigir uma pergunta e uma resposta de referência exclusivamente a partir
   dessa evidência.
4. Guardar a citação literal, o hash do documento e a proveniência da criação.
5. Rejeitar perguntas triviais, duplicadas, ambíguas ou que dependam de conhecimento
   externo.
6. Revisar pergunta, resposta e evidência sem mostrar qual recuperador as encontrou.
7. Materializar automaticamente a evidência para os chunks de cada condição.

Essa rota conhece a evidência por construção, mas pode gerar perguntas lexicalmente
fáceis. Por isso, a amostra deve incluir paráfrases humanas e diferentes categorias.

### 4.2 Rota para perguntas humanas: pooling

1. Executar BM25, denso e híbrido até uma profundidade pré-definida, por exemplo 20.
2. Unir os resultados, remover duplicatas e incluir chunks vizinhos das evidências.
3. Embaralhar o pool e ocultar método, posição e escore.
4. Usar um LLM, se desejado, apenas para sugerir grau e citação exata.
5. Fazer julgamento humano 0/1/2 e adjudicar discordâncias.

Rotular somente os resultados visíveis de um único método cria viés circular. Itens não
julgados também não devem ser automaticamente descritos como irrelevantes sem que a
profundidade e o protocolo de pooling sejam reportados.

### 4.3 Validação da automação

- o materializador deve normalizar Unicode, espaços, caixa, pontuação e hifenização,
  preservando diacríticos para reduzir falsos positivos;
- toda evidência precisa produzir um relatório com chunk, grau e cobertura;
- evidência não localizada deve interromper a preparação, não virar qrel vazio;
- matches parciais ou ambíguos devem voltar para revisão;
- comparar a saída automática com uma amostra rotulada manualmente;
- reportar precisão/recall/F1 do mapeamento e concordância humana;
- usar dois anotadores em toda a amostra, se viável, ou em uma subamostra
  estratificada, reportando Cohen kappa ou Krippendorff alpha.

O núcleo executável dessa etapa fica em `benchmark/groundtruth/`.

## 5. Estudo 1 — SciQ controlado

### 5.1 Papel no artigo

O SciQ fornece comparabilidade e uma primeira verificação da implementação. O corpus
é formado somente por `support`; `question` é a consulta; e `question -> support` é o
qrel. Perguntas, respostas e distratores não são indexados.

Após o filtro atual de supports com menos de oito palavras, há 12.135 supports únicos,
12.146 consultas e 878 consultas no teste. O dataset é inglês, de ciências e apresenta
forte correspondência lexical; essas características limitam a validade externa.

### 5.2 Protocolo

1. Usar `validation` para escolher `k`, chunking e qualquer parâmetro de fusão.
2. Registrar a decisão antes de abrir os resultados finais.
3. Recriar uma coleção limpa para cada condição que altere chunking ou modelo.
4. Executar BM25, denso e híbrido nas mesmas 878 consultas de teste.
5. Não ajustar parâmetros depois de observar o teste.
6. Registrar resultados por consulta, inclusive quando nenhum documento for retornado.

Como há exatamente um qrel binário por consulta:

- Hit@k é igual a Recall@k;
- Precision@k é Hit@k dividido por `k`;
- AP@k é igual ao reciprocal rank truncado, logo MAP@k é igual a MRR@k.

Assim, definir **MRR@10 como métrica primária** e Hit@1, Hit@5 e Hit@10 como
secundárias. nDCG pode ser reportada, mas não representa relevância graduada nesse
dataset. Reportar também latência mediana e p95; a média isolada é sensível a outliers.

### 5.3 Rodada preliminar canônica

A tabela e as figuras atuais devem usar a mesma origem. A rodada timestampada
`20260707-183922` foi escolhida como registro preliminar porque as figuras existentes
correspondem a ela. O manifesto rastreável está em
`benchmark/runs/antigos/sciq/20260707-183922/manifest.json`.

Essa rodada, arquivada em `benchmark/runs/antigos/sciq/20260707-183922`, ainda não
é a rodada final: faltam revisões exatas dos modelos, versões do ambiente e hardware.
Ela pode sustentar uma seção de resultados preliminares, mas deve ser repetida após o
protocolo e o manifesto estarem congelados.

## 6. Estudo 2 — corpus educacional em PDF

### 6.1 Corpus

O PDF `importancia_ato_ler.pdf` pode ser usado como piloto técnico local. Um artigo que
pretenda generalizar para corpora educacionais deve incluir mais de um documento e,
preferencialmente, materiais com licença aberta que possam ser redistribuídos com o
benchmark. Registrar por documento: título, edição, idioma, licença, SHA-256, páginas,
forma de extração e taxa de erro/OCR.

### 6.2 Amostra de consultas

- conduzir primeiro um piloto para estimar variância e esforço de anotação;
- fazer análise de poder ou justificar o tamanho final;
- como meta prática, buscar pelo menos 50 consultas válidas e preferir 80–100;
- separar desenvolvimento e teste antes do ajuste dos recuperadores;
- estratificar por categoria, dificuldade, necessidade de uma/múltiplas evidências e
  presença/ausência de termos literais do documento;
- manter consultas não respondíveis em um conjunto separado, avaliado por cobertura e
  abstenção, não misturado automaticamente às métricas clássicas de ranking.

### 6.3 Variáveis

A comparação principal mantém o chunking fixo e varia apenas o recuperador. A
comparação de chunking é uma ablação separada; cruzar todos os chunkers e todos os
recuperadores sem amostra suficiente aumenta o número de comparações e dificulta a
interpretação.

Condições principais:

- C1: BM25;
- C2: denso, com modelo e revisão registrados;
- C3: híbrido RRF, registrando profundidade dos candidatos e parâmetros do Qdrant.

Um reranker só entra como C4 depois de ter identidade, versão, entrada, profundidade de
candidatos e saída persistidas no trace.

## 7. Estudo 3 — RAG exploratório

Reutilizar exatamente os mesmos `case_id`, perguntas, respostas e evidências aprovadas
do Estudo 2. O runner deve executar uma matriz de condições, em vez de uma única
configuração por arquivo:

- sem contexto, para estimar conhecimento paramétrico;
- contexto-oráculo, formado pelas evidências gold;
- BM25;
- denso;
- híbrido;
- reranqueado somente se C4 tiver sido validada na camada de RI.

Fixar e registrar: modelo gerador, revisão/data, prompt ou hash do prompt, idioma,
temperatura, seed quando suportada, máximo de tokens, política de contexto e número de
repetições. O SciQ não deve ser enviado ao prompt português atual sem controlar a
mudança de idioma.

Para o artigo, coletar respostas sem RAGAS inline e avaliar depois, em lote. Preservar
três listas distintas:

1. candidatos recuperados antes do reranqueamento;
2. ranking final depois do reranqueamento;
3. chunks efetivamente enviados ao gerador.

`Faithfulness` deve usar a terceira lista. Métricas de recuperação/contexto usam a
lista correspondente à condição avaliada. RAGAS é uma medida baseada em LLM e deve
ser complementada por avaliação humana em amostra, com juiz diferente do gerador
sempre que possível.

## 8. Inferência estatística

Todos os métodos respondem às mesmas consultas; portanto, as comparações são pareadas.

- reportar média ou mediana, IC95% e distribuição por consulta;
- usar bootstrap pareado para diferenças de MRR/nDCG e latência;
- usar McNemar para Hit@k entre dois métodos;
- usar teste de randomização pareado ou Wilcoxon para métricas contínuas por consulta,
  registrando a escolha antes da análise;
- reportar tamanho de efeito, não apenas valor de p;
- corrigir múltiplos contrastes, por exemplo pelo método de Holm;
- usar Spearman ou Kendall para associação RI/RAG, com IC por bootstrap;
- se muitas consultas vierem do mesmo documento/seção, considerar bootstrap por
  cluster para não tratá-las como completamente independentes.

Uma diferença numérica pequena, como híbrido versus BM25 em Hit@10, só deve ser
descrita como ganho após intervalo de confiança e teste pareado.

## 9. Manifesto mínimo de reprodutibilidade

Cada rodada precisa registrar:

- `study_id`, `run_id`, data UTC e status (`pilot`, `validation`, `test`);
- commit Git e indicação de árvore modificada;
- hash dos casos, qrels e corpus;
- dataset, revisão, splits e filtros;
- PDF hashes, licenças e configurações de extração/OCR;
- estratégia e parâmetros de chunking;
- modelos dense, sparse, reranker, gerador e juiz, com versões/revisões;
- versões Python, Qdrant, FastEmbed, Docling, RAGAS e dependências relevantes;
- parâmetros de busca, `k`, pool de candidatos, RRF e limiares de relevância;
- prompt ou hash, temperatura, máximo de tokens e política de contexto;
- hardware e dispositivo;
- sementes e número de repetições;
- tempos separados de embedding, busca, reranking, geração e avaliação;
- contagem de consultas válidas, erros e valores ausentes por métrica.

Não registrar segredos, tokens ou valores de `.env`.

## 10. Sequência de execução e critérios de saída

### Estado do código em 11 de julho de 2026

- [x] orquestrador único com rodadas e coleções timestampadas;
- [x] correção da avaliação de subconjuntos do SciQ;
- [x] manifesto preliminar da baseline histórica;
- [x] materializador estrito de evidências para qrels e casos RAGAS;
- [x] runner PDF/RI com relevância graduada e falha explícita;
- [x] IC95% bootstrap, diferenças pareadas, McNemar e Holm automatizados;
- [x] 32 casos de desenvolvimento, com 37 evidências auditadas no PDF piloto;
- [ ] rodada SciQ final com revisões congeladas;
- [ ] conjunto gold de PDF aprovado por anotadores;
- [ ] rodada acadêmica real de PDF/RI e, depois, RAG.

Os itens marcados abaixo descrevem resultados experimentais ainda necessários, não
ausência de implementação da pipeline.

### Fase A — corrigir e congelar o SciQ

- [x] corrigir avaliação de `--limit-queries`;
- [ ] escolher configuração na validação;
- [ ] gerar uma rodada final com manifesto completo;
- [ ] gerar tabela e figuras exclusivamente dessa rodada;
- [ ] executar testes pareados e IC95%.

**Saída:** RQ1 respondida no benchmark controlado.

### Fase B — construir ground truth de PDF

- [ ] versionar corpus por hash/licença;
- [ ] aprovar guideline 0/1/2 em piloto;
- [ ] criar casos mestres e evidências;
- [ ] materializar qrels por chunking e revisar casos ambíguos;
- [ ] medir concordância entre anotadores e automação;
- [ ] congelar desenvolvimento/teste.

**Saída:** conjunto gold reutilizável e RQ3 respondida.

### Fase C — executar RI no PDF

- [ ] fixar chunking principal;
- [ ] executar C1–C3 de forma pareada;
- [ ] fazer ablação de chunking separada;
- [ ] analisar por tipo de pergunta e erros;
- [ ] calcular inferência estatística.

**Saída:** RQ1 e RQ2 respondidas no cenário ecológico.

### Fase D — RAG opcional

- [ ] preservar trace dos três níveis de contexto;
- [ ] executar sem contexto, oráculo e C1–C3;
- [ ] avaliar RAGAS offline e amostra humana;
- [ ] juntar RI e RAG por `case_id + condition_id`;
- [ ] calcular associação e analisar discordâncias.

**Saída:** evidência exploratória para RQ4-E.

## 11. Limpeza segura do repositório

- remover notas/transcrições que não participam da execução ou documentação;
- remover funções privadas comprovadamente sem chamadas;
- não apagar rodadas SciQ divergentes antes de congelar a que sustenta o artigo;
- não versionar `.venv`, caches, modelos ou saídas regeneráveis;
- não tratar o PDF piloto como artefato redistribuível sem verificar sua licença;
- manter exemplos e testes dos benchmarks, pois documentam contratos públicos.

## 12. Ajustes necessários no artigo

1. Distinguir explicitamente os três estudos dos três recuperadores.
2. Apresentar RI como contribuição principal e RAG como extensão exploratória.
3. Não chamar a heurística atual de modelo reranqueador.
4. Descrever evidências canônicas e materialização automática de qrels.
5. Declarar a redundância das métricas no SciQ com um qrel por consulta.
6. Separar método executado, resultado preliminar e trabalho futuro.
7. Não afirmar que prompt, versões e reranking são persistidos até isso existir.
8. Incluir protocolo de anotação, concordância, IC95%, testes pareados, tamanhos de
   efeito, correção múltipla e ameaças de licença/idioma/pooling.
9. Se o sistema continuar sem modelo do estudante e política pedagógica adaptativa,
   descrevê-lo como QA/RAG educacional e tratar a aplicação em ITS como motivação, não
   como funcionalidade já validada.

## 13. Referências metodológicas essenciais

- Welbl, Liu e Gardner (2017), SciQ, DOI `10.18653/v1/W17-4413`.
- Manning, Raghavan e Schütze (2008), *Introduction to Information Retrieval*.
- TREC/NIST, julgamentos de relevância e pooling:
  <https://trec.nist.gov/data/reljudge_eng.html>.
- Es et al. (2023), RAGAS, arXiv `2309.15217`.
- Salemi e Zamani (2024), avaliação de recuperação em RAG, arXiv `2404.13781`.

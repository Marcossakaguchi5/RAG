# Guia de experimentação e avaliação

Este documento explica como os experimentos do repositório são desenhados,
executados e interpretados. Ele é a referência conceitual; os comandos completos
estão em [EXECUCAO_COMPLETA.md](EXECUCAO_COMPLETA.md) e os detalhes metodológicos
e decisões de escopo estão em [PLANO_EXPERIMENTAL.md](PLANO_EXPERIMENTAL.md).

## 1. Objetivo e unidades de avaliação

O objeto principal de estudo é a **recuperação da informação (RI)**: dada uma
consulta, o sistema deve ordenar os chunks que contêm evidência para respondê-la.
O RAG é uma extensão: ele usa os chunks recuperados para gerar uma resposta, mas
não substitui a avaliação da recuperação.

As unidades importantes são:

- **corpus**: conjunto de documentos indexados;
- **documento**: PDF ou texto-fonte do corpus;
- **chunk**: unidade textual indexada e retornada pela busca;
- **consulta/caso**: pergunta identificada por `case_id` ou `query_id`;
- **qrel** (*query relevance judgment*): vínculo entre uma consulta e um chunk
  relevante, eventualmente com grau de relevância;
- **run**: ranking produzido por um método para todas as consultas de um split;
- **condição**: combinação controlada de corpus, chunking, recuperador e parâmetros;
- **rodada**: execução completa, imutável e rastreável de uma ou mais condições.

Em termos simples, cada pergunta possui uma verdade de referência e cada método
produz uma lista ordenada de chunks. As métricas comparam as duas coisas.

```text
caso mestre (pergunta + resposta + evidência)
                  |
                  v
chunks da condição avaliada ---> qrels materializados
                  |                     |
                  +----------+----------+
                             v
                 ranking por método e consulta
                             |
                             v
             métricas por consulta -> agregados -> estatística
                             |
                             v
                RAG opcional com os mesmos casos aprovados
```

## 2. Estudos realizados

| Estudo | Papel | Dados e qrels | Comparação principal |
| --- | --- | --- | --- |
| **SciQ** | baseline controlada e verificação da implementação | `support` como corpus; `question -> support` como qrel binário | BM25, denso e híbrido nas mesmas consultas do split |
| **PDF/RI** | avaliação ecológica no corpus educacional | casos mestres com citações do PDF, materializadas para os chunks de cada chunking | BM25, denso e híbrido com chunking fixo |
| **Ablação de chunking** | isolar o efeito da segmentação | os mesmos casos gold e qrels rematerializados | um recuperador fixo, variando somente o chunking |
| **RAG/RAGAS** | extensão exploratória de geração | mesmos `case_id`, perguntas e respostas do PDF/RI | método de recuperação como condição de contexto |
| **QASPER** | benchmark adicional de RI em artigos longos | parágrafos textuais anotados como evidência | pipeline própria em `apps/ingest/api/benchmarks/qasper/` |

Não se deve confundir os **três recuperadores** (BM25, denso e híbrido) com os
**estudos**. Um reranker só é uma condição experimental quando sua identidade,
versão, candidatos de entrada, ranking de saída e métricas forem persistidos.

### 2.1 SciQ

SciQ é um cenário de controle em inglês e de domínio científico. Apenas os textos
`support` são indexados; perguntas, respostas e distratores não entram no índice.
Há um qrel binário por pergunta. Por isso, no SciQ:

- `Hit@K = Recall@K`;
- `Precision@K = Hit@K / K`;
- `AP@K = RR@K` e, portanto, `MAP@K = MRR@K` na implementação;
- `MRR@10` é a métrica principal mais informativa para a posição da primeira
  evidência, com Hit@1, Hit@5 e Hit@10 como complementares.

O split `validation` serve para escolher parâmetros. O split `test` só deve ser
usado depois de congelar essas escolhas. Ajustar a configuração após consultar o
teste transforma o teste em desenvolvimento e superestima o resultado.

#### Pipeline executada no SciQ

```text
dataset allenai/sciq em uma revisão fixa
  -> filtrar supports com menos de 8 palavras e normalizar espaços
  -> corpus: supports únicos; consulta: question; qrel: support correto
  -> chunking textual e indexação dense+sparse no Qdrant
  -> BM25, denso e híbrido nas mesmas perguntas de teste
  -> ranking de documentos, métricas por K e por consulta
  -> IC95% bootstrap, contrastes pareados e gráficos
```

1. **Preparação:** `prepare_sciq.py` baixa `allenai/sciq` na revisão informada no
   manifesto. Cada `support` é normalizado e descartado se tiver menos de oito
   palavras. Supports idênticos são deduplicados; seu `doc_id` é determinístico,
   derivado do MD5 do texto. Para cada exemplo válido, a `question` vira uma
   consulta e o `support` correspondente vira seu único qrel, com `relevance=1`.
   `correct_answer` e os distratores são preservados no arquivo de consultas, mas
   **não são indexados** nem usados para pontuar RI. O embaralhamento das opções
   usa a seed, mas não afeta a recuperação.
2. **Artefatos preparados:** são gravados `corpus.jsonl` (supports únicos),
   `queries.jsonl` (pergunta, split e resposta) e `qrels.jsonl`
   (`query_id -> doc_id`). Na configuração usada nas rodadas atuais, há 12.135
   supports únicos, 12.146 consultas e 878 consultas no split de teste após o
   filtro.
3. **Indexação:** `ingest_corpus.py` aplica `recursive_text` ou `fixed_token` ao
   texto de cada support. Cada chunk recebe ID determinístico
   `sciq_doc_..._chunk_0000` e ponto Qdrant UUIDv5 também determinístico. O texto
   recebe embedding denso BGE-M3 e vetor sparse `Qdrant/bm25`, configurado para
   inglês; a coleção é recriada para uma rodada limpa quando solicitado.
4. **Busca:** `run_retrieval.py` carrega os modelos uma vez por método e consulta
   as mesmas perguntas do split. Ele busca até `4 × top_k` chunks, remove chunks
   repetidos do mesmo `document_id` e conserva os `top_k` documentos distintos.
   Isso é importante: a relevância do SciQ é do **support/documento**, não de um
   chunk específico. A latência por consulta envolve codificação da consulta e
   busca, mas não o carregamento inicial dos modelos.
5. **Métodos:** BM25 consulta o vetor sparse; denso consulta o embedding por cosseno;
   híbrido pré-busca nas duas representações e funde os candidatos por RRF. Todos
   usam a mesma coleção, o mesmo conjunto de consultas e o mesmo `top_k`.
6. **Avaliação:** `evaluate_retrieval.py` compara os `doc_id` do ranking com o qrel
   binário. Toda consulta esperada entra no denominador, inclusive se não houver
   resultado salvo para ela; nesse caso suas métricas são zero. O runner calcula
   Hit/Precision/Recall/MAP/nDCG/MRR para K=1, 3, 5 e 10 por padrão.
7. **Estatística e saída:** `analyze_sciq.py` reconstrói métricas por consulta,
   gera IC95% bootstrap e diferenças pareadas entre métodos; McNemar exato/Holm é
   aplicado a Hit@K. A rodada preserva rankings em `retrieval/`, agregados em
   `results/`, estatística em `statistics/`, gráficos/relatório em `plots/` e todos
   os parâmetros, hashes e ambiente em `manifest.json`.

O ponto de entrada acadêmico é `benchmark/run_experiment.py sciq`. Ele executa
o pipeline interno, cria uma pasta nova em `benchmark/runs/novas/sciq/<run-id>/`
e registra revisão do dataset, modelo, chunking, seed, commits e versões. O
comando de baixo nível `apps/ingest/api/benchmarks/sciq/run_all.py` continua útil para
diagnóstico, mas não cria por si só o manifesto acadêmico completo.

### 2.2 PDF/RI

O experimento em PDF avalia o cenário do sistema. A anotação não deve depender de
UUIDs do Qdrant, pois eles mudam ao reingerir ou trocar o chunking. A fonte de
verdade é o **caso mestre**, que contém pergunta, resposta de referência e uma ou
mais citações canônicas do documento.

Exemplo reduzido:

```json
{
  "id": "freire-q001",
  "split": "test",
  "category": "conceitual",
  "query": "Pergunta avaliada",
  "reference_answer": "Resposta sustentada pelo texto.",
  "evidence": [{
    "document_name": "documento.pdf",
    "document_sha256": "hash-opcional-mas-recomendado",
    "page": 12,
    "quote": "Citação literal curta.",
    "relevance": 2
  }],
  "provenance": {"review_status": "approved", "annotators": ["A1"]}
}
```

Os graus usados são `0` (não relevante), `1` (evidência parcial/complementar) e
`2` (evidência suficiente). As métricas binárias consideram graus positivos como
relevantes; `nDCG` preserva os graus. Uma análise de sensibilidade pode considerar
somente o grau 2 como relevante, desde que seja declarada no protocolo.

Os arquivos `*.draft.jsonl` são **silver/piloto**. Só casos com
`review_status` igual a `approved` ou `adjudicated` são aceitos normalmente pelo
orquestrador; `--allow-draft-cases` é uma exceção explícita para testes técnicos.

### 2.3 RAG como extensão

O RAG não cria outro conjunto de perguntas. Ele reutiliza a projeção RAGAS do caso
mestre já aprovado. Para cada pergunta, é importante manter separadas estas listas:

1. candidatos antes do reranqueamento;
2. ranking final após o reranqueamento;
3. chunks efetivamente enviados ao gerador.

`Faithfulness` deve usar a lista 3, pois ela representa o contexto que realmente
fundamentou a resposta. Métricas de contexto/recuperação devem usar a lista da
condição que está sendo medida. Uma resposta boa não prova que a recuperação foi boa:
o modelo pode responder por conhecimento paramétrico; por isso as condições sem
contexto e com contexto-oráculo são controles recomendados.

## 3. Recuperadores e parâmetros controlados

- **BM25/sparse**: representa consulta e chunks como vetores esparsos e prioriza
  coincidência lexical ponderada. A implementação usa `Qdrant/bm25`; o idioma do
  analisador é registrado (`english` no SciQ e normalmente `portuguese` no PDF).
- **Denso**: codifica consulta e chunks em vetores densos e recupera por
  similaridade de cosseno. O modelo e sua revisão devem constar no manifesto.
- **Híbrido**: busca candidatos nas duas representações e os funde por
  **Reciprocal Rank Fusion (RRF)**. Conceitualmente,
  `RRF(d) = soma_m 1 / (c + rank_m(d))`; o valor de `c` é controlado pela fusão do
  Qdrant e precisa ser registrado se for configurável. No código atual, cada lista
  de candidatos tem limite `min(4 × top_k, 200)`.

Para uma comparação justa, a condição principal mantém corpus, casos, split,
chunking, `top_k`, modelos e parâmetros constantes e altera apenas o recuperador.
A comparação de chunking é uma ablação separada, pois alterar recuperador e
segmentação simultaneamente impede atribuir a causa de uma diferença.

### 3.1 Reranker atual do `chat`: heurística, não modelo treinado

O `chat` possui uma etapa chamada `rerank`, mas ela **não** usa uma biblioteca de
reranking neural nem um *cross-encoder*. A função em
`apps/chat/api/app/services/reranker.py` usa somente a biblioteca padrão do Python:
`math`, `re` e `unicodedata`. O `LangGraph` apenas orquestra a sequência
recuperar → reranquear → gerar; ele também não calcula a relevância.

O fluxo é:

1. O `ingest` recupera `candidate_k` chunks pelo método escolhido. A API usa
   `max(top_k, candidate_k)`; os valores padrão são `candidate_k=20` e `top_k=5`.
2. A heurística normaliza pergunta e chunks (minúsculas, NFKD, remoção de acentos),
   extrai tokens com a expressão `[a-z0-9]{2,}` e remove uma lista fixa de
   *stopwords* em português.
3. Para cada candidato, calcula `retrieval_score`, normalizando o escore devolvido
   pela recuperação, e `lexical_score`, que combina sobreposição de termos únicos,
   frequência logarítmica dos termos da pergunta e bônus de frase de `0,12`.
4. Com reranking ativo, calcula
   `score_final = 0,62 × retrieval_score + 0,38 × lexical_score`, ordena de forma
   decrescente e conserva os `top_k` primeiros. Empates preservam a posição do
   ranking original.

Com `use_reranker=false`, não há reordenação: os primeiros `top_k` da recuperação
seguem para a geração. O trace da resposta guarda `retrieval_rank` e, quando ativo,
`rerank_score`, permitindo distinguir a posição antes e depois da heurística.

Consequência metodológica: esta condição pode ser chamada de **combinação heurística
de escore de recuperação e sobreposição lexical**, não de “reranker treinado” ou
“reranker semântico”. Ela não deve ser apresentada como resultado principal de RI
até que seja incluída no mesmo benchmark pareado de BM25, denso e híbrido, com seus
candidatos e rankings persistidos.

### 3.2 Extração do PDF e estratégias de chunking

**Extração** transforma o PDF em texto/estrutura; **chunking** divide esse conteúdo
em unidades que receberão `chunk_id`, embedding e índice. As opções abaixo não
mudam apenas o tamanho do chunk: algumas também mudam o extrator e o texto que será
indexado. Por isso, cada estratégia requer coleção própria, nova exportação de
`chunks.jsonl` e nova materialização dos qrels.

#### Como o texto do PDF é obtido

| Caminho | Quando é usado | Biblioteca/ferramenta | Característica |
| --- | --- | --- | --- |
| `pdftotext -raw` | `fixed_token` e `recursive_text` | Poppler (`pdftotext`) | produz uma leitura linear e estável para as baselines; não executa OCR |
| Docling sem OCR | estratégias `docling_*` | `docling.DocumentConverter` | tenta preservar a estrutura detectada do PDF (por exemplo, blocos e hierarquia) |
| Docling com OCR | quando a extração estrutural inicial tem menos de 80 caracteres normalizados e OCR está habilitado | Docling + EasyOCR | atende PDFs digitalizados/imagem; idiomas configurados: `pt,en` |

Nas estratégias estruturais, a API tenta primeiro Docling sem OCR. Se o texto não
for suficiente, tenta novamente com OCR; o OCR completo de página só é forçado se
`DOCLING_OCR_FORCE_FULL_PAGE=true` (o padrão é `false`). Nas baselines, a escolha
intencional por `pdftotext -raw` evita que a exportação Markdown do Docling altere a
ordem linear de leitura. Se o PDF for somente imagem, essas duas baselines falham;
as estratégias Docling com OCR são as adequadas.

#### Estratégias disponíveis

| Estratégia | Como forma os chunks | Biblioteca principal | Quando faz sentido |
| --- | --- | --- | --- |
| `fixed_token` | janela fixa de 512 tokens, com sobreposição de 64 por padrão | tokenizer Hugging Face do modelo de embedding; fallback por palavras | baseline simples e controlada para medir o efeito de janelas uniformes |
| `recursive_text` | agrega parágrafos; se necessário divide recursivamente por quebras, frases, ponto e vírgula, vírgulas e palavras; até 700 palavras com 100 de overlap por padrão | lógica própria + `pdftotext` | baseline textual geralmente mais legível, preservando fronteiras linguísticas quando possível |
| `docling_hierarchical` | usa blocos e hierarquia identificados no `DoclingDocument` | `docling` `HierarchicalChunker` | PDFs bem estruturados, quando títulos/seções/blocos ajudam a manter o sentido |
| `docling_hybrid` | parte da estrutura do Docling e aplica refinamento orientado ao limite de tokens, com `merge_peers=True` | `docling` `HybridChunker` e, se disponível, tokenizer Hugging Face | equilíbrio entre coerência estrutural e tamanho compatível com o embedding |
| `docling_hybrid_contextual` | é o `HybridChunker`, mas indexa `contextualize(chunk)` em vez do texto cru | `HybridChunker.contextualize()` | quando incluir contexto estrutural, como cabeçalhos, pode desambiguar chunks curtos |
| `docling_hybrid_parent_child` | obtém pais híbridos e divide o texto de cada pai em filhos de 320 palavras, com overlap 60; prefixo estrutural do pai é anexado aos filhos | `HybridChunker` + janelas por palavras da aplicação | busca granular com contexto de seção; os filhos, não os pais, são os itens indexados |

Em `fixed_token`, se o tokenizer não puder ser carregado, o fallback usa janelas de
**palavras** com os números configurados como tamanho/overlap. Em `recursive_text`,
o `page_number` é uma estimativa baseada na posição do chunk; nos métodos Docling,
a API tenta usar a proveniência de página do bloco estrutural e só estima quando ela
não está disponível. Portanto, página de baseline é metadado de inspeção, não prova
de localização exata; a auditoria do PDF é a fonte para validar citações.

Para o experimento principal, fixe uma estratégia — atualmente `recursive_text` é a
baseline padrão — e compare BM25, denso e híbrido. Para estudar a segmentação, fixe
o recuperador e execute uma ablação pareada entre estratégias. Não compare métricas
de qrels criados para uma coleção com ranking de outra: cada chunking gera IDs e
limites textuais diferentes.

## 4. Ground truth: da evidência ao qrel

O fluxo em `benchmark/groundtruth/` separa julgamento científico e associação
técnica:

1. Definir a citação, a resposta e o grau de relevância por anotação humana.
2. Congelar o PDF e registrar nome, licença e SHA-256.
3. Auditar se a citação ocorre no PDF (`audit_pdf.py`).
4. Indexar o documento com a estratégia de chunking da condição.
5. Exportar os chunks efetivamente indexados.
6. Materializar cada citação para os IDs de chunks (`materialize.py`).
7. Inspecionar o relatório e só então executar RI.

O materializador normaliza Unicode (NFKC), caixa, espaços, pontuação e hifenização
em quebra de linha, preservando diacríticos. Por padrão exige cobertura integral
(`--min-coverage 1.0`) e falha se uma evidência não for coberta; normalmente a
citação precisa ocorrer de forma contígua em um chunk. Como exceção controlada para
chunkers estruturais, até três chunks ordenados da mesma página podem reconstruir a
citação exatamente. Isso evita qrels silenciosamente incompletos. Ele gera:

- `ingest-groundtruth.jsonl`: qrels binários e `relevance_by_chunk`;
- `ragas-groundtruth.jsonl`: pergunta e resposta de referência para a etapa RAG;
- `matching-report.json`: cobertura, candidatos, chunks associados e hashes.

O casamento automático responde “qual chunk contém esta citação?”, não “a citação
é uma evidência semanticamente adequada?”. A segunda pergunta continua sendo uma
decisão de anotação e deve passar por revisão humana. Para perguntas humanas sem
evidência previamente definida, use *pooling*: una resultados de métodos distintos,
oculte método/rank/escore e julgue o pool embaralhado. Rotular só o resultado de um
método cria viés circular.

### 4.1 O que é comparado de fato

O mapeamento ocorre em duas validações diferentes e uma comparação final por ID:

```text
gold.jsonl: citação + página + hash do PDF
        |  auditoria do PDF-fonte
        v
citação existe na página declarada?
        |  materialização após o chunking
        v
citação -> IDs dos chunks que a contêm (qrels)
        |  execução da busca
        v
IDs recuperados no ranking versus IDs relevantes do qrel
```

1. **PDF-fonte:** `audit_pdf.py` extrai texto por página e verifica se a citação
   normalizada está na página declarada do PDF de mesmo nome/hash. Esta é uma
   verificação literal da âncora documental; ela não julga a qualidade da resposta.
2. **Chunks indexados:** após a ingestão, `materialize.py` recebe
   `chunks.jsonl`, em que cada chunk já possui um `chunk_id` único (na prática,
   normalmente um UUID), conteúdo, documento e página. Ele filtra primeiro pelo
   nome/hash do documento e procura a sequência de tokens normalizada da citação no
   conteúdo de cada candidato. Não é um `regex` aplicado ao resultado da busca:
   o algoritmo usa normalização e `SequenceMatcher` para encontrar a maior sequência
   contígua de tokens; com cobertura `1.0`, a citação inteira deve estar presente.
3. **Métrica de RI:** o recuperador devolve uma lista ordenada de `chunk_id`. O
   runner compara diretamente essa lista com `relevant_chunk_ids` e
   `relevance_by_chunk` produzidos pelo materializador. Assim, por exemplo,
   `Recall@10` e MRR não relêem o PDF nem fazem casamento textual durante a busca.

O campo `reference_answer` do `gold.jsonl` é a resposta de referência para a
avaliação RAG e para a revisão humana. O código valida o esquema e mapeia a
**citação**, mas não possui um verificador automático que prove semanticamente que a
resposta está correta. Essa garantia vem da anotação/revisão registrada em
`provenance` (anotador, status `approved`/`adjudicated`, notas) e deve ser auditada
por humanos. Em outras palavras: a automação garante a rastreabilidade
`citação -> chunk_id`; a validade da pergunta e da resposta é responsabilidade do
processo de anotação gold.

## 5. Cálculo das métricas de RI

Considere uma consulta `q`, o ranking `d_1, ..., d_K`, o conjunto de chunks
relevantes `R_q`, `rel_i = 1` quando `d_i` pertence a `R_q` e `g_i` como seu grau
de relevância (0, 1 ou 2). Todas as métricas são calculadas **por consulta** e depois
agregadas pela média das consultas válidas.

| Métrica | Cálculo | Leitura |
| --- | --- | --- |
| Hit@K | `1` se `soma rel_i > 0`; senão `0` | há alguma evidência no topo K? |
| Precision@K | `(soma rel_i) / K` | qual fração do topo K é relevante? |
| Recall@K | `(soma rel_i) / |R_q|` | qual fração de todas as evidências foi encontrada? |
| RR | `1 / r`, em que `r` é o primeiro rank relevante; `0` se não houver | quão cedo aparece a primeira evidência? |
| MRR | média de RR entre consultas | qualidade média da primeira evidência |
| AP@K | `(1 / |R_q|) × soma_{i=1..K}(Precision@i × rel_i)` | premia relevantes encontrados cedo e múltiplos relevantes |
| MAP | média de AP@K | qualidade global do ranking em várias consultas |
| DCG@K | `soma_{i=1..K} (2^g_i - 1) / log2(i + 1)` | ganho acumulado, descontando posições tardias |
| nDCG@K | `DCG@K / IDCG@K` | DCG normalizado pelo ranking ideal; preserva relevância graduada |

`IDCG@K` é o DCG que seria obtido ao ordenar os graus relevantes do maior para o
menor. Assim, nDCG vale de 0 a 1. Na implementação atual, AP divide por `|R_q|`,
mesmo com ranking truncado em K; portanto o campo agregado é chamado `map`, e deve
ser lido como AP/MAP calculado no ranking retornado até K.

Exemplo: se os relevantes de uma pergunta aparecem nas posições 2 e 5 de um
ranking de tamanho 10, então `Hit@10 = 1`, `Precision@10 = 2/10`,
`Recall@10 = 2/|R_q|`, `RR = 1/2` e
`AP@10 = (Precision@2 + Precision@5) / |R_q|`.

### 5.1 Métricas na interface do `ingest`: corretas, mas para inspeção manual

A tela do `ingest` envia a pergunta, o método, `top_k` e a lista de
`relevant_chunk_ids` para `POST /api/search`. O backend compara os IDs retornados
com essa lista e calcula Precision@K, Recall@K, AP, nDCG binário e RR. Portanto, a
**fórmula é correta** para uma consulta quando os IDs relevantes já foram julgados
de forma independente e a lista está completa.

Há três limites que impedem tratar o número mostrado na tela como resultado
acadêmico por si só:

1. A tela calcula uma só consulta. O campo mostrado como `MAP` é, tecnicamente, o
   **AP** daquela pergunta; MAP só existe depois de calcular AP para várias
   perguntas e tirar a média. O `MRR` mostrado também é o RR da consulta individual.
2. A interface usa somente relevância binária: um ID está ou não está na lista. Ela
   não recebe `relevance_by_chunk`, logo não calcula nDCG graduado com os graus 1 e
   2 do conjunto gold.
3. Marcar como relevante apenas os chunks que a própria busca exibiu é útil para
   explorar o sistema, mas é inadequado para avaliar esse mesmo método: o qrel fica
   incompleto e dependente do ranking observado. Isso pode ocultar evidências que o
   método não recuperou e inflar Recall/MAP. IDs gold que não aparecem no topo K
   precisam continuar na lista de relevantes para que a falha seja contabilizada.

Assim, use a interface para verificar uma pergunta, inspecionar chunks e conferir se
o backend faz o cálculo esperado. Para tabela, gráfico ou conclusão do experimento,
use `benchmark/run_experiment.py pdf-ir` ou
`apps/ingest/api/benchmarks/groundtruth/run_groundtruth.py`: eles recebem qrels
materializados antes da busca, executam todos os casos e métodos, preservam ranking e
latência por consulta e produzem médias, IC95% e comparações pareadas. A interface
não substitui essa rodada, seu manifesto ou a revisão humana do `gold.jsonl`.

### Latência

O runner mede o tempo de parede da chamada de busca por caso/método em
milissegundos. O resumo reporta:

- **mediana**: latência típica, menos sensível a extremos;
- **p95**: percentil 95 empírico, calculado na posição `ceil(0,95 × n) - 1` da
  lista ordenada.

Compare latência apenas em ambiente e carga comparáveis. Tempo de indexação,
embedding, busca, reranking e geração são componentes diferentes e devem ser
registrados separadamente quando fizerem parte da conclusão.

## 6. Estatística e comparação entre métodos

Os métodos respondem às **mesmas consultas**; portanto, a análise é pareada. Não é
apropriado tratar as médias de BM25 e denso como amostras independentes.

O código produz os seguintes controles para SciQ e PDF/RI:

- **IC95% bootstrap da média**: reamostra com reposição os valores por consulta
  (2.000 repetições por padrão), calcula a média em cada reamostra e usa os
  percentis 2,5% e 97,5% como intervalo.
- **Diferença pareada**: para cada consulta calcula
  `métrica(método_esquerdo) - métrica(método_direito)`, e aplica bootstrap a essa
  lista de diferenças. O resultado traz média, IC95%, vitórias, empates e derrotas.
- **McNemar exato**: aplicado a Hit@K, que é binário por consulta. Ele usa apenas
  os casos discordantes: esquerda acerta/direita erra versus o inverso.
- **Holm**: ajusta os valores de p de McNemar para os múltiplos contrastes dentro
  da estratégia (ou do mesmo K no SciQ).

Uma diferença pequena não deve ser apresentada como ganho comprovado apenas porque
as médias diferem. Verifique magnitude, IC95%, quantidade de vitórias/derrotas,
p ajustado e relevância prática (por exemplo, latência adicional). Para inferências
adicionais planejadas, o plano recomenda randomização pareada ou Wilcoxon para
métricas contínuas e Spearman/Kendall para associação entre RI e RAG.

## 7. Estrutura de `benchmark/`

| Caminho | Responsabilidade |
| --- | --- |
| `benchmark/run_experiment.py` | ponto de entrada para rodadas `sciq`, `pdf-ir` e `rag`; cria diretório e manifesto |
| `benchmark/analyze_sciq.py` | IC95% e contrastes pareados da rodada SciQ |
| `benchmark/plot_pdf_ir.py` | tabelas, SVGs, ICs, curvas e contrastes do PDF/RI |
| `benchmark/groundtruth/` | auditoria do PDF e materialização estrita de citações para qrels |
| `benchmark/cases/` | casos mestres; `draft` é piloto e `gold` é a versão revisada |
| `benchmark/runs/novas/` | destino de novas rodadas, sem sobrescrever resultados |
| `benchmark/runs/antigos/` | registros históricos congelados |

Uma rodada nova usa `run_id` exclusivo. O orquestrador se recusa a gravar em uma
pasta não vazia, o que protege resultados anteriores. As interfaces web servem para
inspeção e anotação, mas tabelas e figuras acadêmicas devem ser derivadas dos
runners em lote.

### 7.1 Artefatos que uma rodada deve preservar

```text
benchmark/runs/novas/<estudo>/<run-id>/
├── manifest.json                 # configuração, ambiente, hashes e status
├── source-audit.json             # PDF/RI: verificação das citações no PDF
├── <chunking>/
│   ├── chunks.jsonl              # texto realmente indexado
│   ├── ingest-groundtruth.jsonl  # qrels materializados
│   ├── ragas-groundtruth.jsonl   # projeção dos mesmos casos para RAG
│   ├── matching-report.json      # auditoria da materialização
│   └── retrieval/results.jsonl   # ranking, métricas e latência por caso
├── results/ ou retrieval/        # resultados por método (SciQ)
├── statistics/                   # ICs e contrastes pareados (SciQ)
└── plots/                        # CSV, JSON, SVG e relatório HTML
```

O `manifest.json` deve permitir reproduzir e auditar a rodada: commit Git e árvore
modificada, hashes de corpus/casos/artefatos, dataset e split, parâmetros de
chunking, modelos e versões, idioma sparse, `top_k`, seed, ambiente, hardware e
estado final. Segredos, chaves e conteúdo de `.env` nunca devem entrar no manifesto.

## 8. Execução recomendada

1. Definir previamente a pergunta de pesquisa, a métrica principal e os parâmetros
   que serão escolhidos na validação.
2. Preparar o corpus, verificar licença e congelar seus hashes.
3. Construir/revisar casos e evidências sem olhar os resultados dos métodos.
4. Auditar citações e materializar qrels para cada estratégia de chunking.
5. Executar a comparação principal de RI com os mesmos casos para todos os métodos.
6. Gerar ICs, contrastes, gráficos e análise de erros por categoria de pergunta.
7. Executar a ablação de chunking separadamente, se ela fizer parte do estudo.
8. Somente depois, coletar respostas RAG e avaliar RAGAS offline.
9. Relatar limitações: tamanho/amostragem do corpus, idioma, qualidade dos qrels,
   dependência entre consultas de uma mesma seção, custo/variância do juiz LLM e
   diferença entre recuperação e qualidade pedagógica.

Comandos típicos, sempre a partir da raiz, são:

```bash
python benchmark/run_experiment.py sciq --run-id sciq-01 \
  --qdrant-url http://localhost:6335 \
  --methods bm25,dense,hybrid

python benchmark/run_experiment.py pdf-ir --run-id pdf-ri-01 \
  --pdf benchmark/sources/importancia_ato_ler.pdf \
  --cases benchmark/cases/freire_50.gold.jsonl \
  --chunking-strategies recursive_text \
  --methods bm25,dense,hybrid --top-k 10
```

Antes de executar PDF/RI, o serviço `ingest` deve estar ativo e
`INGEST_APP_PASSWORD` configurada. Para RAG, também é necessário o serviço `chat`,
`CHAT_APP_PASSWORD` e, se RAGAS for avaliado, a chave do modelo juiz. Consulte
[EXECUCAO_COMPLETA.md](EXECUCAO_COMPLETA.md) para o roteiro executável completo.

## 9. Como interpretar e relatar resultados

Relate sempre a condição completa, não apenas o nome do método. Exemplo: “híbrido,
`recursive_text`, BGE-M3 na revisão X, BM25 em português, `top_k=10`, 50 casos gold”.

Uma tabela final deve incluir, no mínimo, a métrica primária, métricas secundárias,
IC95%, mediana/p95 de latência, número de consultas e uma indicação dos contrastes
pareados. Preserve também a distribuição por consulta: uma média alta pode esconder
falhas sistemáticas em uma categoria.

Evite as seguintes conclusões inválidas:

- chamar uma rodada `draft/silver` de benchmark gold;
- misturar consultas de desenvolvimento e teste;
- comparar métodos em corpora, qrels ou K diferentes;
- concluir causalidade de recuperação a partir de uma métrica RAG isolada;
- tratar itens não julgados como automaticamente irrelevantes em pooling;
- afirmar que uma heurística é reranker treinado;
- substituir artefatos rastreáveis por números obtidos manualmente na interface.

## 10. Leituras e documentos relacionados

- [PLANO_EXPERIMENTAL.md](PLANO_EXPERIMENTAL.md): questões de pesquisa, escopo e
  ameaças à validade.
- [benchmark/README.md](../benchmark/README.md): visão rápida da pipeline.
- [benchmark/groundtruth/README.md](../benchmark/groundtruth/README.md): formato,
  normalização e falhas do materializador.
- [BENCHMARKS.md](BENCHMARKS.md): comandos e benchmarks disponíveis.
- [apps/chat/api/benchmarks/ragas/README.md](../apps/chat/api/benchmarks/ragas/README.md): coleta
  e métricas RAGAS oficiais.

# Ground truth canônico para PDFs

Este diretório mantém uma única fonte de verdade para dois experimentos:

- **RI/ingest:** pergunta, chunks relevantes e relevância graduada;
- **RAG/RAGAS:** pergunta e resposta de referência.

O materializador usa apenas a biblioteca padrão do Python. Ele não extrai texto do PDF: recebe os chunks efetivamente produzidos pela configuração de ingestão avaliada. Assim, quando o tamanho, a sobreposição ou o método de chunking mudar, o mesmo arquivo mestre pode ser rematerializado e os novos IDs relevantes são obtidos de forma automática e auditável.

## 1. Arquivo mestre

O formato é JSONL, com um caso por linha:

```json
{"id":"q1","query":"Como precisão é definida?","reference_answer":"Precisão é a proporção dos itens recuperados que são relevantes.","evidence":[{"document_name":"metricas.pdf","quote":"Precisão é a proporção dos itens recuperados que são relevantes.","relevance":2}]}
```

Campos obrigatórios do caso:

- `id`: identificador único e estável;
- `query`: pergunta avaliada;
- `reference_answer`: resposta esperada, sustentada somente pelas evidências;
- `evidence`: lista não vazia de pontos probatórios.

Cada evidência contém `document_name`, `document_sha256` ou ambos. Para a versão
final do corpus, recomenda-se guardar os dois: o nome facilita auditoria e o hash evita
confundir arquivos homônimos ou edições diferentes. Além dos seletores, a evidência
contém:

- `page`: página de origem opcional, usada para auditoria;
- `quote`: transcrição literal curta do PDF;
- `relevance`: inteiro `2` para evidência diretamente responsiva e `1` para evidência complementar.

O arquivo [cases.example.jsonl](cases.example.jsonl) é apenas um modelo e deve ser copiado para um arquivo versionado do experimento. Para reduzir viés, recomenda-se que as perguntas e evidências sejam anotadas antes de comparar os métodos de recuperação.

## 2. Exportação de chunks

Antes da ingestão, as citações também podem ser auditadas diretamente nos PDFs. O
comando verifica nome, SHA-256, página física, correspondência textual e ocorrências
duplicadas; usa `pypdf` quando disponível e `pdftotext` como alternativa:

```bash
python3 benchmark/groundtruth/audit_pdf.py \
  --cases caminho/cases.jsonl \
  --pdf caminho/material.pdf \
  --report caminho/source-audit.json
```

Essa auditoria confirma a âncora documental. A etapa posterior de materialização
continua necessária para descobrir quais IDs de chunk contêm cada evidência.

Exporte um JSONL após cada configuração de ingestão, com estes campos por linha:

```json
{"chunk_id":"uuid-ou-id-estável","document_name":"metricas.pdf","content":"Texto efetivamente indexado...","page_number":4}
```

Os quatro campos são obrigatórios. `page_number` é baseado em 1. Quando o arquivo mestre selecionar documentos por hash, cada chunk correspondente deve trazer também o campo opcional `document_sha256` com o SHA-256 dos bytes do PDF:

```json
{"chunk_id":"c1","document_name":"metricas.pdf","document_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","content":"...","page_number":4}
```

O hash não pode ser inferido corretamente apenas pelo texto do chunk; por isso ele deve ser calculado no PDF original durante a exportação.

Com as dependências do `ingest` instaladas, a collection completa pode ser exportada
diretamente do Qdrant:

```bash
python3 apps/ingest/api/benchmarks/groundtruth/export_chunks.py \
  --collection freire_recursive \
  --qdrant-url http://localhost:6335 \
  --pdf benchmark/sources/importancia_ato_ler.pdf \
  --output benchmark/data/freire_recursive/chunks.jsonl
```

`--pdf` pode ser repetido. O script calcula o SHA-256 de cada PDF e o associa aos
chunks cujo `document_name` coincide com o nome do arquivo.

## 3. Validação e materialização

Na raiz do repositório:

```bash
python3 benchmark/groundtruth/materialize.py validate \
  --cases benchmark/groundtruth/cases.example.jsonl \
  --chunks caminho/chunks.jsonl
```

Para gerar as projeções:

```bash
python3 benchmark/groundtruth/materialize.py materialize \
  --cases caminho/cases.jsonl \
  --chunks caminho/chunks.jsonl \
  --ingest-out caminho/generated/ingest-groundtruth.jsonl \
  --ragas-out caminho/generated/ragas-groundtruth.jsonl \
  --report-out caminho/generated/matching-report.json
```

As saídas são:

- `ingest-groundtruth.jsonl`: `id`, `query`, `reference_answer`,
  `relevant_chunk_ids` e `relevance_by_chunk`. O runner atual usa a lista binária; o
  mapa graduado permite calcular nDCG no experimento de RI;
- `ragas-groundtruth.jsonl`: `id`, `query` e `reference_answer`, diretamente aceitos pelo benchmark RAGAS;
- `matching-report.json`: hashes dos insumos, política de normalização, limiar, cobertura por evidência, páginas, chunks associados e melhores candidatos.

O mapeamento padrão é conservador (`--min-coverage 1.0`): todos os tokens da citação normalizada devem aparecer de forma contígua em pelo menos um chunk do documento selecionado. A normalização aplica NFKC, *case folding*, reparo de palavra hifenizada em quebra de linha, troca pontuação por espaços e colapso de espaços. Diacríticos são mantidos para evitar falsos positivos.

É possível definir explicitamente um limiar menor, por exemplo `--min-coverage 0.9`, mas isso transforma o casamento em parcial e deve ser justificado e registrado no protocolo experimental. A cobertura é sempre salva no relatório.

Se uma única evidência não atingir o limiar, o comando termina com código `3`, não escreve os arquivos de benchmark e salva um relatório com estado `failed_unmapped_evidence`. Isso evita avaliar silenciosamente uma pergunta com qrels incompletos. Erros de esquema terminam com código `2`.

## 4. Protocolo acadêmico recomendado

1. Congele PDFs e registre seus SHA-256.
2. Defina perguntas, respostas de referência e citações por dupla anotação; resolva discordâncias e registre o acordo entre anotadores.
3. Exporte os chunks de cada estratégia com todos os parâmetros de segmentação.
4. Materialize o mesmo mestre para cada estratégia e arquive o relatório, cujos hashes vinculam casos e chunks.
5. Execute RI primeiro. Use `relevance_by_chunk` para métricas graduadas (nDCG) e `relevant_chunk_ids` para Recall, Precision, MRR e MAP.
6. Somente depois, use a projeção RAGAS como análise adicional da geração. Não altere a resposta de referência após observar as respostas dos modelos.

A associação automática resolve **citação → IDs de chunk**. Ela não decide sozinha se a citação é semanticamente relevante: essa decisão continua sendo a anotação científica a ser validada. Amostragem manual dos relatórios de casamento deve fazer parte do controle de qualidade.

## Testes

```bash
python3 -m unittest discover -s benchmark/groundtruth -p 'test_*.py' -v
```

Os testes cobrem auditoria de páginas, casamento exato, normalização, falha estrita e
conteúdo das três saídas.

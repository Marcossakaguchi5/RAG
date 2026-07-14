# Amostra de perguntas e respostas de referência

Duas amostras por dataset. A terceira coluna é a **resposta esperada do ground
truth Gold** (`reference_answer` nos dois livros e `correct_answer` no SciQ);
não é uma resposta gerada pelo RAG.

## SciQ — benchmark de ciências (inglês)

| ID | Pergunta | Resposta esperada (ground truth/gold) |
|---|---|---|
| `sciq_test_00000` | Compounds that are capable of accepting electrons, such as O₂ or F₂, are called what? | oxidants |
| `sciq_test_00001` | What term in biotechnology means a genetically exact copy of an organism? | clone |

## Ato de Ler — 49 páginas

| ID | Pergunta | Resposta esperada (ground truth/gold) |
|---|---|---|
| `freire-pilot-001` | Por que a compreensão crítica do ato de ler não se reduz à decodificação da palavra escrita? | Porque ela se antecipa e se prolonga na compreensão do mundo, relacionando linguagem, realidade, texto e contexto. |
| `freire-pilot-002` | Qual relação o texto estabelece entre a leitura do mundo e a leitura da palavra? | A leitura do mundo vem antes da leitura da palavra, e a leitura posterior da palavra deve manter continuidade com a compreensão daquele mundo. |

## Livro de Recuperação da Informação — 581 páginas

| ID | Pergunta | Resposta esperada (ground truth/gold) |
|---|---|---|
| `iir-2009-001` | Como o livro define recuperação da informação como área acadêmica? | É a busca, em grandes coleções, de materiais geralmente textuais e não estruturados que satisfaçam uma necessidade de informação. |
| `iir-2009-005` | Por que indexar um livro inteiro como um único documento pode prejudicar a recuperação? | Porque termos distantes, presentes em capítulos diferentes, podem produzir correspondências espúrias. Unidades menores melhoram a localização da passagem, mas unidades pequenas demais podem separar evidências necessárias. |

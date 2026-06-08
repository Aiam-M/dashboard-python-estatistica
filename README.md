# Análise do Mercado de Trabalho 💼

## Contexto

Projeto desenvolvido para a disciplina de **Estatística e Probabilidade**. A ideia
é pegar um conjunto de dados real de vagas de emprego e aplicar nele os conceitos
vistos em aula — estatística descritiva, detecção de outliers, correlação e, na
segunda parte, probabilidade (Teorema de Bayes) para classificação.

O resultado final é um **dashboard interativo** onde dá pra explorar as vagas por
estado, tipo de contrato, faixa salarial e experiência exigida, além de uma seção
que tenta prever a faixa de salário de uma vaga a partir do cargo e do tipo de
contrato.

## Sobre o projeto

Trabalhamos com um dataset de vagas (`data/job_market.csv`) que tem informações
como cargo, empresa, localização, salário mínimo/máximo, experiência exigida, data
de publicação, área e nível de escolaridade.

O projeto faz duas coisas principais:

1. **Análise exploratória (estatística descritiva):** distribuições de salário e
   experiência, identificação de outliers pelo método do IQR, correlação entre as
   variáveis numéricas e cruzamentos por estado, categoria e tipo de vaga.
2. **Classificação (probabilidade):** o salário é transformado em três classes
   (`Low`, `Medium`, `High`) e três modelos tentam prever essa classe a partir do
   cargo (`job_title`) e do tipo de vaga (`job_type`):
   - **Naive Bayes** implementado na mão (priors + likelihoods com suavização de Laplace);
   - **Árvore de Decisão** (scikit-learn);
   - **Regressão Logística** (scikit-learn).

   No fim, os três são comparados por acurácia, matriz de confusão e relatório de
   classificação.

## Como o código está organizado

Uma regra importante do projeto: **cada arquivo tem uma única responsabilidade**.
Isso evita que a mesma limpeza apareça repetida em vários lugares (e com pequenas
diferenças que viram bug).

| Arquivo            | Responsabilidade                                                                                                                             | O que lê                    |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| `programacao.py`   | **Único responsável pela limpeza dos dados.** Lê o CSV cru, trata nulos, normaliza colunas, cria as colunas derivadas e exporta o CSV limpo. | `data/job_market.csv` (cru) |
| `probabilidade.py` | **Só modelagem.** Naive Bayes manual, Árvore de Decisão e Regressão Logística.                                                               | `data/job_market_clean.csv` |
| `app.py`           | **Só dashboard/UI.** Filtros, gráficos e a página de probabilidade.                                                                          | `data/job_market_clean.csv` |

### O que o `programacao.py` faz na limpeza

- Remove linhas duplicadas;
- Preenche textos vazios (`job_type`, `category`, `skills`) com "Não Informado";
- Normaliza o `job_type` (ex: "full time", "full-time" e "fulltime" viram `Full-time`);
- Converte salários para número e cria a **média salarial** (`salary_avg`);
- Preenche a experiência faltante com a **mediana por cargo** e converte para inteiro;
- Arruma a data de publicação (aceita formato normal e timestamp Unix) e cria uma
  versão no padrão BR (`publication_date_BR`);
- Separa `location` em `cidade` e `estado`;
- Cria a coluna `is_remote` (Sim/Não);
- Calcula z-score do salário, normalização da experiência e flags de outliers (IQR);
- Cria a `salary_class` (Low/Medium/High) usada na parte de probabilidade;
- Exporta tudo para `data/job_market_clean.csv`.

## Tecnologias

- **Python 3.12**
- **pandas** e **numpy** — manipulação de dados
- **scikit-learn** — Árvore de Decisão, Regressão Logística e métricas
- **streamlit** — dashboard interativo
- **plotly** — gráficos

## Como rodar

> Recomendado usar um ambiente virtual (venv).

```bash
# 1. (opcional) criar e ativar o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# 2. instalar as dependências
pip install -r requirements.txt

# 3. gerar o CSV limpo (precisa rodar pelo menos uma vez)
python programacao.py

# 4. abrir o dashboard
streamlit run app.py
```

Depois do `streamlit run`, o app abre no navegador (normalmente em
`http://localhost:8501`).

Se quiser ver os resultados da classificação direto no terminal, sem o dashboard:

```bash
python probabilidade.py
```

## Usando o dashboard

O app tem duas seções, selecionáveis na barra lateral:

### 📊 Dashboard

- **Filtros (barra lateral):** estado, tipo de vaga e faixa de experiência. Tem também uma opção para remover outliers.
- **Distribuições e Outliers:** histogramas de salário e experiência + boxplot por categoria.
- **Localização e Categorias:** vagas por categoria, por escolaridade, por estado e
  um **mapa de calor de tipos de vaga por estado** (útil pra descobrir, por exemplo,
  quais estados concentram mais vagas remotas).
- **Correlação e Insights:** matriz de correlação, dispersão experiência × salário e
  alguns insights automáticos.

### 📈 Probabilidade

Aqui rolam os três modelos de classificação de salário. Dá pra escolher um cargo e
um tipo de vaga e ver as **probabilidades estimadas** de cada faixa salarial, além
da comparação de acurácia, matrizes de confusão e relatórios dos três modelos.

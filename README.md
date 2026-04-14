# Análise do Mercado de Trabalho para Programadores

## Descrição do Projeto

Este projeto consiste em uma análise detalhada do mercado de trabalho para profissionais de programação, utilizando dados reais de vagas disponíveis. O objetivo é fornecer insights sobre tendências salariais, requisitos de experiência, tipos de contratação e outras informações relevantes para quem está ingressando ou já atua na área de tecnologia.

## Escolha do Dataset

Optamos por utilizar o dataset "job_market.csv" por ser extremamente relevante para a nossa realidade como estudantes de programação. Este conjunto de dados contém informações valiosas sobre vagas de emprego no setor de tecnologia, incluindo:

- Títulos de vagas e categorias
- Requisitos de experiência
- Faixas salariais
- Tipos de contrato
- Localização geográfica
- Benefícios oferecidos

A escolha deste dataset se deu porque ele representa diretamente o mercado ao qual estamos nos preparando para ingressar, tornando a análise especialmente útil para compreender tendências, salários esperados e competências mais demandadas.

## Escolha do Streamlit

Para a visualização dos dados optamos pelo Streamlit por conta da sua simplicidade de implementação e deploy. O Streamlit permite criar aplicações web interativas com poucas linhas de código, ideal para prototipagem rápida e apresentação de análises de dados. Além disso, sua curva de aprendizado é baixa, permitindo focar na análise dos dados em vez de complexidades de interface.

## Estrutura dos Arquivos

O projeto está dividido em dois arquivos principais com responsabilidades bem definidas:

### programacao.py
Responsável pelo tratamento completo dos dados brutos. Este script realiza:

- Leitura do CSV original
- Limpeza e padronização dos dados
- Tratamento de valores nulos
- Categorização de tipos de trabalho
- Classificação de níveis de senioridade
- Criação de faixas salariais para facilitar a análise
- Exportação dos dados limpos para "job_market_clean.csv"

### app.py
Responsável pela interface do usuário. Este script:

- Lê o CSV já limpo gerado pelo programacao.py
- Cria um dashboard interativo com filtros
- Apresenta visualizações gráficas dos dados
- Oferece métricas importantes sobre o mercado de trabalho

## Tratamentos Realizados

Alguns tratamentos específicos merecem destaque por sua complexidade:

1. **Tratamento de datas**: O dataset continha datas em diferentes formatos (normais e timestamps UNIX), então foi necessário identificar e converter ambos os formatos para um padrão único.

2. **Experiência requerida**: Para preencher valores nulos de experiência, utilizamos a mediana da experiência requerida para cada título de vaga específico, aumentando a precisão dos dados.

3. **Classificação de senioridade**: Criamos uma função para identificar automaticamente o nível de senioridade a partir do título da vaga, categorizando como Júnior, Pleno, Sênior, Estágio, etc.

4. **Tipos de trabalho**: Padronizamos os tipos de trabalho (remoto, presencial, CLT, PJ) para facilitar a análise e criação de filtros.

## Uso do Projeto

Para executar o projeto:

1. Certifique-se de ter os pacotes necessários instalados: `pandas`, `streamlit`, `plotly`, `matplotlib`, `seaborn`
2. Execute `python programacao.py` para realizar o tratamento dos dados
3. Execute `streamlit run app.py` para iniciar a aplicação web
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from probabilidade import compute_priors, compute_likelihoods, bayes_predict, encode_features, encode_features_with

# config pagina
st.set_page_config(
    page_title="Dashboard de Mercado de Trabalho",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Colunas textuais tratadas como categoria (o CSV não preserva dtype)
COLUNAS_CATEGORICAS = [
    'job_title', 'company', 'location', 'job_type', 'category',
    'skills', 'education_level', 'company_size', 'benefits',
    'cidade', 'estado', 'is_remote'
]


@st.cache_data
def carregar_dados():
    # Consome apenas o CSV limpo exportado por programacao.py.
    # Toda limpeza/derivação de colunas mora em programacao.py.
    df = pd.read_csv("data/job_market_clean.csv", parse_dates=['publication_date'])
    for col in COLUNAS_CATEGORICAS:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df


def compute_posterior_probs(row, priors, likelihoods, feature_cols, alpha=1.0):
    scores = {}
    for cls in priors.index:
        log_score = np.log(priors.loc[cls])
        for feature in feature_cols:
            value = row[feature]
            if value in likelihoods[feature].index:
                prob = likelihoods[feature].loc[value, cls]
            else:
                n_values = likelihoods[feature].shape[0]
                prob = alpha / (likelihoods[feature].shape[0] + alpha * n_values)
            log_score += np.log(prob)
        scores[cls] = log_score
    exp_scores = {cls: np.exp(score - np.max(list(scores.values()))) for cls, score in scores.items()}
    probs = {cls: exp_scores[cls] / sum(exp_scores.values()) for cls in scores}
    return probs


def render_probability_page(df):
    st.title("📊 Probabilidade e Classificação")
    st.markdown("Nesta seção, a variável de interesse é o salário, tratado como `salary_class` (Low/Medium/High). Os preditores usados são `job_title` e `job_type` juntos.")
    st.markdown("---")

    df_prob = df.copy()
    st.write(f"Total de registros para a análise: **{len(df_prob)}**")
    st.write("A variável alvo é o salário, representado como `salary_class`: Low, Medium ou High.")

    salary_ranges = df_prob.groupby('salary_class')['salary_avg'].agg(['min', 'max', 'mean']).reset_index()
    salary_ranges['min'] = salary_ranges['min'].map("R$ {:,.2f}".format)
    salary_ranges['max'] = salary_ranges['max'].map("R$ {:,.2f}".format)
    salary_ranges['mean'] = salary_ranges['mean'].map("R$ {:,.2f}".format)
    st.write("Faixas aproximadas de salário por classe:")
    st.dataframe(salary_ranges)

    if len(df_prob) < 30:
        st.warning("Dados insuficientes para uma análise robusta. Ajuste o filtro ou use o conjunto completo.")

    st.info("A análise usa `job_title` e `job_type` para prever o salário em classes Low/Medium/High. Alterar os campos abaixo atualiza a previsão automaticamente.")

    target_col = 'salary_class'
    feature_cols = ['job_title', 'job_type']
    train_df, test_df = train_test_split(df_prob, test_size=0.30, random_state=42, stratify=df_prob[target_col])

    y_test = test_df[target_col]
    y_pred_bayes, priors_train, likelihoods_train = bayes_predict(train_df, test_df, target_col, feature_cols, alpha=1.0)
    bayes_acc = accuracy_score(y_test, y_pred_bayes)

    X_train, encoder = encode_features(train_df, feature_cols)
    X_test = encode_features_with(encoder, test_df, feature_cols)
    y_train = train_df[target_col]

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, y_pred_dt)

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, y_pred_lr)

    st.subheader("Exemplo de previsão para salário")
    col_a, col_b = st.columns(2)
    with col_a:
        job_title_choice = st.selectbox(
            "Job Title:",
            sorted(df_prob['job_title'].dropna().unique()),
            index=0
        )
    with col_b:
        job_type_choice = st.selectbox(
            "Job Type:",
            sorted(df_prob['job_type'].dropna().unique()),
            index=0
        )

    posterior_example = compute_posterior_probs(
        pd.Series({'job_title': job_title_choice, 'job_type': job_type_choice}),
        priors_train,
        likelihoods_train,
        feature_cols,
        alpha=1.0
    )
    st.write(f"Probabilidades estimadas para `{job_title_choice}` + `{job_type_choice}`:")
    st.write({k: f"{v:.3f}" for k, v in posterior_example.items()})

    st.subheader("Resultados gerais")
    st.markdown(
        f"- Bayes (manual): **{bayes_acc:.3f}**  \n"
        f"- Decision Tree: **{dt_acc:.3f}**  \n"
        f"- Logistic Regression: **{lr_acc:.3f}**"
    )

    st.subheader("Priors e likelihoods")
    st.write("Priors para cada classe de salário:")
    st.dataframe(priors_train.to_frame('prior'))
    for feature in feature_cols:
        st.write(f"Likelihoods P({feature} | salary_class):")
        st.dataframe(likelihoods_train[feature])

    st.subheader("Confusion matrices")
    labels = ['Low', 'Medium', 'High']
    cm_bayes = confusion_matrix(y_test, y_pred_bayes, labels=labels)
    cm_dt = confusion_matrix(y_test, y_pred_dt, labels=labels)
    cm_lr = confusion_matrix(y_test, y_pred_lr, labels=labels)

    st.write("**Bayes (manual)**")
    st.dataframe(pd.DataFrame(cm_bayes, index=labels, columns=labels))
    st.write("**Decision Tree**")
    st.dataframe(pd.DataFrame(cm_dt, index=labels, columns=labels))
    st.write("**Logistic Regression**")
    st.dataframe(pd.DataFrame(cm_lr, index=labels, columns=labels))

    st.subheader("Relatórios de classificação")
    st.text("Bayes (manual)")
    st.text(classification_report(y_test, y_pred_bayes, labels=labels, zero_division=0))
    st.text("Decision Tree")
    st.text(classification_report(y_test, y_pred_dt, labels=labels, zero_division=0))
    st.text("Logistic Regression")
    st.text(classification_report(y_test, y_pred_lr, labels=labels, zero_division=0))


df = carregar_dados()

st.sidebar.header("Navegação")
page = st.sidebar.radio("Seção:", ["Dashboard", "Probabilidade"] )

if page == "Probabilidade":
    render_probability_page(df)
    st.stop()

# Barra lateral de filtro e opções de análise
st.sidebar.header("🔍 Filtros de Pesquisa")
estados = sorted(df['estado'].dropna().unique())
estado_sel = st.sidebar.multiselect("Estado:", estados, default=[])

tipos_vaga = sorted(df['job_type'].dropna().unique())
tipo_sel = st.sidebar.multiselect("Tipo de Vaga:", tipos_vaga, default=[])

min_exp, max_exp = int(df['experience_required'].min()), int(df['experience_required'].max())
exp_sel = st.sidebar.slider("Anos de Experiência:", min_exp, max_exp, (min_exp, max_exp))
remove_outliers = st.sidebar.checkbox("Remover outliers de salário/experiência", value=False)
variavel_alvo = 'category'

# Filtragem do DF — filtro vazio significa "incluir tudo" (não agressivo)
filtro = df['experience_required'].between(exp_sel[0], exp_sel[1])
if estado_sel:
    filtro = filtro & df['estado'].isin(estado_sel)
if tipo_sel:
    filtro = filtro & df['job_type'].isin(tipo_sel)

df_filtered = df[filtro].copy()
if remove_outliers:
    df_filtered = df_filtered[~(df_filtered['salary_outlier'] | df_filtered['experience_outlier'])]

# Estatísticas rápidas
total_vagas = len(df_filtered)
media_salarial = df_filtered['salary_avg'].mean() if total_vagas else 0
vagas_remotas = int((df_filtered['is_remote'] == 'Sim').sum())
media_experiencia = df_filtered['experience_required'].mean() if total_vagas else 0
outliers_removidos = df[filtro].shape[0] - total_vagas

st.title("💼 Análise do Mercado de Trabalho")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de Vagas", f"{total_vagas}")
col2.metric("Média Salarial", f"R$ {media_salarial:,.2f}")
col3.metric("Vagas Remotas", f"{vagas_remotas}")
col4.metric("Experiência Média (Anos)", f"{media_experiencia:.1f}")

if remove_outliers:
    st.info(f"{outliers_removidos} registros foram removidos do conjunto filtrado porque eram outliers de salário ou experiência.")

st.markdown("### Variável categórica de interesse: **{}**".format(variavel_alvo))
st.markdown("---")

aba1, aba2, aba3 = st.tabs([
    "📊 Distribuições e Outliers",
    "📍 Localização e Categorias",
    "📈 Correlação e Insights"
])

with aba1:
    st.subheader("Distribuição de variáveis quantitativas")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist_salary = px.histogram(
            df_filtered,
            x='salary_avg',
            nbins=30,
            title='Distribuição da Média Salarial',
            labels={'salary_avg': 'Média Salarial'},
            marginal='box'
        )
        st.plotly_chart(fig_hist_salary, width='stretch')

    with col_b:
        fig_hist_exp = px.histogram(
            df_filtered,
            x='experience_required',
            nbins=15,
            title='Distribuição de Experiência Requerida',
            labels={'experience_required': 'Experiência Requerida (anos)'},
            marginal='rug'
        )
        st.plotly_chart(fig_hist_exp, width='stretch')

    st.subheader("Outliers identificados")
    contagem_outliers = df_filtered[['salary_outlier', 'experience_outlier']].sum()
    st.write(
        f"Outliers de salário: {int(contagem_outliers['salary_outlier'])} | "
        f"Outliers de experiência: {int(contagem_outliers['experience_outlier'])}"
    )

    st.subheader("Média salarial por {}".format(variavel_alvo.replace('_', ' ')))
    fig_cat_salary = px.box(
        df_filtered,
        x=variavel_alvo,
        y='salary_avg',
        color='is_remote',
        labels={'salary_avg': 'Média Salarial', variavel_alvo: variavel_alvo.replace('_', ' ')},
        title=f'Média salarial por {variavel_alvo.replace("_", " ")}'
    )
    st.plotly_chart(fig_cat_salary, width='stretch')

with aba2:
    st.subheader("Contagem de categorias qualitativas")
    col_c, col_d = st.columns(2)
    with col_c:
        fig_category = px.bar(
            df_filtered['category'].value_counts().reset_index(),
            x='category',
            y='count',
            title='Vagas por Categoria',
            labels={'count': 'Quantidade', 'category': 'Categoria'}
        )
        st.plotly_chart(fig_category, width='stretch')

    with col_d:
        fig_education = px.bar(
            df_filtered['education_level'].value_counts().reset_index(),
            x='education_level',
            y='count',
            title='Vagas por Nível de Escolaridade',
            labels={'count': 'Quantidade', 'education_level': 'Escolaridade'}
        )
        st.plotly_chart(fig_education, width='stretch')

    st.subheader("Vagas por estado")
    vagas_estado = df_filtered['estado'].value_counts().reset_index()
    fig_pie = px.pie(vagas_estado, values='count', names='estado', hole=0.4,
                     title='Distribuição de vagas por estado')
    st.plotly_chart(fig_pie, width='stretch')

    st.subheader("Tipos de vaga por estado")
    st.caption("Use para descobrir, por exemplo, quais estados concentram mais vagas remotas. "
               "Filtre por Tipo de Vaga na barra lateral para focar em um tipo específico.")
    if df_filtered.empty:
        st.info("Nenhuma vaga no conjunto filtrado atual.")
    else:
        tabela_estado_tipo = pd.crosstab(
            df_filtered['estado'].astype(str),
            df_filtered['job_type'].astype(str)
        )
        # Ordena os estados pelo total de vagas (do maior para o menor)
        tabela_estado_tipo = tabela_estado_tipo.loc[
            tabela_estado_tipo.sum(axis=1).sort_values(ascending=False).index
        ]
        fig_estado_tipo = px.imshow(
            tabela_estado_tipo,
            text_auto=True,
            aspect='auto',
            color_continuous_scale='Blues',
            labels={'x': 'Tipo de Vaga', 'y': 'Estado', 'color': 'Nº de Vagas'},
            title='Quantidade de vagas por estado e tipo de vaga'
        )
        st.plotly_chart(fig_estado_tipo, width='stretch')

with aba3:
    st.subheader("Correlação entre variáveis quantitativas")
    corr = df_filtered[['salary_min', 'salary_max', 'salary_avg', 'experience_required']].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        title='Matriz de Correlação'
    )
    st.plotly_chart(fig_corr, width='stretch')

    st.subheader("Relações importantes")
    fig_scatter = px.scatter(
        df_filtered,
        x='experience_required',
        y='salary_avg',
        color='category',
        size='salary_max',
        hover_name='job_title',
        title='Experiência vs. Salário Médio'
    )
    st.plotly_chart(fig_scatter, width='stretch')

    corr_exp_salary = corr.loc['experience_required', 'salary_avg'] if 'experience_required' in corr.index else 0
    remote_share = df_filtered['is_remote'].value_counts(normalize=True).get('Sim', 0) * 100
    sal_highest_category = df_filtered.groupby('category')['salary_avg'].mean().idxmax()
    avg_highest_category = df_filtered.groupby('category')['salary_avg'].mean().max()

    st.markdown("### Insights relevantes")
    st.markdown(
        f"- A correlação entre experiência e salário médio é de **{corr_exp_salary:.2f}**, indicando uma relação moderada a alta.\n"
        f"- **{remote_share:.1f}%** das vagas filtradas são remotas.\n"
        f"- A categoria com maior salário médio no conjunto filtrado é **{sal_highest_category}** com média de **R$ {avg_highest_category:,.2f}**.\n"
        f"- A variável categórica destacada para análise é **{variavel_alvo}**, que é um bom candidato para a segunda parte probabilística do projeto."
    )

with st.expander("📄 Visualizar Dados Brutos (Filtrados)"):
    st.dataframe(df_filtered.sort_values(by='publication_date', ascending=False), width='stretch')

# rodapé
st.markdown("---")
st.caption(f"Última atualização dos dados: {df['publication_date'].max().strftime('%d/%m/%Y')}")
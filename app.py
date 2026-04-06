import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# config pagina
st.set_page_config(
    page_title="Dashboard de Mercado de Trabalho",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("data/job_market.csv")
    
    # Correção que tu fizeste dos nulos lá
    colunas_texto = ['job_type', 'category', 'skills']
    df[colunas_texto] = df[colunas_texto].fillna('Não Informado')
    
    # Nem sei como q tu fizeste isso aqui pra falar a real, entendi nada, só aceitei
    df['experience_required'] = df.groupby('job_title')['experience_required'].transform(
        lambda x: x.fillna(x.median()) if not x.median() is np.nan else x
    )
    df = df.dropna(subset=['experience_required'])
    df['experience_required'] = df['experience_required'].astype(int)
    
    # O tratamento de datas aqui, tudo certo!
    timestamps_normais = pd.to_datetime(df['publication_date'], errors='coerce')
    timestamps_Unix = pd.to_numeric(df['publication_date'], errors='coerce')
    datas_Unix = pd.to_datetime(timestamps_Unix, unit='s')
    df['publication_date'] = timestamps_normais.fillna(datas_Unix)
    
    # As counas de salário, mas adicionei fillna(0) pra ter certza.
    df['salary_min'] = pd.to_numeric(df['salary_min'], errors='coerce').fillna(0)
    df['salary_max'] = pd.to_numeric(df['salary_max'], errors='coerce').fillna(0)
    df['salary_avg'] = (df['salary_max'] + df['salary_min']) / 2
    
    # Tratamento da parte de localização aqui, juntei tudo e fiz um tanto diferente;
    seperacao = df['location'].str.split(',', expand=True)
    df['cidade'] = seperacao[0].str.strip()
    df['estado'] = seperacao[1].str.strip() if len(seperacao.columns) > 1 else "Não Informado"
    df.loc[df['cidade'].str.lower() == "remote", 'estado'] = "Remote"
    
    # basicamente fui atrás de entender aquele teu lá em cima e percebi q era só um condicional e eu que fui burro.
    # Aqui é pra saber se o trabalho é remoto ou não.
    df['is_remote'] = df['job_type'].apply(
        lambda x: "Sim" if 'remote' in str(x).lower() else ("Não Informado" if x == 'Não Informado' else "Não")
    )

    # Normalização dos tipos de trabalho
    def classify_job_type(val):
        v = str(val).lower().strip()
        if 'remote' in v:
            return 'Remote'
        elif v in ('full-time', 'full time', 'fulltime'):
            return 'Full-time'
        elif v in ('part-time', 'part time', 'parttime'):
            return 'Part-time'
        elif 'contract' in v or 'freelance' in v or 'temp' in v:
            return 'Contract'
        elif 'intern' in v or 'praktik' in v or 'werkstudent' in v or 'student' in v:
            return 'Internship'
        else:
            return 'Other'

    df['job_type_normalized'] = df['job_type'].apply(classify_job_type)

    # Extração de nível de senioridade a partir do job_title
    def classify_seniority(title):
        t = str(title).lower()
        if any(k in t for k in ['praktik', 'intern', 'trainee', 'estágio', 'estagio', 'working student', 'werkstudent']):
            return 'Intern/Trainee'
        if 'junior' in t or '(junior)' in t:
            return 'Junior'
        if any(k in t for k in ['lead', 'principal', 'head of', 'staff']):
            return 'Lead/Principal'
        if 'senior' in t:
            return 'Senior'
        return 'Mid/Other'

    df['seniority'] = df['job_title'].apply(classify_seniority)

    # Faixa salarial para filtros mais amigáveis
    df['salary_bucket'] = pd.cut(
        df['salary_max'],
        bins=[0, 50000, 80000, 120000, 160000, 200000, 99999999],
        labels=['≤ 50k', '50k–80k', '80k–120k', '120k–160k', '160k–200k', '> 200k']
    )

    return df

df = load_and_clean_data()

# Barra lateral de filtros
st.sidebar.header("🔍 Filtros de Pesquisa")

# Por estado
estados = sorted(df['estado'].dropna().unique())
estado_sel = st.sidebar.multiselect("Selecione o Estado:", estados, default=estados)

# Por tempo de xp
min_exp, max_exp = int(df['experience_required'].min()), int(df['experience_required'].max())
exp_sel = st.sidebar.slider("Anos de Experiência:", min_exp, max_exp, (min_exp, max_exp))

# Por tipo de trabalho
job_types = sorted(df['job_type_normalized'].dropna().unique())
job_type_sel = st.sidebar.multiselect("Tipo de Trabalho:", job_types, default=job_types)

# Por nível de senioridade
seniority_order = ['Intern/Trainee', 'Junior', 'Mid/Other', 'Senior', 'Lead/Principal']
seniority_sel = st.sidebar.multiselect("Nível Senioridade:", seniority_order, default=seniority_order)

# Por faixa salarial
salary_buckets = df['salary_bucket'].dropna().cat.categories.tolist()
salary_sel = st.sidebar.multiselect("Faixa Salarial Máx:", salary_buckets, default=salary_buckets)

# Filtragem do DF
mask = (
    df['estado'].isin(estado_sel) &
    df['experience_required'].between(exp_sel[0], exp_sel[1]) &
    df['job_type_normalized'].isin(job_type_sel) &
    df['seniority'].isin(seniority_sel) &
    df['salary_bucket'].isin(salary_sel)
)
df_filtered = df[mask].copy()

# Parte princiapl aqui!
st.title("💼 Análise do Mercado de Trabalho")
st.markdown("---")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total de Vagas", len(df_filtered))
with m2:
    st.metric("Média Salarial", f"R$ {df_filtered['salary_avg'].mean():,.2f}")
with m3:
    st.metric("Vagas Remotas", len(df_filtered[df_filtered['is_remote'] == "Sim"]))
with m4:
    st.metric("Exp. Média (Anos)", f"{df_filtered['experience_required'].mean():.1f}")

st.markdown("###")

# Parte interativa visual aqui!
aba1, aba2, aba3 = st.tabs(["📊 Distribuição e Salários", "📍 Localização e Categorias", "📈 Análise Complementar"])

with aba1:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Distribuição Salarial por Categoria")
        fig_box = px.box(df_filtered, x="category", y="salary_avg",
                         color="is_remote", points="all",
                         labels={'salary_avg': 'Média Salarial', 'category': 'Categoria'})
        st.plotly_chart(fig_box, use_container_width=True)

    with col_right:
        st.subheader("Relação Experiência x Salário")
        fig_scatter = px.scatter(df_filtered, x="experience_required", y="salary_avg",
                         size="salary_max", color="category", hover_name="job_title")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Salário por Senioridade
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("Salário Médio por Senioridade")
        seniority_salary = df_filtered.groupby('seniority', observed=False)['salary_avg'].mean().reset_index()
        fig_sen = px.bar(seniority_salary, x='seniority', y='salary_avg',
                         color='salary_avg', color_continuous_scale='Blues',
                         labels={'salary_avg': 'Salário Médio', 'seniority': 'Senioridade'})
        st.plotly_chart(fig_sen, use_container_width=True)

    with col_s2:
        st.subheader("Salário por Tipo de Trabalho")
        jt_salary = df_filtered.groupby('job_type_normalized', observed=False)['salary_avg'].mean().reset_index()
        fig_jt = px.bar(jt_salary, x='job_type_normalized', y='salary_avg',
                        color='job_type_normalized', labels={'salary_avg': 'Salário Médio'})
        st.plotly_chart(fig_jt, use_container_width=True)

with aba2:
    col_vagas, col_geo = st.columns([1, 1.5])

    with col_vagas:
        st.subheader("Top 10 Cargos com Mais Vagas")
        top_jobs = df_filtered['job_title'].value_counts().nlargest(10).reset_index()
        top_jobs.columns = ['job_title', 'count']
        fig_bar = px.bar(top_jobs, x='count', y='job_title', orientation='h',
                         color='count', color_continuous_scale='Viridis')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_geo:
        st.subheader("Vagas por Estado")
        vagas_estado = df_filtered['estado'].value_counts().reset_index()
        vagas_estado.columns = ['estado', 'count']
        fig_pie = px.pie(vagas_estado, values='count', names='estado', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

with aba3:
    # Distribuição de vagas por senioridade
    ac1, ac2, ac3 = st.columns(3)

    with ac1:
        st.subheader("Distribuição por Senioridade")
        seniority_dist = df_filtered['seniority'].value_counts().reset_index()
        seniority_dist.columns = ['seniority', 'count']
        fig_donut = px.pie(
            seniority_dist,
            values='count', names='seniority', hole=0.6,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with ac2:
        st.subheader("Distribuição por Tipo de Trabalho")
        jt_dist = df_filtered['job_type_normalized'].value_counts().reset_index()
        jt_dist.columns = ['job_type_normalized', 'count']
        fig_type = px.pie(
            jt_dist,
            values='count', names='job_type_normalized', hole=0.6,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_type, use_container_width=True)

    with ac3:
        st.subheader("Distribuição por Faixa Salarial")
        fig_bucket = px.histogram(
            df_filtered, x='salary_bucket', color='seniority',
            barmode='group', labels={'salary_bucket': 'Faixa Salarial Máx', 'count': 'Qtd Vagas'},
            category_orders={'salary_bucket': ['≤ 50k', '50k–80k', '80k–120k', '120k–160k', '160k–200k', '> 200k']},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig_bucket, use_container_width=True)

    # Salário médio por categoria e senioridade (heatmap via bar group)
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.subheader("Salário Médio: Categoria × Senioridade")
        cat_sen = df_filtered.groupby(['category', 'seniority'], observed=False)['salary_avg'].mean().reset_index()
        fig_cat_sen = px.bar(cat_sen, x='category', y='salary_avg', color='seniority',
                             barmode='group', labels={'salary_avg': 'Salário Médio'},
                             category_orders={'seniority': seniority_order})
        st.plotly_chart(fig_cat_sen, use_container_width=True)

with st.expander("📄 Visualizar Dados Brutos (Filtrados)"):
    st.dataframe(df_filtered.sort_values(by='publication_date', ascending=False),
                 use_container_width=True)

# rodapé
st.markdown("---")
st.caption(f"Última atualização dos dados: {df['publication_date'].max().strftime('%d/%m/%Y')}")
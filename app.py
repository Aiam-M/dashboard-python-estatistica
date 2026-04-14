import streamlit as st
import pandas as pd
import plotly.express as px

# config pagina
st.set_page_config(
    page_title="Dashboard de Mercado de Trabalho",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_and_clean_data():
    # Carrega o DataFrame já limpo gerado pelo script de tratamento
    df = pd.read_csv("data/job_market_clean.csv")
    df['salary_bucket'] = df['salary_bucket'].astype('category')
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
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
    
    return df

df = load_and_clean_data()

# Barra lateral de filtro de estado e tempo de experiência
st.sidebar.header("🔍 Filtros de Pesquisa")

# Por estado
estados = sorted(df['estado'].dropna().unique())
estado_sel = st.sidebar.multiselect("Selecione o Estado:", estados, default=estados)

# Por tempo de xp
min_exp, max_exp = int(df['experience_required'].min()), int(df['experience_required'].max())
exp_sel = st.sidebar.slider("Anos de Experiência:", min_exp, max_exp, (min_exp, max_exp))

# Filtragem do DF
df_filtered = df[
    (df['estado'].isin(estado_sel)) &
    (df['experience_required'].between(exp_sel[0], exp_sel[1]))
]

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
aba1, aba2 = st.tabs(["📊 Distribuição e Salários", "📍 Localização e Categorias"])

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

with aba2:
    col_vagas, col_geo = st.columns([1, 1.5])
    
    with col_vagas:
        st.subheader("Top 10 Cargos com Mais Vagas")
        top_jobs = df_filtered['job_title'].value_counts().nlargest(10).reset_index()
        fig_bar = px.bar(top_jobs, x='count', y='job_title', orientation='h',
                         color='count', color_continuous_scale='Viridis')
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_geo:
        st.subheader("Vagas por Estado")
        vagas_estado = df_filtered['estado'].value_counts().reset_index()
        fig_pie = px.pie(vagas_estado, values='count', names='estado', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

with st.expander("📄 Visualizar Dados Brutos (Filtrados)"):
    st.dataframe(df_filtered.sort_values(by='publication_date', ascending=False), 
                 use_container_width=True)

# rodapé
st.markdown("---")
st.caption(f"Última atualização dos dados: {df['publication_date'].max().strftime('%d/%m/%Y')}")
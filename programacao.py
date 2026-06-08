import pandas as pd
import numpy as np

# =============================================================================
# Único responsável pela limpeza dos dados.
#
# Toda limpeza/derivação de coluna mora aqui. Se app.py ou probabilidade.py
# precisarem de algo novo, a coluna deve ser adicionada nesta função e os
# outros apenas consomem o CSV limpo (data/job_market_clean.csv).
# Nenhum consumidor deve ler o CSV cru (data/job_market.csv).
# =============================================================================

CAMINHO_CRU = "data/job_market.csv"
CAMINHO_LIMPO = "data/job_market_clean.csv"

# Colunas categóricas (o CSV não guarda dtype, então os consumidores recastam)
COLUNAS_CATEGORICAS = [
    'job_title', 'company', 'location', 'job_type', 'category',
    'skills', 'education_level', 'company_size', 'benefits',
    'cidade', 'estado', 'is_remote'
]


def _normalizar_job_type(df):
    """Unifica variantes como 'full time' e 'full-time'."""
    df['job_type'] = (
        df['job_type'].astype(str).str.strip()
        .str.replace(r'[\s-]+', ' ', regex=True).str.lower()
    )
    df['job_type'] = df['job_type'].replace({
        'não informado': 'Não Informado',
        'full time': 'Full-time',
        'fulltime': 'Full-time',
        'part time': 'Part-time',
        'parttime': 'Part-time',
        'remote': 'Remote',
        'contract': 'Contract',
        'manager': 'Manager'
    })
    return df


def _tratar_experiencia(df):
    """Preenche experiência nula com a mediana por cargo e converte para int."""
    df['experience_required'] = pd.to_numeric(df['experience_required'], errors='coerce')
    df['experience_required'] = df.groupby('job_title')['experience_required'].transform(
        lambda x: x.fillna(x.median()) if not np.isnan(x.median()) else x
    )
    # Se ainda for NaN é porque não há nenhuma informação para esse job_title: descarta
    df = df.dropna(subset=['experience_required'])
    df['experience_required'] = df['experience_required'].astype(int)
    return df


def _tratar_datas(df):
    """Aceita timestamp normal e Unix; cria também a coluna no padrão BR."""
    timestamps_normais = pd.to_datetime(df['publication_date'], errors='coerce')
    timestamps_unix = pd.to_numeric(df['publication_date'], errors='coerce')
    datas_unix = pd.to_datetime(timestamps_unix, unit='s', errors='coerce')
    df['publication_date'] = timestamps_normais.fillna(datas_unix)
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
    df['publication_date_BR'] = df['publication_date'].dt.strftime('%d/%m/%Y')
    return df


def _extrair_local(df):
    """Separa location em cidade e estado e marca remoto no estado."""
    separacao = df['location'].astype(str).str.split(',', expand=True)
    df['cidade'] = separacao[0].str.strip()
    if separacao.shape[1] > 1:
        df['estado'] = separacao[1].str.strip().fillna("Não Informado")
    else:
        df['estado'] = "Não Informado"
    df.loc[df['cidade'].str.lower().isin(['remote', 'remoto']), 'estado'] = "Remote"
    return df


def _classificar_remoto(verificacao):
    if verificacao == 'Não Informado' or pd.isna(verificacao):
        return 'Não Informado'
    if 'remote' in str(verificacao).lower():
        return 'Sim'
    return 'Não'


def _detectar_outliers_iqr(serie):
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    return (serie < limite_inferior) | (serie > limite_superior)


def limpar_dados(caminho_cru=CAMINHO_CRU, caminho_limpo=CAMINHO_LIMPO, salvar=True):
    """Lê o CSV cru, aplica toda a limpeza/derivação e exporta o CSV limpo.

    Retorna o DataFrame limpo (com publication_date como datetime).
    """
    df = pd.read_csv(caminho_cru)

    # Remoção de duplicatas
    df = df.drop_duplicates()

    # Preenchimento de valores textuais ausentes
    colunas_texto = ['job_type', 'category', 'skills']
    df[colunas_texto] = df[colunas_texto].fillna('Não Informado')

    # Normalização de job_type
    df = _normalizar_job_type(df)

    # Salários: numérico + média
    df['salary_min'] = pd.to_numeric(df['salary_min'], errors='coerce').fillna(0.0)
    df['salary_max'] = pd.to_numeric(df['salary_max'], errors='coerce').fillna(0.0)
    df['salary_avg'] = (df['salary_min'] + df['salary_max']) / 2

    # Experiência: mediana por cargo
    df = _tratar_experiencia(df)

    # Datas (normal + Unix) e formato BR
    df = _tratar_datas(df)

    # Cidade/Estado
    df = _extrair_local(df)

    # Trabalho remoto (padrão Sim/Não)
    df['is_remote'] = df['job_type'].apply(_classificar_remoto)

    # Normalização e padronização para análises adicionais
    df['salary_avg_z'] = (df['salary_avg'] - df['salary_avg'].mean()) / df['salary_avg'].std(ddof=0)
    amplitude_exp = df['experience_required'].max() - df['experience_required'].min()
    if amplitude_exp != 0:
        df['experience_norm'] = (df['experience_required'] - df['experience_required'].min()) / amplitude_exp
    else:
        df['experience_norm'] = 0.0

    # Flags de outliers (IQR)
    df['salary_outlier'] = _detectar_outliers_iqr(df['salary_avg'])
    df['experience_outlier'] = _detectar_outliers_iqr(df['experience_required'])

    # Classe de salário (terços globais) — alvo para a parte probabilística
    df['salary_class'] = pd.qcut(
        df['salary_avg'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop'
    )
    df = df.dropna(subset=['salary_class'])
    df['salary_class'] = df['salary_class'].astype(str)

    if salvar:
        df.to_csv(caminho_limpo, index=False)

    return df


if __name__ == '__main__':
    df = limpar_dados()

    print("-" * 30)
    print("Nulos por coluna:")
    print(df.isnull().sum())
    print("-" * 30)
    print("Tipos:")
    print(df.dtypes)
    print("-" * 30)
    print("Distribuição de salary_class:")
    print(df['salary_class'].value_counts())
    print("-" * 30)
    print(f"CSV limpo exportado para {CAMINHO_LIMPO} ({len(df)} linhas)")

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("data/job_market.csv")


#Corrigindo as colunas nulas
colunas_texto=['job_type', 'category', 'skills']
df[colunas_texto]= df[colunas_texto].fillna('Não Informado')

print("-"*30)
print((df[colunas_texto]=='Não Informado').sum())

print("-"*30)
#Aqui eu agrupo a experiencia necessária por cargo de trabalho(ex:junior developer)
#Aí eu transformo os nulos na mediana da experiência dos cargos
df['experience_required'] = df.groupby('job_title')['experience_required'].transform(
    lambda x: x.fillna(x.median()) if not x.median() is np.nan else x
)
df=df.dropna(subset=['experience_required'])

#Agora, convertemos para inteiro já que não há valores float no df
df['experience_required']=df['experience_required'].astype(int)

#Fazemos uma separação do que é TimeStamp do Unix TimeStamp
#Pegamos os timestamps certos e os que não forem timestamp viram NaT(Not a Time)
timestamps_normais = pd.to_datetime(df['publication_date'],errors='coerce')

#Transformamos para numéricos, se o valor for NaT ele vai virar certinho, mas se não for é porque ele já tá no formato date
timestamps_Unix = pd.to_numeric(df['publication_date'], errors='coerce')

#Transformamos esses valores para datas normais avisando que ele está em Unix TimeStamp
datas_Unix = pd.to_datetime(timestamps_Unix, unit='s')

#Juntamos tudo e corrigimos a coluna
df['publication_date'] = timestamps_normais.fillna(datas_Unix)
df['publication_date'] = pd.to_datetime(df['publication_date'])

#Por fim, criamos uma coluna no padrão BR
df['publication_date_BR'] = pd.to_datetime(df['publication_date']).dt.strftime('%d/%m/%Y')

#Converte os salários para float, com tratamento de valores nulos
df['salary_min'] = pd.to_numeric(df['salary_min'], errors='coerce').fillna(0)
df['salary_max'] = pd.to_numeric(df['salary_max'], errors='coerce').fillna(0)

#Criando algumas colunas

#Criando a coluna de média salarial
df['salary_avg'] = (df['salary_max'] + df['salary_min']) / 2

#Separa o location em duas partes (Cidade e Estado)
separacao = df['location'].str.split(',', expand=True)
df['cidade'] = separacao[0].str.strip()
df['estado'] = separacao[1].str.strip() if len(separacao.columns) > 1 else "Não Informado"
df.loc[df['cidade'].str.lower() == "remote", 'estado'] = "Remote"

#Definindo a regra para uma coluna verificando se algum trabalho é remoto ou não
#Se tiver escrito que a vaga é remota ele vai retornar o Sim ou Não
def verificar_remoto(verificacao):
    if verificacao == 'Não Informado':
        return 'Não Informado'
    elif 'remote' in str(verificacao).lower():
        return "Sim"
    else:
        return "Não"

#Cria a coluna
df['is_remote'] = df['job_type'].apply(verificar_remoto)

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
df['salary_bucket']=df['salary_bucket'].astype('category')

#Exporta a planilha já salva e sem os índices
df.to_csv("data/job_market_clean.csv",index=False)


print(df.isnull().sum())
print("-"*30)
print(df.dtypes)
print("-"*30)
print(df['publication_date_BR'].unique())
print(df['publication_date'].unique())
print(df['salary_min'].unique())
print(df['salary_max'].unique())
print(df['salary_avg'].unique())
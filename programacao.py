import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


df = pd.read_csv("data/job_market.csv")


#Corrigindo as colunas nulas
colunas_texto=['job_type', 'category', 'skills']
df[colunas_texto]= df[colunas_texto].fillna('Não Informado')

print("-"*30)
print((df[colunas_texto]=='Não Informado').sum())

print("-"*30)
#Aqui eu agrupo a experiencia necessária por cargo de trabalho(ex:junior developer)
#Aí eu transformo os nulos na mediana da experiência dos cargos
df['experience_required'] = df.groupby('job_title')['experience_required'].transform(lambda x: x.fillna(x.median()))

#Se ainda assim alguma experience_required for NaN é porque não temos nenhuma infomração de nenhuma experience_required desse jobtitle
#dito isso, excluiremos
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

#Converte os salários para float
df['salary_min'] = df['salary_min'].astype(float)
df['salary_max'] = df['salary_max'].astype(float)


#Criando algumas colunas

#Criando a coluna de média salarial
df['salary_avg'] = (df['salary_max'] + df['salary_min']) / 2

#Separa o location em duas partes (Cidade e Estado)
seperacao = df['location'].str.split(',', expand=True)

#Pega a cidade
df['cidade'] = seperacao[0]

#Pega o segundo valor da separacao e coloca não informado se não tiver nada
if len(seperacao.columns) > 1:
    df['estado'] = seperacao[1].fillna("Não Informado")
else:
    df['estado'] = "Não Informado"

#Se o trabalho for remoto, informa na coluna de Estado também
df.loc[df['cidade']== "Remote", 'estado'] = "Remote"
df.loc[df['cidade']== "remote", 'estado'] = "Remote"


#Definindo a regra para uma coluna verificando se algum trabalho é remoto ou não
#Se tiver escrito que a vaga é remota ele vai retornar o false ou true
def verificar_remoto(verificacao):
    if verificacao == 'Não Informado':
        return 'Não Informado'
    elif 'remote' in str(verificacao).lower():
        return "Yes"
    else:
        return "No"
#Cria a coluna
df['is_remote'] = df['job_type'].apply(verificar_remoto)

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
# Importando as Bibliotecas necessárías

import pandas as pd
from matplotlib import pyplot
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn import metrics
import pydot
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# Abrindo o arquivo 

dados = pd.read_csv('dados/hr.csv')

# Verificar as 5 primeiras linhas para observar como os dados saíram.

dados.head()

# Verificar o volume do Dataset

dados.shape

# Verificar como são os tipos de dados

dados.dtypes

# Analisar o dataframe

dados.describe().transpose()

# Verificar a quantidade de registros faltantes

print('Quantidade de registros faltantes {}'. format(dados[dados['Rising_Star'].isnull()].shape[0]))
dados[dados['Rising_Star'].isnull()]

# Verificar a correlação dos dados

dados_cor = dados.corr()
print(dados_cor)

# Representação Gráfica

pyplot.scatter(dados.last_evaluation, dados.time_spend_company)
pyplot.title('Gráfico de Dispersão entre ultima avaliação e o tempo gasto na companhia')
pyplot.show()
dados.hist()

pyplot.scatter(dados.last_evaluation, dados.average_montly_hours)
pyplot.title('Gráfico de Dispersão entre ultima avaliação e a média de gastos mensais')
pyplot.show()

pyplot.scatter(dados.time_spend_company, dados.average_montly_hours)
pyplot.title('Gráfico de Dispersão entre o tempo gasto da companhia e a média de gastos mensais')
pyplot.show()

# Verificar a correlação entre algumas variáveis de forma mais profunda
dados['average_montly_hours'].corr(dados['last_evaluation'])
dados['time_spend_company'].corr(dados['last_evaluation'])
dados['Will_Relocate'].corr(dados['last_evaluation'])
dados['average_montly_hours'].corr(dados['Will_Relocate'])
dados['time_spend_company'].corr(dados['Will_Relocate'])
dados['time_spend_company'].corr(dados['Work_accident'])
dados['average_montly_hours'].corr(dados['Work_accident'])
dados['average_montly_hours'].corr(dados['number_project'])
dados['Will_Relocate'].corr(dados['number_project'])
dados['average_montly_hours'].corr(dados['number_project'])
dados['time_spend_company'].corr(dados['number_project'])
dados['time_spend_company'].corr(dados['left_Company'])
dados['time_spend_company'].corr(dados['promotion_last_5years'])
dados['left_Company'].corr(dados['promotion_last_5years'])
dados['average_montly_hours'].corr(dados['promotion_last_5years'])
dados['last_evaluation'].corr(dados['promotion_last_5years'])
dados['Will_Relocate'].corr(dados['promotion_last_5years'])
dados['number_project'].corr(dados['promotion_last_5years'])






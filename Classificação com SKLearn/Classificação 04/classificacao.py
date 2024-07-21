""" O projeto consiste em classificar se um carro será vendido ou não, baseado em algumas características.
    Para isso, será utilizado o algoritmo LinearSVC, que é um classificador linear. """
import graphviz
import numpy as np
import pandas as pd


from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# Carregando a base de dados
uri = "https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv"
dados = pd.read_csv(uri)
dados.head()

# Renomeando as colunas
renomear = {
    'mileage_per_year': 'milhas_por_ano',
    'model_year': 'ano_do_modelo',
    'price': 'preco',
    'sold': 'vendido'
}
dados = dados.rename(columns=renomear)
dados.head()

# Substituindo os valores da coluna 'vendido'
trocar = {
    'yes': 1,
    'no': 0
}
dados.vendido = dados.vendido.map(trocar)

# Criando a coluna 'idade_do_modelo'
ano_atual = datetime.today().year
dados['idade_do_modelo'] = ano_atual - dados.ano_do_modelo

# Criando a coluna 'km_por_ano'
dados['km_por_ano'] = dados.milhas_por_ano * 1.60934
dados.head()

# Removendo as colunas 'milhas_por_ano' e 'ano_do_modelo'
dados = dados.drop(columns=['milhas_por_ano', 'ano_do_modelo'], axis=1)

# Separando os dados de treino e teste
x = dados[['preco', 'idade_do_modelo', 'km_por_ano']]
y = dados['vendido']

SEED = 5
np.random.seed(SEED)
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

# Criando o modelo
modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

# Calculando a acurácia
acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

# Fazendo um 'chute' para comparar com o modelo
dummy_straified = DummyClassifier()
dummy_straified.fit(treino_x, treino_y)
acuracia = dummy_straified.score(teste_x, teste_y) * 100

print("A acurácia do dummy foi %.2f%%" % acuracia)

# Fazendo um 'chute' com base na proporção dos dados
dummy_mostfrequent = DummyClassifier()
dummy_mostfrequent.fit(treino_x, treino_y)
acuracia = dummy_mostfrequent.score(teste_x, teste_y) * 100

print("A acurácia do dummy mostfrequent foi %.2f%%" % acuracia)


# Separando os dados de treino e teste
raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)

scaler = StandardScaler()
scaler.fit(raw_treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

modelo = SVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia utilizando SVC foi %.2f%%" % acuracia)

#utilizando decision tree
raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)

modelo = DecisionTreeClassifier(max_depth=3)
modelo.fit(raw_treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia utilizando DecisionTree foi %.2f%%" % acuracia)

#visualizando a árvore de decisão
dot_data = export_graphviz(modelo, out_file='tree.dot',filled = True, rounded = True, feature_names = x.columns, class_names = ['não', 'sim'])
grafico = graphviz.Source(dot_data)
grafico.view()
"""Usando e validando com cross validation"""

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


def imprime_resultados(results):
  media = results['test_score'].mean()
  desvio_padrao = results['test_score'].std()
  print(f"Acurácia média: {media * 100:.2f}")
  print(f"Intervalo de confiança: {((media - 2 * desvio_padrao) * 100):.2f} a {(media + 2 * desvio_padrao) * 100:.2f}")

# Carregando os dados
uri = "https://gist.githubusercontent.com/guilhermesilveira/e99a526b2e7ccc6c3b70f53db43a87d2/raw/1605fc74aa778066bf2e6695e24d53cf65f2f447/machine-learning-carros-simulacao.csv"
dados = pd.read_csv(uri).drop(columns=["Unnamed: 0"], axis=1)
dados.head()

# Separando os dados
dados = pd.read_csv(uri).drop(columns=["Unnamed: 0"], axis=1)

# Separando os dados
x = dados[["preco", "idade_do_modelo", "km_por_ano"]]
y = dados["vendido"]

# Treinando o modelo
SEED = 301
np.random.seed(SEED)

# Usando cross validation
cv = KFold(n_splits = 10, shuffle=True)

modelo = DecisionTreeClassifier(max_depth=2)
results = cross_validate(modelo, x, y, cv = cv, return_train_score=False)
imprime_resultados(results)

#Gerando uma nova coluna
np.random.seed(SEED)

dados['modelo'] = dados.idade_do_modelo + np.random.randint(-2, 3, size = 10000)
dados.modelo_aleatorio = dados.modelo + abs(dados.modelo.min()) + 1

# Cross validation com standard scaler
scaler = StandardScaler()

scaler.fit(x)
treino_x_escalado = scaler.transform(x)
teste_x_escalado = scaler.transform(x)

modelo = SVC()
modelo.fit(treino_x_escalado, y)
previsoes = modelo.predict(teste_x_escalado)

acuracia = accuracy_score(y, previsoes) * 100
print(f"A acurácia foi {acuracia:.2f}")

# Cross validate com agrupamento com SVC
np.random.seed(SEED)

cv = GroupKFold(n_splits = 10)
modelo = SVC()
results = cross_validate(modelo, x, y, cv = cv, groups = dados.modelo, return_train_score=False)
imprime_resultados(results)

# Utilizando pipeline
np.random.seed(SEED)

scaler = StandardScaler()
modelo = SVC()

pipeline = Pipeline([('transformacao', scaler), ('estimador', modelo)])

cv = GroupKFold(n_splits = 10)
results = cross_validate(pipeline, x, y, cv = cv, groups = dados.modelo, return_train_score=False)
imprime_resultados(results)
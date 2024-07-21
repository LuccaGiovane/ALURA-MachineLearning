"""Usando e validando com cross validation"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, KFold
from sklearn.tree import DecisionTreeClassifier


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
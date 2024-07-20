import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
dados = pd.read_csv(uri)
dados.head()

# Renomeando as colunas
mapa = {
    'expected_hours': 'horas_esperadas',
    'price': 'preco',
    "unfinished": "nao_finalizado"
}
dados = dados.rename(columns = mapa)

# Trocando os valores da coluna 'nao_finalizado'
troca = {
    0: 1,
    1: 0
}
dados['finalizado'] = dados.nao_finalizado.map(troca)

# Visualizando os dados
sns.scatterplot(x = 'horas_esperadas', y = 'preco', hue = 'finalizado', data = dados)
sns.relplot(x = 'horas_esperadas', y = 'preco', hue = 'finalizado', col = 'finalizado', data = dados)

# Separando os dados de treino e teste
x = dados[['horas_esperadas', 'preco']]
y = dados['finalizado']

SEED = 20

# Treinando o modelo
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25, stratify = y)

model = LinearSVC()
model.fit(treino_x, treino_y)
previsoes = model.predict(teste_x)

taxa_acerto = accuracy_score(teste_y, previsoes) * 100
print("Taxa de acerto: %.2f" % taxa_acerto)

# Comparando os dados
previsoes_de_base = np.ones(540)
taxa_acerto = accuracy_score(teste_y, previsoes_de_base) * 100
print("Taxa de acerto do algoritmo de baseline: %.2f" % taxa_acerto)

x_min = teste_x.horas_esperadas.min()
x_max = teste_x.horas_esperadas.max()
y_min = teste_x.preco.min()
y_max = teste_x.preco.max()

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]

Z = model.predict(pontos)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha = 0.3)
plt.scatter(teste_x.horas_esperadas, teste_x.preco, c = teste_y, s = 1)
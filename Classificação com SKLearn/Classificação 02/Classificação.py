import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 20

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"

# Lendo os dados
dados = pd.read_csv(uri)

mapa = {
    "home": "principal",
    "how_it_works": "como_funciona",
    "contact": "contato",
    "bought": "comprou"
}
dados.rename(columns = mapa, inplace = True)

x = dados[['principal', 'como_funciona', 'contato']]
y = dados[['comprou']]

# Separando os dados de treino e teste
dados.shape
treino_x = x[:75]
treino_y = y[:75]

teste_x = x[75:]
teste_y = y[75:]

# Treinando o modelo
model = LinearSVC()
model.fit(treino_x, treino_y)

# Testando a taxa de acerto
previsoes = model.predict(teste_x)

taxa_acerto = accuracy_score(teste_y, previsoes)
print("Taxa de acerto: %.2f" % (taxa_acerto * 100))

# Separando os dados de treino e teste com train_test_split
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state = SEED,
                                                        test_size = 0.25, stratify = y)
model = LinearSVC()
model.fit(treino_x, treino_y)

previsoes = model.predict(teste_x)

taxa_acerto = accuracy_score(teste_y, previsoes)
print("Taxa de acerto: %.2f" % (taxa_acerto * 100))

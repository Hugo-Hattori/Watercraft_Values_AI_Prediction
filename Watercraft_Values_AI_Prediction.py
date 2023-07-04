import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


#Extração/Obtenção de Dados
tabela = pd.read_csv("barcos_ref.csv")
display(tabela)

#Tratamento de Dados
print(tabela.info())

#Análise Exploratória
correlacao = tabela.corr()[["Preco"]]
display(correlacao)
sns.heatmap(correlacao, annot=True, cmap="Greens")
plt.show()

#Modelagem
y = tabela["Preco"]
x = tabela.drop("Preco", axis=1)
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.3, random_state = 1)

#Treinamento da Inteligência Artificial
modelo_regressaoLinear = LinearRegression()
modelo_arvoreDecisao = RandomForestRegressor()
modelo_regressaoLinear.fit(x_treino, y_treino)
modelo_arvoreDecisao.fit(x_treino, y_treino)

#Interpretação de Resultados
previsao_regressaoLinear = modelo_regressaoLinear.predict(x_teste)
previsao_arvoreDecisao = modelo_arvoreDecisao.predict(x_teste)
print(r2_score(y_teste, previsao_regressaoLinear))
print(r2_score(y_teste, previsao_arvoreDecisao))

#Comparando Regressão Linear com "y_teste"
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["RegressaoLinear"] = previsao_regressaoLinear
sns.lineplot(data=tabela_auxiliar)
plt.show()

#Comparando Árvore de Decisão com "y_teste"
tabela_auxiliar2 = pd.DataFrame()
tabela_auxiliar2["y_teste"] = y_teste
tabela_auxiliar2["ArvoreDecisao"] = previsao_arvoreDecisao
sns.lineplot(data=tabela_auxiliar2)
plt.show()

#Utilizando a IA treinada com novos dados
tabela_nova = pd.read_csv("novos_barcos.csv")
display(tabela_nova)
nova_previsao = modelo_arvoreDecisao.predict(tabela_nova)
print(nova_previsao)
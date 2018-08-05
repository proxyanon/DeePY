from pynn import *

# idade, renda, sexo, estadocivil, filhos
# masculino = 1, feminino = 0
# casado = 0, solteiro 1

# dataset simples
entradas = array([
				 [18.0, 0.0, 1.0, 0.0, 0.0],
				 [22.0, 1210.0, 1.0, 0.0, 0.0],
				 [31.0, 2500.0, 0.0, 1.0, 1.0],
				 [45.0, 1000.0, 0.0, 0.0, 0.0],
				 [56.0, 1250.0, 1.0, 1.0, 3.0],
				 [24.0, 2500.0, 0.0, 1.0, 0.0],
				 [21.0, 980.0, 1.0, 0.0, 0.0],
				 [20.0, 980.0, 0.0, 0.0, 0.0]
				]) / 10000

# saidas baseadas no dataset
saidas = array([
				 [0],
				 [1],
				 [1],
				 [0],
				 [1],
				 [1],
				 [0],
				 [0]
				])

# entrada arbitraria
entrada = array([19.0, 300.0, 1.0, 0.0, 0.0]) / 10000

# entradas para o teste do treino
entrada_teste = array([19.0, 980.0, 1.0, 0.0, 0.0]) / 10000 # => mais proximo de 0.000 possivel
entrada_teste_2 = array([33.0, 1210.0, 0.0, 1.0, 1.0]) / 10000 # => mais proximo de 1.0 possivel

# instanciamento da rede e treino usando a sigmoid
nn = PyNN(n_camadas=3, n_entradas=5, n_hidden=10, n_saida=1, activation='sigmoid')
#nn.train(entradas=entradas, saidas=saidas, entrada=entrada, saida=array([0.0001]), eta=1)

# execucao dos pesos treinados
print entrada_teste * 10000, '=>', nn.run(entrada=entrada_teste, act='sigmoid')
print entrada_teste_2 * 10000, '=>', nn.run(entrada=entrada_teste_2, act='sigmoid')

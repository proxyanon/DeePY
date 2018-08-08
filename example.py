from pynn import *

# idade, renda, sexo, estadocivil, filhos
# masculino = 1, feminino = 0
# casado = 0, solteiro 1

entradas = array([
				 [18.0, 0.0, 1.0, 0.0, 0.0],
				 [22.0, 1210.0, 1.0, 0.0, 0.0],
				 [31.0, 2500.0, 0.0, 1.0, 1.0],
				 [45.0, 1000.0, 0.0, 0.0, 0.0],
				 [56.0, 1250.0, 1.0, 1.0, 3.0],
				 [24.0, 2500.0, 0.0, 1.0, 0.0],
				 [21.0, 980.0, 1.0, 0.0, 0.0],
				 [20.0, 980.0, 0.0, 0.0, 0.0]
				], dtype=float) / 10000

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

entrada_teste = array([
					 [40.0, 2000.0, 1.0, 1.0, 2.0],
					 [33.0, 1578.0, 0.0, 0.0, 0.0],
					 [26.0, 1000.0, 0.0, 0.0, 0.0],
					 [30.0, 980.0, 1.0, 0.0, 1.0],
					], dtype=float) / 10000

saida_teste = array([
					 [1.0],
					 [1.0],
					 [0.0],
					 [0.0]
					])

entrada_desconhecida = array([
						 [19.0, 550.0, 1.0, 0.0, 0.0], # 0
						 [56.0, 2300.0, 0.0, 1.0, 3.0], # 1
						 [33.0, 1800.0, 1.0, 0.0, 1.0], # 1
						 [22.0, 2000.0, 1.0, 1.0, 0.0], # 1
						 [20.0, 900.0, 0.0, 0.0, 0.0], # 0
						 [48.0, 1100.0, 1.0, 0.0, 0.0], # 0
						 [70.0, 2500.0, 0.0, 0.0, 1.0] # 1
						]) / 10000

nn = PyNN(n_camadas=3, n_entradas=5, n_hidden=5, n_saida=1, eta=1, activation='sigmoid')

for x, entrada in enumerate(entrada_teste):
	gen_name = 'generation_{}/'.format(x)
	nn.train(
		 entradas=entradas, 
		 saidas=saidas, 
		 entrada=entrada, 
		 saida=saida_teste[x], 
		 create_generation=True, 
		 gen_name=gen_name, 
		 use_epochs=True, 
		 epochs=150000,
		 autodel=True
		)

for entrada in entrada_desconhecida:
	print entrada * 10000, '=>', nn.run(entrada=entrada, path='generations/{}'.format(gen_name))

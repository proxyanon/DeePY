from pynn import *

''' Entradas genericas '''
entradas = array([
			 [1, 0, 1],
             [0, 1, 1],
             [1, 1, 1],
             [1, 0, 0]
            ])

''' Saidas levando em consideracao o primeiro numero da entrada '''                
saidas = array([
			 [1],
			 [0],
			 [1],
			 [1]
			])

''' Entrada generica '''
entrada = array([0, 0, 0])

nn = PyNN(3, 50, 1)
print nn.train(entradas=entradas, saidas=saidas, entrada=entrada, epochs=10000, eta=0.5)

#-*-coding: utf8-*-

''' Neural Network Implementation '''

# @author Daniel Victor Freire Feitosa
# @license GPL 3.0
# @editor Sublime Text 3
# @copyrigths by Daniel Victor Freire Feitosa - 2018

from numpy import exp, dot, array, random

''' Funcao de ativacao sigmoid '''
def sigmoid(x):
	return 1 / (1 + exp(-x))

''' Derivada da funcao de ativacao '''
def sigmoid_derivate(x):
	return x * (1 - x)


''' Implementacao simples, redes => entradasx4xsaidas '''
class PyNN:

	def __init__(self, inputSize, hiddenSize, outputSize):

		self.inputSize = inputSize # quantidade de entradas
		self.hiddenSize = hiddenSize # quantidade de neurononios na camada hidden
		self.outputSize = outputSize # quantidade da camda de saida

		random.seed(1)

		''' pesos (no maximo 4 camadas na hidden) '''
		self.pesos_1 = 2 * random.random((self.inputSize, self.hiddenSize)) - 1 # pesos da camada de entrada para a hidden
		self.pesos_2 = 2 * random.random((self.hiddenSize, self.hiddenSize)) - 1 # pesos da camada hidden para a hidden
		self.pesos_3 = 2 * random.random((self.hiddenSize, self.hiddenSize)) - 1 # pesos da camada hidden para a hidden
		self.pesos_4 = 2 * random.random((self.hiddenSize, self.outputSize)) - 1 # pesos da camada hidden para a saida

	''' feedfoward = entradas * pesos [ z(I, W) = Iij * Wij ] '''
	def feedfoward(self, entradas):
		
		''' perceptron '''
		self.layer1 = sigmoid(dot(entradas, self.pesos_1))
		self.layer2 = sigmoid(dot(self.layer1, self.pesos_2))
		self.layer3 = sigmoid(dot(self.layer2, self.pesos_3))
		layer4 = sigmoid(dot(self.layer3, self.pesos_4))
		
		return layer4

	''' backprop = targets - output '''
	def backpropagation(self, entradas, saidas, eta): # backpropagation

		layer4 = self.feedfoward(entradas) # saida

		''' 
			e = delta[camada_anterior] * pesos
			delta = e * derivada(camada)
		'''

		''' calculos de erros e deltas '''
		layer4_error = saidas - layer4
		layer4_delta = layer4_error * sigmoid_derivate(layer4)

		layer3_error = layer4_delta.dot(self.pesos_4.T)
		layer3_delta = layer3_error * sigmoid_derivate(self.layer3)

		layer2_error = layer3_delta.dot(self.pesos_3.T)
		layer2_delta = layer2_error * sigmoid_derivate(self.layer2)

		layer1_error = layer2_delta.dot(self.pesos_2.T)
		layer1_delta = layer1_error * sigmoid_derivate(self.layer1)

		''' ajuste de pesos '''
		self.pesos_4 += self.layer3.T.dot(layer4_delta) * eta
		self.pesos_3 += self.layer2.T.dot(layer3_delta) * eta
		self.pesos_2 += self.layer1.T.dot(layer2_delta) * eta
		self.pesos_1 += entradas.T.dot(layer1_delta) * eta

	def train(self, entradas, saidas, entrada, epochs=10000, eta=1.0):

		for i in xrange(epochs): # treino
			self.backpropagation(entradas, saidas, eta) # faz o backprop para ajuste dos pesos

		return self.feedfoward(entrada) # faz o feedfoward da entrada

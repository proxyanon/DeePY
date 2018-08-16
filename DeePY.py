'''

	@author Daniel Victor Freire Feitosa
	@version 5.0.21
	@editor Sublime Text 3

	@license GPL 3.0
	@copyrights by Daniel Victor Freire Feitosa

'''

from sys import stdout, exit

try:
	from tqdm import tqdm
except ImportError:
	print("O modulo tqmd nao foi encontrado execute o comando: python -m pip install tqdm")
	exit()

try:
	import numpy as np, json
except ImportError:
	print("O modulo numpy nao foi encontrado execute o comando: python -m pip install numpy")
	exit()

__version_date__ = '11/08/2018'
__version__ = '5.0.1'
__author__ = 'Daniel Victor Freire Feitosa'
__github__ = 'https://github.com/proxyanon/'

np.seterr(all='ignore') # sem warnings de overflow ...

''' funcoes de ativacao dos neuronios '''
def _activation(x, act='sigmoid'):
	if act == 'sigmoid':
		return 1 / (1 + np.exp(-x))
	elif act == 'tanh':
		return np.tanh(x)
	elif act == 'relu':
		return x * (x > 0)

''' funcoes de derivadas dos neuronios '''
def _derivate(x, act='sigmoid'):
	if act == 'sigmoid':
		return x * (1 - x)
	elif act == 'tanh':
		return 1 - x * x
	elif act == 'relu':
		return 1 * (x > 0)

''' Classe para criacao das camadas da rede '''
class Layer:

	def __init__(self, random_constant=True):

		self.lrs = []
		self.random_constant = random_constant

		if self.random_constant:
			np.random.seed(1)

	def create(self, _type, sizes, activation='sigmoid'):

		if _type == 'inputs' or _type == 'input':
			l = 2 * np.random.random((sizes[0], sizes[1])) - 1
		elif _type == 'hiddens' or _type == 'hidden':
			l = 2 * np.random.random((sizes[0], sizes[1])) - 1
		elif _type == 'outputs' or _type == 'output':
			l = 2 * np.random.random((sizes[0], sizes[1])) - 1

		self.lrs.append([{'layer': l, 'activation': activation}])

	def layers(self):
		return self.lrs


''' Rede neural by @DanielFreire00 '''
class Network:

	def __init__(self, layers, eta=1, verbose=True):

		self.weights = layers
		self.eta = eta
		self.verbose = verbose
		self.errs = [0]
		self.weights_size = len(self.weights)

	''' Foward => Ziw = (i1 * w1) + (i2 * w2) ... '''
	def foward(self, inputs):

		layers = []
		for x in xrange(self.weights_size):
			if x == 0:
				l = _activation(x=np.dot(inputs, self.weights[x][0]['layer']), act=self.weights[x][0]['activation'])
				layers.append([{'layer': l, 'activation': self.weights[x][0]['activation']}])
			else:
				l = _activation(x=np.dot(layers[x-1][0]['layer'], self.weights[x][0]['layer']), act=self.weights[x][0]['activation'])
				layers.append([{'layer': l, 'activation': self.weights[x][0]['activation']}])

		return layers

	''' Foward unico para o predict com os pesos carregados '''
	def single_foward(self, inputs, weights):

		layers = []
		for x in xrange(len(weights)):
			if x == 0:
				l = _activation(x=np.dot(inputs, weights[x][0]['weight']), act=weights[x][0]['activation'])
				layers.append([{'layer': l, 'activation': weights[x][0]['activation']}])
			else:
				l = _activation(x=np.dot(layers[x-1][0]['layer'], weights[x][0]['weight']), act=weights[x][0]['activation'])
				layers.append([{'layer': l, 'activation': weights[x][0]['activation']}])

		return layers

	''' Backprop => weights += layer * (error * deriv(layer)) '''
	def backward(self, inputs, outputs):

		layers = self.foward(inputs=inputs)
		deltas = []


		for x in xrange(self.weights_size):
			if x == 0:
				e = outputs - layers[len(layers)-1][0]['layer']
				d = e * _derivate(x=layers[self.weights_size-1][0]['layer'], act=layers[self.weights_size-1][0]['activation'])
				deltas.append(d)
			else:
				e = deltas[x-1].dot(self.weights[self.weights_size-x][0]['layer'].T)
				d = e * _derivate(x=layers[(self.weights_size-x)-1][0]['layer'], act=layers[(self.weights_size-x)-1][0]['activation'])
				deltas.append(d)

		for x in xrange(self.weights_size):
			if x != 0:
				self.weights[x][0]['layer'] += layers[x-1][0]['layer'].T.dot(deltas[(self.weights_size-x)-1]) * self.eta
			else:
				self.weights[x][0]['layer'] += inputs.T.dot(deltas[len(deltas)-1]) * self.eta

	''' Treino da rede '''
	def train(self, inputs, outputs, epochs=1000, weights=[]):
		
		if len(weights) > 0:
			self.weights = weights

		try:
			if self.verbose:
				for e in tqdm(range(epochs)):
					self.backward(inputs=inputs, outputs=outputs)
			else:
				for e in xramge(epochs):
					self.backward(inputs=inputs, outputs=outputs)	
		except KeyboardInterrupt:
			pass

		if self.verbose:
			print "\n"

	def train_with_test(self, inputs, outputs):

		self.backward(inputs=inputs, outputs=outputs)
		out = self.foward(inputs=inputs)
		flag = False

		print out[len(out)-1][0]['layer'], outputs
		exit()

		while not flag:
			for x, value in enumerate(out[len(out)-1][0]['layer']):
				out[len(out)-1][0]['layer'][x][0] = round(out[len(out)-1][0]['layer'][x][0])
			
			if out[len(out)-1][0]['layer'] == outputs:
				flag = True

			self.backward(inputs=inputs, outputs=outputs)
			out = self.foward(inputs=inputs)


	''' Previsao de resultados baseado nos pesos passados ou pos treino '''
	def predict(self, inpt, weights=[]):
		
		if len(weights) > 0:
			ret = self.single_foward(inputs=inpt, weights=weights)
		ret = self.foward(inputs=inpt)

		return ret[len(ret)-1][0]

	''' Salva o modelo da rede '''
	def savemodel(self, model_name='model.json'):
		
		model = [{'weights_size': self.weights_size, 'eta': self.eta, 'verbose': self.verbose}]
		handle = open(model_name, 'w')
		handle.write(json.dumps(model))
		handle.close()

	''' Carrega o modelo da rede '''
	def loadmodel(self, model_name='model.json'):
		
		handle = open(model_name, 'r')
		read = handle.read()
		handle.close()

		return json.loads(read.strip())

	''' Salva os pesos da rede '''
	def saveweights(self, weights_name='weights.json', weights=[], model=[]):
		
		handle = open(weights_name, 'w')

		for i, weight in enumerate(self.weights):
			w_name = 'weight_{}'.format(i)
			w_string = ''
			for z, w in enumerate(weight[0]['layer']):
				string = ''
				for x, value in enumerate(w):
					string += ';{}'.format(value)
				w_string += string
			js = [{w_name: w_string[1:len(w_string)], 'shape0': weight[0]['layer'].shape[0], 'shape1': weight[0]['layer'].shape[1], 'activation': weight[0]['activation']}]
			handle.write(json.dumps(js)+"\n")

		handle.close()

	''' Carrega os pesos da rede '''
	def loadweights(self, weights_name='weights.json'):
		
		handle = open(weights_name, 'r')
		weights = []

		for x, row in enumerate(handle):
			array = json.loads(row)[0]
			w_name = 'weight_{}'.format(x)
			w = array[w_name].split(';')
			weight = np.array(w, dtype=np.longdouble).reshape(array['shape0'], array['shape1'])
			weights.append([{'weight': weight, 'activation': array['activation']}])

		handle.close()

		return weights

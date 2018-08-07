#-*-coding: utf8-*-

''' 
	
	@author Daniel Victor Frerie Feitosa
	@version 4.0.0
	@editor Sublime Text 3
	
	@copyrights by Daniel Victor Frerie Feitosa - 2018
	@license GPL 3.0

	Twitter: @DanielFreire00
	Youtube: ProXy Sec
	
	feito com muita prometazina e cafe ...
'''
from os import path, makedirs, remove, walk, _exit as exit
try:
	from numpy import dot, array, random, exp, savetxt, tanh, sort
except ImportError:
	print("O modulo numpy nao foi encontrado execute o comando: python -m pip install numpy")
	exit(1)


''' funcoes de ativacao dos neuronios '''
def _activation(x, act='sigmoid'):
	if act == 'sigmoid':
		return 1 / (1 + exp(-x))
	elif act == 'tanh':
		return tanh(x)
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

''' foward unico para a execucao da rede '''
def single_foward(entradas, pesos, act='sigmoid'):
	layers = []
	for x in xrange(len(pesos)):
		if x == 0:
			l = _activation(x=dot(entradas, pesos[x]), act=act)
			layers.append(l)
		else:
			l = _activation(x=dot(layers[x-1], pesos[x]), act=act)
			layers.append(l)
	return layers

''' Rede neural by @DanielFreire00 '''
class PyNN:

	def __init__(self, n_camadas, n_entradas, n_hidden, n_saida, activation='sigmoid'):

		self.n_camadas = n_camadas
		self.n_entradas = n_entradas
		self.n_hidden = n_hidden
		self.n_saida = n_saida
		self.activation = activation

		random.seed(1)

		self.pesos = []
		self.pesos.append(2*random.random((self.n_entradas, self.n_hidden))-1)
		
		for x in xrange(n_camadas):
			self.pesos.append(2*random.random((self.n_hidden, self.n_hidden))-1)
		
		self.pesos.append(2*random.random((self.n_hidden, self.n_saida))-1)
		self.n_pesos = len(self.pesos)

		if path.exists('weights/') == False:
			makedirs('weights')

	''' Foward => Ziw = (i1 * w1) + (i2 * w2) ... '''
	def feedfoward(self, entradas):

		layers = []
		for x in xrange(self.n_pesos):
			if x == 0:
				l = _activation(x=dot(entradas, self.pesos[x]), act=self.activation)
				layers.append(l)
			else:
				l = _activation(x=dot(layers[x-1], self.pesos[x]), act=self.activation)
				layers.append(l)

		return layers

	''' Backprop => pesos += layer * (error * deriv(layer)) '''
	def backpropagation(self, entradas, saidas, eta=1):

		layers = self.feedfoward(entradas)
		deltas = []

		for x in xrange(self.n_pesos):
			if x == 0:
				e = saidas - layers[len(layers)-1]
				d = e * _derivate(x=layers[self.n_pesos-1], act=self.activation)
				deltas.append(d)
			else:
				e = deltas[x-1].dot(self.pesos[self.n_pesos-x].T)
				d = e * _derivate(x=layers[(self.n_pesos-x)-1], act=self.activation)
				deltas.append(d)

		for x in xrange(self.n_pesos):
			if x != 0:
				self.pesos[x] += layers[x-1].T.dot(deltas[(self.n_pesos-x)-1]) * eta
			else:
				self.pesos[x] += entradas.T.dot(deltas[len(deltas)-1]) * eta

	''' Carrega os pesos a partir dos pesos salvos pela funcao saveweights '''
	def loadweights(self, path='weights/', delimiter=';'):
		
		arrays = []
		pesos_ajustados = []
		indexs = []
		shapes = []

		x = 0

		for paths,dirs,files in walk(path):
			for file in files:
				
				try:
					handle = open(paths+'/'+file, 'r')
					read = handle.read()
					handle.close()
				except IOError:
					print("Nao foi possivel carregar o seguite csv => {filename}".format(filename=file))

				try:
					if '[network]' in read:
						read = read.replace(read[0:len('[network]')+1], '')
						spl = read.split(delimiter)
						arr = array(spl, dtype=int)
						shapes.append(arr)
					else:
						spl = read.split(delimiter)
						arr = array(spl[1:len(spl)], dtype=float)
						pesos_ajustados.append({int(file.split('-')[1].split('.')[0]): arr})
						indexs.append(int(file.split('-')[1].split('.')[0]))
						x += 1
				except:
					print("Pesos ou configuracoes da rede invalidos ...")
					exit(1)

		pesos_ajustados = sort(pesos_ajustados)
		
		for x, value in enumerate(pesos_ajustados):
			pesos_ajustados[x] = pesos_ajustados[x][x]

		arrays.append(pesos_ajustados)		
		arrays.append(shapes)

		return arrays

	''' Roda a rede neural com os pesos treinados, e com uma entrada qualquer '''
	def run(self, entrada, path='weights/', act='sigmoid', delimiter=';'):

		arrays = self.loadweights(path=path, delimiter=delimiter)
		pesos_ajustados = arrays[0]
		shapes = arrays[1][0]

		for x, value in enumerate(pesos_ajustados):
			try:
				if x == 0:
					pesos_ajustados[x] = value.reshape(shapes[1], shapes[2])
				elif x < (int(shapes[4])-1) and x > 0:
					pesos_ajustados[x] = value.reshape(shapes[2], shapes[2])
				else:
					pesos_ajustados[x] = value.reshape(shapes[2], shapes[3])
			except:
				print("Valores do arquivo de arquitetura da rede estao ivalidos !")
				exit(1)

		output = single_foward(entradas=entrada, pesos=pesos_ajustados, act=act)
		return output[len(output)-1]

	''' Salva os pesos na pasta weights/ '''
	def saveweights(self, path='weights/', delimiter=';'):
		for i, ps in enumerate(self.pesos):
			for z, p in enumerate(ps):
				for c, x in enumerate(p):
					try:
						handle = open('{path}/weight-{num}.csv'.format(path=path, num=i), 'a')
						handle.write(delimiter+str(x))
						handle.close()
					except IOError:
						print("Nao foi possivel salvar o peso => {peso_err}".format(peso_err=i))

		try:
			handle = open('{path}/network-arch.csv'.format(path=path), 'w')
			handle.write('[network]\n')
			handle.write('{}{delimiter}{}{delimiter}{}{delimiter}{}{delimiter}{}'.format(self.n_camadas, self.n_entradas, self.n_hidden, self.n_saida, self.n_pesos, delimiter=delimiter))
			handle.close()
		except IOError:
			print("Nao foi possivel salvar a configuracao da rede ...")

	''' Treina a rede com as entradas, saidas, entrada epecifica, saida esperada depois salva os pesos do treino '''
	def train(self, entradas, saidas, entrada, saida, eta=1, path='weights/', savenetwork=True, autodel=False, banners=True, delimiter=';'):

		if not autodel:
			q = raw_input('Continuar vai remover todos os pesos salvos ... ')
			if len(q) == '' or q.upper() != 'N':
				for paths,dirs,files in walk('weights/'):
					for file in files:
						remove(paths+'/'+file)
		else:
			for paths,dirs,files in walk('weights/'):
				for file in files:
					remove(paths+'/'+file)

		epochs = 0
		try:
			try:
				dc = len(str(saida[0]).split('.')[1])+2
			except:
				dc = int(str(saida[0]).split('-')[1])+2
			o = self.feedfoward(entrada)
			outputs = o[len(o)-1]
			while float(str(outputs[0])[0:dc]) != saida[0]:
				self.backpropagation(entradas, saidas, eta)
				outputs = self.feedfoward(entrada)[len(self.feedfoward(entrada))-1]
				epochs += 1
				if banners:
					print outputs, saida, epochs
		except KeyboardInterrupt:
			pass

		if savenetwork:
			if banners:
				print("\n[+] Salvando pesos ...")
			try:
				self.saveweights(path=path, delimiter=delimiter)
			except KeyboardInterrupt:
				exit(1)

		#return outputs[len(outputs)-1]
	
	def read_csv(self, filename, delimiter=';'):
		read = []
		handle = open(filename, 'r')
		
		for r in handle:
			r = r.strip()
			arr = r.split(delimiter)
			read.append(array(arr, dtype=float))

		handle.close()
		return array(read)

	def format_dataset(self, read, delimiter=';', outlast=True, iout=0):
		inputs = []
		outputs = []
		ret = []

		if outlast:
			for arr in read:
				outputs.append(array([arr[len(arr)-1]], dtype=float))
			iout = len(arr)-1
		else:
			for arr in read:
				outputs.append(array(arr[iout], dtype=float))

		if outlast:
			for arr in read:
				arr = arr[0:len(arr)-1]
				inputs.append(array(arr, dtype=float))
		else:
			for arr in read:
				arr = arr[iout:len(arr)-1]
				inputs.append(array(arr, dtype=float))

		ret.append(array(outputs, dtype=float))
		ret.append(array(inputs, dtype=float))

		return ret

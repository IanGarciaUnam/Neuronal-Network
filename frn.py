import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.random import uniform as u
class Perceptron():

	def __init__(self, n_layer=1, n_neurons=4):
		"""
		Constructor de la clase
		Inicializa pesos aleatorios en la red
		
		Parámetros
		---------
		n_capas:int
			Número de capas ocultas en la red
		n_neuronas:int
			Número de neuronas por capa, por default el valor es 4

		"""
		self.w_in=u(low=-1, high=1, size=(2, n_neurons))
		self.b_in=u(low=-1, high=1, size= n_neurons)# TODO u [-1, 1] n_neurons

		#Definicion de pesos y biases en la capa de entrada
		self.w_hidden=u(low=-3, high=3, size=(n_layer, n_neurons,n_neurons))# TODO u [-3, 3] (n_layers, n_neurons, n_neurons)
		self.b_hidden = u(low=-1, high=1, size=(n_layer, n_neurons))# TODO u [-1, 1] 1
		#Definimos pesos y biases en las capas de salida
		self.w_out = u(low=-1, high=1, size=(n_neurons,1)) # TODO u [-1, 1] (n_neurons, 1)
		self.b_out = u(low=-1, high=1, size=1)# TODO u [-1, 1] 1


	def activate_layer(self, y_in, w,b):
		"""
		Calcula el producto interno y aplica función de activación
		por caja


		Parámetros
		----------
		y_in:ndarray
			Vector con datos de entrada.
		w: ndarray
			Vector de pesos
		b: ndarray
			El vector de biases.
		
		Return
		------
		a: ndarray
			Vector de activación resultante
		"""
		z=np.dot(y_in,w)+b
		s=1./(1. + np.exp(-z))#FUncion sigmoide
		return s

	def feedforward(self, y_in):
		"""
		Calcula el producto punto y apliaca la funcion de activaci
		cion por capa
		"""
		y=self.activate_layer(y_in, self.w_in, self.b_in)
		for i in range(self.w_hidden.shape[0]):
			y=self.activate_layer(y, self.w_hidden[i], self.b_hidden[i])
		output = self.activate_layer(y, self.w_out, self.b_out)
		return output

	def visualize(self, grid_size=50, colormap='viridis', c_reverse=False):
		"""
		Visualiza el mapeo de la red neuronal en un plano 2D

		Parámetros
		----------
		grid_size:int
			tamaño de la rejilla
		colormap: str
			Color a utilizar
		c_reverse:bool
			Flag para especificar si se invierte el mapa de color
			El valor default es False
		"""
		mpl.rcParams['figure.dpi'] = 300
		# Creamos una rejilla
		x = np.linspace(-0.5, 0.5, grid_size)
		y = np.linspace(-0.5, 0.5, grid_size)
		xx, yy = np.meshgrid(x, y)
		# Para todas las coordenadas (x, y) en la rejilla,
        # hacemos una única lista con los pares de puntos
		x_flat = xx.flatten()
		y_flat = yy.flatten()
		y_in = zip(x_flat, y_flat)
		y_in = np.array(list(y_in))

        # Hacemos feedforward con la red
		y_out = self.feedforward(y_in)# TODO Apply feedforward on y_in
        # Redimensionamos a la rejilla
		y_out_2d = np.reshape(y_out, (grid_size, grid_size))
		if c_reverse:
			cmap = plt.cm.get_cmap(colormap)
			cmap = cmap.reversed()
		else:
			cmap = colormap
        # Graficamos los resultados de la red
		plt.figure(figsize=(10, 10))
		plt.axes([0, 0, 1, 1])
		plt.imshow(y_out_2d,extent=[-0.5, 0.5, -0.5, 0.5],interpolation='nearest',cmap=cmap)
		plt.axis(False)
		plt.show()

y_in=np.array([0.8, 0.2])
n=Perceptron(n_layer=10, n_neurons=100)
n.feedforward(y_in)
#n.visualize()
n.visualize(grid_size=750)
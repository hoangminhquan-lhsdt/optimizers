import numpy as np
from test_functions import Rosenbrock as F

class Momentum:
	def __init__(self, lr, momentum):
		self.func = F()
		self.lr = lr
		self.momentum = momentum
		self.v = np.zeros(2)
	def step(self, x, y):
		g_t = self.func.df(x, y)
		self.v = self.momentum*self.v + self.lr*g_t
		return (x - self.v[0], y - self.v[1])
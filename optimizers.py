import numpy as np

class Momentum:
	def __init__(self, lr, momentum):
		self.lr = lr
		self.momentum = momentum
		self.v = np.zeros(2)
	def step(self, x, y):
		g_t = df(x, y)
		self.v = self.momentum*self.v + self.lr*g_t
		return (x - self.v[0], y - self.v[1])
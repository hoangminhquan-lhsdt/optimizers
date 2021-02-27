import numpy as np

class Rosenbrock:
	def __init__(self, a=1, b=5):
		self.minima = [1, 1]
		self.a = 1
		self.b = 5
	def f(self, x, y):
		return (self.a - x)**2 + self.b*(y - x**2)**2
	def df(self, x, y):
		return np.array((2*(2*self.b*(x**3) - 2*self.b*x*y + x - self.a), 2*self.b*(y - x**2)))


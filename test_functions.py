import numpy as np

class Rosenbrock:
	def __init__(self):
		pass
	def f(self, x, y):
		return (1 - x**2) + 100*(y - x**2)**2
	def df(self, x, y):
		return np.array((2*(10*(x**3) - 10*x*y + x - 1), 10*(y - x**2)))
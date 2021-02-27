import numpy as np

class Rosenbrock:
	def __init__(self, a=1, b=5):
		self.name = 'Rosenbrock'
		self.minima = [[1, 1]]
		self.a = 1
		self.b = 5
	def f(self, x, y):
		return (self.a - x)**2 + self.b*(y - x**2)**2
	def df(self, x, y):
		return np.array((2*(2*self.b*(x**3) - 2*self.b*x*y + x - self.a), 2*self.b*(y - x**2)))

class Himmelblau:
	def __init__(self):
		self.name = "Himmelblau"
		self.minima = [[3.0, 2.0], [-2.805118, 3.131312],
					   [-3.779310, -3.283186], [3.584428, -1.848126]]
	def f(self, x, y):
		return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
	def df(self, x, y):
		return np.array((
			2*(2*x*(x**2 + y - 11) + x + y**2 - 7),
			2*(x**2 + 2*y*(x + y**2 - 7) + y - 11)
		))

class Booth:
	def __init__(self):
		self.name = 'Booth'
		self.minima = [[1, 3]]
	def f(self, x, y):
		return (x + 2*y - 7)**2 + (2*x + y - 5)**2
	def df(self, x, y):
		return np.array((10*x + 8*y - 34, 8*x + 10*y - 38))
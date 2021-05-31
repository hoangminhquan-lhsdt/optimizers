import numpy as np
np.seterr(all='raise')

class Rosenbrock:
	def __init__(self, a=1, b=5):
		self.name = 'Rosenbrock'
		self.minima = [[1, 1]]
		self.a = 1
		self.b = 5
		self.x0 = -2
		self.y0 = -1
	def f(self, x, y):
		value = 0
		try:
			value = (self.a - x)**2 + self.b*(y - x**2)**2
		except FloatingPointError:
			# print(f'Overflow at x={x}, y={y}')
			raise FloatingPointError
		finally:
			return value
	def df(self, x, y):
		value = np.zeros(2)
		try:
			value = np.array((2*(2*self.b*(x**3) - 2*self.b*x*y + x - self.a), 2*self.b*(y - x**2)))
		except FloatingPointError:
			# print(f'Overflow at x={x}, y={y}')
			raise FloatingPointError
		else:
			return value

class Himmelblau:
	def __init__(self):
		self.name = "Himmelblau"
		self.minima = [[3.0, 2.0], [-2.805118, 3.131312],
					   [-3.779310, -3.283186], [3.584428, -1.848126]]
		self.x0 = 4.5
		self.y0 = 4.5
	def f(self, x, y):
		value = 0
		try:
			value = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
		except FloatingPointError:
			# print(f'Overflow at x={x}, y={y}')
			raise FloatingPointError
		else:
			return value
	def df(self, x, y):
		value = np.zeros(2)
		try:
			value = np.array((
				2*(2*x*(x**2 + y - 11) + x + y**2 - 7),
				2*(x**2 + 2*y*(x + y**2 - 7) + y - 11)
			))
		except FloatingPointError:
			# print(f'Overflow at x={x}, y={y}')
			raise FloatingPointError
		else:
			return value

class Booth:
	def __init__(self):
		self.name = 'Booth'
		self.minima = [[1, 3]]
		self.x0 = -1
		self.y0 = -1
	def f(self, x, y):
		value = 0
		try:
			value = (x + 2*y - 7)**2 + (2*x + y - 5)**2
		except FloatingPointError:
			# print(f'f(x, y) overflow at x={x}, y={y}')
			raise FloatingPointError
		else:
			return value
	def df(self, x, y):
		value = np.zeros(2)
		try:
			value = np.array((10*x + 8*y - 34, 8*x + 10*y - 38))
		except FloatingPointError:
			# print(f'df(x, y) overflow at x={x}, y={y}')
			raise FloatingPointError
		else:
			return value
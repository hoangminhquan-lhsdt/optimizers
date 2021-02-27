import numpy as np
from test_functions import Rosenbrock as F

class Momentum:
	'''Implementation of SGD with Momentum from "On the Momentum Term in Gradient Descent Learning Algorithms"

	Arguments:
		lr (float, optional): learning rate
		momentum (float, optional): momentum term
	'''
	def __init__(self, a=0.001, momentum=0.9):
		self.name = 'SGD with Momentum'
		self.func = F()
		self.a = a
		self.momentum = momentum
		self.v = np.zeros(2)
	def step(self, x):
		g_t = self.func.df(x[0], x[1])
		self.v = self.momentum*self.v + self.a*g_t
		return x - self.v

class Adam:
	'''Implementation of Adam from "Adam: A Method for Stochastic Optimization"

	Arguments:
		a (float, optional): alpha
		b1 (float, optional): beta1
		b2 (float, optional): beta2
		eps (float, optional): epsilon
	'''
	def __init__(self, a=0.001, b1=0.9, b2=0.999, eps=1e-8):
		self.name = 'Adam'
		self.F = F()
		self.a = a
		self.b1 = b1
		self.b2 = b2
		self.eps = eps
		self.m = np.zeros(2) # 1st moment vector (mean)
		self.v = np.zeros(2) # 2nd moment vector (variance)
		self.t = 0 # timestep
	
	def step(self, x):
		self.t += 1
		g_t = self.F.df(x[0], x[1])
		self.m = self.b1*self.m + (1-self.b1)*g_t
		self.v = self.b2*self.v + (1-self.b2)*g_t*g_t
		m_hat = self.m / (1 - self.b1**self.t)
		v_hat = self.v / (1 - self.b2**self.t)
		# return (x - (self.a*m_hat[0] / (np.sqrt(v_hat[0]) + self.eps)), y - (self.a*m_hat[1] / (np.sqrt(v_hat[1]) + self.eps)))
		return x - self.a*m_hat/ (np.sqrt(v_hat) + self.eps)
		# step_size = self.a * np.sqrt(1 - self.b2 ** self.t) / (1 - self.b1**self.t)
		# return x - step_size*m_hat / (np.sqrt(v_hat) + self.eps)

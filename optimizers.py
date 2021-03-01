import numpy as np


class SGD:
	def __init__(self, F, lr=0.001):
		self.name = 'Stochastic Gradient Descent'
		self.func = F
		self.lr = lr
	def step(self, x, y):
		g = self.func.df(x, y)
		return x - self.lr*g[0], y - self.lr*g[1]


class Momentum:
	'''Implementation of SGD with Momentum from "On the Momentum Term in Gradient Descent Learning Algorithms".

	Arguments:
	----------
	lr : float, optional
		learning rate (default: 0.001)
	momentum : float, optional
		momentum term (default: 0.9)
	'''
	def __init__(self, F, lr=0.001, momentum=0.9):
		self.name = 'SGD with Momentum'
		self.func = F
		self.lr = lr
		self.momentum = momentum
		self.v = np.zeros(2)
	def step(self, x, y):
		g_t = self.func.df(x, y)
		self.v = self.momentum*self.v + self.lr*g_t
		return (x - self.v[0], y - self.v[1])


class AdaGrad:
	'''Implementation of AdaGrad from "Adaptive Subgradient Methods for
Online Learning and Stochastic Optimization".

	Arguments:
	----------
	lr : float, optional
		learning rate (default: 0.01)
	lr_decay : float, optional
		learning rate decay (default: 0)
	weight_decay : float, optional
		weight decay (default: 0)
	eps : float, optional
		epsilon
	'''
	def __init__(self, F, lr=0.01, eps=1e-10):
		self.name = 'Adagrad'
		self.func = F
		self.lr = lr
		self.eps = eps
		self.sq_grad = np.zeros(2)  # sum of squared gradient
	def step(self, x, y):
		g = self.func.df(x, y)
		self.sq_grad += g**2
		v = self.lr * g / (np.sqrt(self.sq_grad)+ self.eps)
		return x - v[0], y - v[1]


class Adam:
	'''Implementation of Adam from "Adam: A Method for Stochastic Optimization".

	Arguments:
	----------
	lr   : float, optional
		alpha, learning rate
	b1  : float, optional
		beta2, exponential decay for moving averages of gradient
	b2  : float, optional
		beta1, exponential decay for moving averages of squared gradient
	eps : float, optional
		epsilon
	'''
	def __init__(self, F, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
		self.name = 'Adam'
		self.F = F
		self.lr = lr
		self.b1 = b1
		self.b2 = b2
		self.eps = eps
		self.m = np.zeros(2) # 1st moment vector (mean)
		self.v = np.zeros(2) # 2nd moment vector (variance)
		self.t = 0 # timestep
	
	def step(self, x, y):
		self.t += 1
		g_t = self.F.df(x, y)
		self.m = self.b1*self.m + (1-self.b1)*g_t
		self.v = self.b2*self.v + (1-self.b2)*g_t*g_t
		m_hat = self.m / (1 - self.b1**self.t)
		v_hat = self.v / (1 - self.b2**self.t)
		v = self.lr*m_hat / (np.sqrt(v_hat) + self.eps)
		return (x - v[0], y - v[1])
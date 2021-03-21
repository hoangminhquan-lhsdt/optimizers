import numpy as np


class SGD:
	def __init__(self, F, lr=0.001):
		self.name = 'SGD'
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
		self.name = 'Momentum'
		self.func = F
		self.lr = lr
		self.momentum = momentum
		self.v = np.zeros(2)
	def step(self, x, y):
		g_t = self.func.df(x, y)
		self.v = self.momentum*self.v + self.lr*g_t
		return (x - self.v[0], y - self.v[1])


class AdaGrad:
	'''Implementation of AdaGrad from "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization".

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
		self.name = 'AdaGrad'
		self.func = F
		self.lr = lr
		self.eps = eps
		self.sq_grad = np.zeros(2)  # sum of squared gradient
	def step(self, x, y):
		g = self.func.df(x, y)
		self.sq_grad += g**2
		v = self.lr * g / (np.sqrt(self.sq_grad)+ self.eps)
		return x - v[0], y - v[1]


class AdaDelta:
	'''Implementation of AdaDelta from "AdaDelta: An Adaptive Learning Rate Method".
	lr  : float, optional
		alpha, learning rate
	eps : float, optional
		epsilon
	'''
	def __init__(self, F, lr=0.95, eps=1e-6):
		self.name = 'AdaDelta'
		self.F = F
		self.lr = lr
		self.eps = eps
		self.E_gt = np.zeros(2)
		self.E_delta = np.zeros(2)
	def step(self, x, y):
		gt = self.F.df(x, y)
		self.E_gt = self.lr*self.E_gt + (1 - self.lr)*(gt**2)
		RMS_gt = np.sqrt(self.E_gt + self.eps)
		RMS_delta = np.sqrt(self.E_delta + self.eps)
		delta = -(RMS_delta / RMS_gt)*gt
		self.E_delta = self.lr*self.E_delta + (1 - self.lr)*(delta**2)
		return (x + delta[0], y + delta[1])


class RMSprop:
	'''Implementation of RMSprop from "Neural Networks for Machine Learning" Lecture 6b.
	Arguments:
	----------
	lr    : float, optional
		learning rate
	gamma : float, optional
		momentum
	eps   : float, optional
		epsilon
	'''
	def __init__(self, F, lr=0.001, gamma=0.9, eps=1e-7):
		self.name = 'Adam'
		self.F = F
		self.lr = lr
		self.gamma = gamma
		self.eps = eps
		self.E_g2 = np.zeros(2) # mean of squared gradient
	def step(self, x, y):
		g = self.F.df(x, y)
		self.E_g2 = gamma*self.E_g2 + (1 - gamma)*(g**2)
		delta = self.lr * g / np.sqrt(self.E_g2 + self.eps)
		return (x - delta[0], y - delta[1])


class Adam:
	'''Implementation of Adam from "Adam: A Method for Stochastic Optimization".

	Arguments:
	----------
	lr  : float, optional
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

class AMSGrad(Adam):
	def __init__(self, F, lr = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-8):
		super().__init__(F,lr,b1,b2,eps)
		self.v_max = np.zeros(2)
		self.name = 'AMSGrad'
	def step(self,x,y):
		self.t += 1
		g_t = self.F.df(x,y)

		# self.b1 = self.b1 / self.t #b1 decay [OPTIONAL]
		
		self.m = self.b1 * self.m + (1-self.b1)*g_t
		self.v = self.b1 * self.v + (1-self.b2)*g_t*g_t

		self.v_max = np.array((max(self.v[0],self.v_max[0]),max(self.v[1],self.v_max[1])))

		# alpha = lr/ np.sqrt(self.t) # learning rate decay [OPTIONAL]
		step_size = self.lr * self.m / (np.sqrt(self.v_max) + self.eps)

		return (x - step_size[0], y - step_size[1])
class NAdam(Adam):
	def __init__(self, F, lr = 0.001, b1 = 0.975, b2 = 0.999, eps = 1e-8):
		super().__init__(F,lr,b1,b2,eps)
		self.name = "NAdam"
	def step(self,x,y):
		self.t += 1
		g_t = self.F.df(x,y)

		self.m = self.b1 * self.m + (1-self.b1) * g_t
		self.v = self.b2 * self.v + (1-self.b2) * g_t * g_t

		m_hat = self.m / (1 - self.b1 ** (self.t + 1))
		v_hat = self.b2 * self.v / (1 - self.b2 ** self.t)

		nes_m = (self.b1 * m_hat) + ((1 - self.b1) * g_t / (1 - self.b1 ** self.t))
		step_size = self.lr * nes_m / (np.sqrt(v_hat) + self.eps)

		return (x - step_size[0], y - step_size[1])

class NAMSGrad(Adam):
	def __init__(self, F, lr = 0.001, b1 = 0.975, b2 = 0.999, eps = 1e-8):
		super().__init__(F,lr,b1,b2,eps)
		self.max_v = np.zeros(2)
		self.name = 'NAMSGrad'
	def step(self,x,y):
		self.t += 1
		g_t = self.F.df(x,y)

		self.m = self.b1 * self.m + (1 - self.b1) * g_t
		self.v = self.b2 * self.v + (1 - self.b2) * g_t * g_t

		m_hat = self.m / (1 - self.b1 ** (self.t + 1))
		v_hat = self.b2 * self.v / (1 - self.b2 ** self.t)

		nes_m = (self.b1 * m_hat) + ((1 - self.b1) * g_t / (1 - self.b1 ** self.t))
		self.max_v =  np.array((max(self.v[0],self.max_v[0]),max(self.v[1],self.max_v[1])))

		step_size = self.lr * nes_m / (np.sqrt(self.max_v) + self.eps)

		return (x - step_size[0], y - step_size[1])

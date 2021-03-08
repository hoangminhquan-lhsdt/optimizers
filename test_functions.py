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

class linear_func:
	def __init__(self, a,b, k):
		self.minima = -1;
		self.a = a
		self.b = b
		self.k = k
		self.b1 = 0.9
		self.b2 = 0.99#1/(1 + self.C**2)
		self.t = 0
		self.m = 0
		self.v = 0
		self.max_v = 0
	def f(self,x,t):
		if t % self.k == 1:
			return self.a*x
		return -self.b*x
	def fs(self,x,t):
		return [self.f(xs,ts) for xs,ts in zip(x,t)]
	def df(self,x,t):
		if t % self.k == 1:
			return self.a
		return -self.b
	def Adam(self,x,lr,eps,amsgrad = False):
		self.t += 1
		g_t = self.df(x,self.t-1)
		self.m = self.b1 * self.m + (1-self.b1)*g_t
		self.v = self.b2 * self.v + (1-self.b2)*g_t*g_t

		m_hat = self.m / (1 - self.b1**self.t)

		# if amsgrad:
		# 	self.max_v = max(self.v,self.max_v)
		# 	denom = np.sqrt(self.max_v) / (np.sqrt(1 - self.b2**self.t) + eps)
		# else:
		# 	denom = np.sqrt(self.v) / (np.sqrt(1 - self.b2 **self.t) + eps)
		if amsgrad:
			self.max_v = max(self.v,self.max_v)
			v_hat = self.max_v / (1 - self.b2**self.t)
			return x - lr*m_hat/ (np.sqrt(v_hat) + eps)

		v_hat = self.v / (1 - self.b2**self.t)
		return x - lr*m_hat/ (np.sqrt(v_hat) + eps)

		# return x - lr*self.m / denom
	def AMSGrad(self,x,lr,eps):
		self.t += 1
		g_t = self.df(x,self.t)
		self.m = self.b1*self.m + (1 - self.b1)*g_t
		self.v = self.b2*self.v + (1- self.b2)*g_t*g_t

		# step_size = lr / (1 - self.b1**self.t)
		# v_hat = self.v / (np.sqrt(self.v) + eps)
		self.max_v = max(self.v,self.max_v)
		denom = np.sqrt(self.max_v) / (np.sqrt(1 - self.b2**self.t) + eps)

		return x - lr* self.m / denom
	def reset(self):
		self.t = 0
		self.m = 0
		self.v = 0
		self.max_v = 0




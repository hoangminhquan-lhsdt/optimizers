import torch
import _functional as F
from optimizer import Optimizer


class my_SGD(Optimizer):
	r"""Implementation of Stochastic Gradient Descent

	Args:
		lr (float): learning rate (default: 1e-3)
	"""
	def __init__(self, params, lr=1e-3):
		if lr <= 0:
			raise ValueError(f'Invalid learning rate: {lr}')
		defaults = dict(lr=lr)
		super(my_SGD, self).__init__(params, defaults)

	def __setstate__(self, state):
		super(my_SGD, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault()

	@torch.no_grad()
	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:
			params_with_grad = []
			grads = []

			for p in group['params']:
				if p.grad is not None:
					params_with_grad.append(p)
					grads.append(p.grad)

					state = self.state[p]

			F.my_sgd(params_with_grad,
					 grads,
					 group['lr'])

		return loss
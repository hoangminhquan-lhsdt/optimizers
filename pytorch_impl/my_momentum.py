import torch
import _functional as F
from optimizer import Optimizer

class my_Momentum(Optimizer):
	r"""Implementation of Stochastic Gradient Descent with Momentum

	Args:
		lr (float): learning rate (default: 1e-3)
		momentum (float): momentum term (default: 0.9)
	"""
	def __init__(self, params, lr=1e-3, momentum=0.9):
		if lr <= 0:
			raise ValueError(f'Invalid learning rate: {lr}')
		if momentum <= 0:
			raise ValueError(f'Invalid momentum term: {momentum}')

		defaults = dict(lr=lr, momentum=momentum)
		super(my_Momentum, self).__init__(params, defaults)

	def __setstate__(self, state):
		super(my_Momentum, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('momentum', 0.9)

	@torch.no_grad()
	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:
			params_with_grad = []
			grads = []
			momentum_list = []

			for p in group['params']:
				if p.grad is not None:
					params_with_grad.append(p)
					grads.append(p.grad)

					state = self.state[p]
					if 'momentum' not in state:
						momentum_list.append(None)
					else:
						momentum_list.append(state['momentum'])

			F.my_momentum(params_with_grad,
						  grads,
						  momentum_list,
						  group['lr'],
						  group['momentum'])

		#update momentum in state
		for p, momentum in zip(params_with_grad, momentum_list):
			state = self.state[p]
			state['momentum'] = momentum

		return loss
import torch
from ._functional import my_rmsprop
from .optimizer import Optimizer

class my_RMSprop(Optimizer):
	r"""Implementation of RMSprop

	Args:
		lr (float): learning rate (default: 1e-3)
		gamma (float): momentum term (default: 0.9)
		eps (float): smoothing value (default: 1e-8)
	"""
	def __init__(self, params, lr=1e-3, gamma=0.9, eps=1e-8):
		if lr <= 0:
			raise ValueError(f'Invalid learning rate: {lr}')
		if gamma <= 0:
			raise ValueError(f'Invalid gamma: {gamma}')
		if eps <= 0:
			raise ValueError(f'Invalid epsilon value: {eps}')

		defaults = dict(lr=lr, gamma=gamma, eps=eps)
		super(my_RMSprop, self).__init__(params, defaults)

		for group in self.param_groups:
			for p in group['params']:
				state = self.state[p]
				state['E_g2'] = torch.zeros_like(p, memory_format=torch.preserve_format)  # initialize mean of squared grad with 0

	@torch.no_grad()
	def step(self, closure=None):
		"""Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:
			params_with_grad = []
			grads = []
			state_Eg2 = []

			for p in group['params']:
				if p.grad is not None:
					params_with_grad.append(p)
					grads.append(p.grad)
					state = self.state[p]
					state_Eg2.append(state['E_g2'])
			
			my_rmsprop(params_with_grad,
						 grads,
						 state_Eg2,
						 group['lr'],
						 group['gamma'],
						 group['eps'])

		return loss
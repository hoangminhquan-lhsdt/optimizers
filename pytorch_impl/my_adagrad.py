import torch
import _functional as F
from optimizer import Optimizer

class my_Adagrad(Optimizer):
	r"""Implementation of AdaGrad
	
	Args:
		lr (float): learning rate (default: 1e-3)
		eps (float): smoothing factor (default: 1e-10)
	"""
	def __init__(self, params, lr=1e-3, eps=1e-10):
		if lr <= 0:
			raise ValueError(f'Invalid learning rate: {lr}')
		if eps <= 0:
			raise ValueError(f'Invalid epsilon value: {eps}')
		
		defaults = dict(lr=lr, eps=eps)
		super(my_Adagrad, self).__init__(params, defaults)

		for group in self.param_groups:
			for p in group['params']:
				state = self.state[p]
				state['sum'] = torch.zeros_like(p, memory_format=torch.preserve_format)  # initialize G (tensor) with 0 values

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
			state_sums = []

			for p in group['params']:
				if p.grad is not None:
					params_with_grad.append(p)
					grads.append(p.grad)
					state = self.state[p]
					state_sums.append(state['sum'])

			F.my_adagrad(params_with_grad,
						 grads,
						 state_sums,
						 group['lr'],
						 group['eps'])

		return loss
import torch
from ._functional import my_adadelta
from torch.optim import Optimizer

class my_Adadelta(Optimizer):
	r"""Implementation of Adadelta

	Args:
		rho (float): decay rate (default: 0.95)
		eps (float): smoothing value (default: 1e-6)
	"""
	def __init__(self, params, rho=0.95, eps=1e-6):
		if rho <= 0:
			raise ValueError(f'Invalid rho value: {rho}')
		if eps <= 0:
			raise ValueError(f'Invalid epsilon value: {eps}')

		defaults = dict(rho=rho, eps=eps)
		super(my_Adadelta, self).__init__(params, defaults)

		for group in self.param_groups:
			for p in group['params']:
				state = self.state[p]
				state['E_g2'] = torch.zeros_like(p, memory_format=torch.preserve_format)
				state['E_delta2'] = torch.zeros_like(p, memory_format=torch.preserve_format)

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
			state_Edelta2 = []

			for p in group['params']:
				if p.grad is not None:
					params_with_grad.append(p)
					grads.append(p.grad)
					state = self.state[p]

					state_Eg2.append(state['E_g2'])
					state_Edelta2.append(state['E_delta2'])

			my_adadelta(params_with_grad,
						 grads,
						 state_Eg2,
						 state_Edelta2,
						 group['rho'],
						 group['eps'])

		return loss
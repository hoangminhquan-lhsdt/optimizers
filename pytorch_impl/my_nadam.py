import torch
from ._functional import my_nadam
from torch.optim import Optimizer

class my_Nadam(Optimizer):
	r"""Implementation of Nadam

	Args:
		lr (float): learning rate (default: 1e-3)
		b1 (float): beta1 (default: 0.9)
		b2 (float): beta2 (default: 0.999)
		eps (float): smoothing value (default: 1e-8)
	"""
	def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
		if lr <= 0:
			raise ValueError(f'Invalid learning rate: {lr}')
		if beta1 <= 0:
			raise ValueError(f'Invalid beta1: {beta1}')
		if beta2 <= 0:
			raise ValueError(f'Invalid beta2: {beta2}')
		if eps <= 0:
			raise ValueError(f'Invalid epsilon value: {eps}')

		defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
		super(my_Nadam, self).__init__(params, defaults)

		for group in self.param_groups:
			for p in group['params']:
				state = self.state[p]
				state['step'] = 0
				state['M'] = torch.zeros_like(p, memory_format=torch.preserve_format)
				state['V'] = torch.zeros_like(p, memory_format=torch.preserve_format)

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
			state_Ms = []
			state_Vs = []
			state_steps = []

			for p in group['params']:
				if p.grad is not None:
					params_with_grad.append(p)
					grads.append(p.grad)
					state = self.state[p]

					state_Ms.append(state['M'])
					state_Vs.append(state['V'])
					state['step'] += 1
					state_steps.append(state['step'])

			my_nadam(params_with_grad,
					   grads,
					   state_Ms,
					   state_Vs,
					   state_steps,
					   group['lr'],
					   group['beta1'],
					   group['beta2'],
					   group['eps'])
		
		return loss
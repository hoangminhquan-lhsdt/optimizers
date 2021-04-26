import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import animation

from optimizers import SGD, Momentum, AdaGrad, AdaDelta, RMSprop, Adam, AMSGrad, Nadam
# put desired test function name here
from test_functions import Booth as Func


# Optimizers
# lr = df.loc[Opt(func, 0).name, func.name]
optimizers = {
	'SGD': SGD(Func(), lr=0.003),
	'Momentum': Momentum(Func(), lr=0.003, momentum=0.9),
	'AdaGrad': AdaGrad(Func(), lr=0.5, eps=1e-3),
	'AdaDelta': AdaDelta(Func(), lr=0.99, eps=1e-3),
	'RMSprop': RMSprop(Func(), lr=0.05, gamma=0.99, eps=1e-8),
	'Adam': Adam(Func(), lr=0.5, b1=0.9, b2=0.99, eps=1e-8),
	'AMSGrad': AMSGrad(Func(), lr=0.5, b1=0.9, b2=0.88, eps=1e-8),
	'Nadam': Nadam(Func(), lr=0.5, b1=0.9, b2=0.99, eps=1e-8),
}


opts = ['SGD', 'Momentum', 'AdaGrad', 'AdaDelta', 'RMSprop', 'Adam', 'AMSGrad', 'Nadam']
# opts = ['AdaDelta']
for o in opts:
	fig = plt.figure()
	func = Func()
	if func.name == "Rosenbrock":
		X = np.arange(-4,3.1,0.1)
		Y = np.arange(-2,4.1,0.1)
		ax = plt.axes(xlim=(-4, 3), ylim=(-2, 4))
		x0 = -2
		y0 = -1
	elif func.name == "Himmelblau":
		X = np.arange(-5,5.1,0.1)
		Y = np.arange(-5,5.1,0.1)
		ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5))
		x0 = 4.5
		y0 = 4.5
	elif func.name == 'Booth':
		X = np.arange(-10,10.1,0.1)
		Y = np.arange(-10,10.1,0.1)
		ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
		x0 = -9
		y0 = -9
	X, Y = np.meshgrid(X, Y)
	Z = func.f(X, Y)

	# Contour of test function
	ax.contourf(X, Y, Z, 100, cmap='viridis')
	ax.plot(func.minima[0][0], func.minima[0][1], 'ro', label='Minima')
	for i in range(1, len(func.minima)):
		ax.plot(func.minima[i][0], func.minima[i][1], 'ro')


	def init():
		point.set_data([], [])
		step_text.set_text('')
		value_text.set_text('')
		grad_text.set_text('')
		return point, step_text, value_text, grad_text
	def animate(i):
		global p
		point.set_data(p[i-1][0], p[i-1][1])
		step_text.set_text(f'step: {i}')
		value_text.set_text(f'z: {func.f(p[i-1][0], p[i-1][1]):.3f}')
		df = func.df(p[i-1][0], p[i-1][1])
		grad_text.set_text(f'grad: ({df[0]:.3f}, {df[1]:.3f})')
		return point, step_text, value_text


	opt = optimizers[o]
	p = []
	point, = ax.plot([], [], 'yo', label=opt.name)
	step_text = ax.text(0.02, 0.95, '', c='white', transform=ax.transAxes)
	value_text = ax.text(0.02, 0.91, '', c='white', transform=ax.transAxes)
	grad_text = ax.text(0.02, 0.87, '', c='white', transform=ax.transAxes)
	p.append(opt.step(x0, y0))
	N = 300
	for i in range(1, N):
		p.append(opt.step(p[i-1][0], p[i-1][1]))

	plt.legend(loc='lower right')
	try:
		os.mkdir(f'test_gifs/{func.name}/')
	except Exception as e:
		pass
	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=N, blit=True)
	print(f'Writing to test_gifs/'+func.name+'/'+opt.name+'.gif')
	anim.save('test_gifs/'+func.name+'/'+opt.name+'.gif', writer='imagemagick', fps=60)

	# ax.plot(p[-1][0],p[-1][1],'yo')


	# plt.show()

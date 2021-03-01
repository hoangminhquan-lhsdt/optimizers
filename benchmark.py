import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('seaborn')

from test_functions import Booth, Himmelblau, Rosenbrock
from optimizers import SGD, Momentum, AdaGrad, Adam

figs, (ax1, ax2) = plt.subplots(1, 2)

def plot_value(F, optimizers, fx, p, N, show=True):
	global figs, ax1, ax2
	for i in range(len(optimizer)):
		p.append([])
		p[i].append(optimizer[i].step(F.x0, F.y0))

		fx.append([])  
		fx[i].append(F.f(p[i][0][0], p[i][0][1])) # get function value at X_0
		for t in range(1, N):
			p[i].append(optimizer[i].step(p[i][t-1][0], p[i][t-1][1]))
			fx[i].append(F.f(p[i][t][0], p[i][t][1]))
			
	for i in range(len(optimizer)):
		ax1.plot(np.arange(0, N), fx[i], label=optimizer[i].name)
	ax1.set_title('Optimization algorithms on ' + F.name + ' function')
	ax1.set_xlabel('iteration')
	ax1.set_ylabel('f(x)')
	if show:
		plt.legend()
		plt.show()

def visualize_steps(F, optimizers, fx, p, N):
	global figs, ax1, ax2
	xmin = 0
	xmax = 0
	ymin = 0
	ymax = 0
	if F.name == "Rosenbrock":
		xmin = -3
		xmax = 3
		ymin = -2
		ymax = 3
	elif F.name == "Himmelblau":
		xmin = -5
		xmax = 5
		ymin = -5
		ymax = 5
	elif F.name == 'Booth':
		xmin = -2
		xmax = 5
		ymin = -2
		ymax = 5
	else:
		raise RuntimeError('Error: Invalid test function')
	X = np.arange(xmin,xmax+0.1,0.1)
	Y = np.arange(ymin,ymax+0.1,0.1)
	X, Y = np.meshgrid(X, Y)
	Z = F.f(X, Y)

	# plotting contour map of the function
	ax2.contour(X, Y, Z, 20, cmap='viridis')
	# plotting minimas
	ax2.plot(F.minima[0][0], F.minima[0][1], 'ko', label='Global Minima')
	for i in range(1, len(F.minima)):
		ax2.plot(F.minima[i][0], F.minima[i][1], 'ko')
	
	# plotting steps
	
	prop_cycle = plt.rcParams['axes.prop_cycle']
	colors = prop_cycle.by_key()['color']
	linestyles = [(0, (1, 1)), (0, (5, 10)), (0, (3, 5, 1, 5, 1, 5)), '-.']
	for i in range(len(optimizers)):
		ax2.plot((F.x0, p[i][0][0]), (F.y0, p[i][0][1]), c=colors[i], ls=linestyles[i], lw=1.5, label=optimizers[i].name)
		for j in range(N-1):
			ax2.plot((p[i][j][0], p[i][j+1][0]), (p[i][j][1], p[i][j+1][1]), c=colors[i], ls=linestyles[i], lw=1.5)
	plt.legend()
	plt.show()


if __name__ == '__main__':
	F = Rosenbrock()
	optimizer = [SGD(F), Momentum(F), AdaGrad(F, lr=0.5), Adam(F, lr=0.5)]

	N = 200
	fx = []
	p = []
	plot_value(F, optimizer, fx, p, N, show=False)
	visualize_steps(F, optimizer, fx, p, N)
	
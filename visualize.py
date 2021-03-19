import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

# put desired optimizer name here
from optimizers import NAdam as Opt
# put desired test function name here
from test_functions import Rosenbrock as Func


df = pd.read_csv('lr.csv')
df = df.set_index('Unnamed: 0')
print(df)


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

# Optimizing
# lr = df.loc[Opt(func, 0).name, func.name]
lr = 0.05
opt = Opt(func, lr=lr, eps=1e-8)
p = []
point, = ax.plot([], [], 'yo', label=opt.name + f'(lr={lr:.3f})')
step_text = ax.text(0.02, 0.95, '', c='white', transform=ax.transAxes)
value_text = ax.text(0.02, 0.91, '', c='white', transform=ax.transAxes)
grad_text = ax.text(0.02, 0.87, '', c='white', transform=ax.transAxes)
p.append(opt.step(x0, y0))
N = 300
for i in range(1, N):
	p.append(opt.step(p[i-1][0], p[i-1][1]))

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

plt.legend(loc='lower right')
# anim = animation.FuncAnimation(fig, animate, init_func=init, frames=N, blit=True)
# print(f'Writing to gifs/'+func.name+'/'+opt.name+'.gif')
# anim.save('gifs/'+func.name+'/'+opt.name+'.gif', writer='imagemagick', fps=60)

ax.plot(p[-1][0],p[-1][1],'yo')


plt.show()

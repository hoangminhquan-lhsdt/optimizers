import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# put desired optimizer name here
from optimizers import Adam as Opt
# put desired test function name here
from test_functions import Rosenbrock as Func

X = np.arange(-4,3.1,0.1)
Y = np.arange(-2,4.1,0.1)
X, Y = np.meshgrid(X, Y)
func = Func()
Z = func.f(X, Y)

fig = plt.figure()
ax = plt.axes(xlim=(-4, 3), ylim=(-2, 4))
ax.contourf(X, Y, Z, 100, cmap='viridis')
ax.plot(func.minima[0], func.minima[1], 'ro', label='Global Minima')

# Optimizing Rosenbrock
opt = Opt()
p = []
point, = ax.plot([], [], 'yo', label=opt.name)
step_text = ax.text(0.02, 0.95, '', c='white', transform=ax.transAxes)
value_text = ax.text(0.02, 0.91, '', c='white', transform=ax.transAxes)
x0 = -2
y0 = -1
p.append(opt.step(x0, y0))
N = 1000
for i in range(1, N):
	p.append(opt.step(p[i-1][0], p[i-1][1]))

def init():
	point.set_data([], [])
	step_text.set_text('')
	value_text.set_text('')
	return point, step_text, value_text
def animate(i):
	global p
	point.set_data(p[i-1][0], p[i-1][1])
	step_text.set_text(f'step: {i}')
	value_text.set_text(f'z: {func.f(p[i-1][0], p[i-1][1]):.3f}')
	return point, step_text, value_text
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=N, blit=True)
plt.legend(loc='lower right')
anim.save('gifs/'+opt.name+'.gif', writer='imagemagick', fps=30)


# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from optimizers import Momentum as Opt
from test_functions import Rosenbrock as Func

X = np.arange(-2.5,2.1,0.1)
Y = np.arange(-2,3.1,0.1)
X, Y = np.meshgrid(X, Y)
func = Func()
Z = func.f(X, Y)

fig = plt.figure()
ax = plt.axes(xlim=(-2.5, 2), ylim=(-2, 3))
ax.contourf(X, Y, Z, 50)
ax.plot(func.minima[0], func.minima[1], 'ro', label='Global Minima')

plt.legend()
plt.show()
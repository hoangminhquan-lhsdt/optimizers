import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from optimizers import Momentum as Opt
from test_functions import linear_func as Func

# X = np.arange(-2.5,2.1,0.1)
# Y = np.arange(-2,3.1,0.1)
# X, Y = np.meshgrid(X, Y)
func = Func(1010,10,101)
# Z = func.f(X, Y)
# Y = func.f(X,[_ for _ in range(100)])

fig = plt.figure()
# ax = plt.axes(xlim=(-2.5, 2), ylim=(-2, 3))
ax = plt.axes()
# ax.contourf(X, Y, Z, 50)
# ax.plot(func.minima[0], func.minima[1], 'ro', label='Global Minima')

# x = [-2,-1]
# Optimer = Opt(a = 0.001)
# maxIter = 7500
# for _ in range(maxIter):
# 	x = Optimer.step(x)
# 	print(x[0],x[1],func.f(x[0],x[1]))

x = 0
maxIter = 25000
x_t = []
for t in range(maxIter):
	x = func.Adam(x,0.001,1e-8,amsgrad = True)
	# print(x)
	if x > 1:
		x = 1
	elif x < -1:
		x = -1
	# print(func.f(x,t+1))
	x_t.append(x)

# ax.plot(range(maxIter),func.fs(x_t,range(maxIter)),'yo')
ax.plot(range(maxIter),x_t,'y')
# ax.plot(0,x_t[0],'yo')
# plt.legend()
plt.show()
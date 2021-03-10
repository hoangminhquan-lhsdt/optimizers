# Optimizers
Implementations of various optimization algorithms using Python and numerical libraries.

This repository serves as the source for visualizations and evaluations used in our thesis.
- [Tasks list](#tasks-list)
- [Implementations](#implementations)
  - [1. Stochastic Gradient Descent with Momentum](#1-stochastic-gradient-descent-with-momentum)
  - [2. AdaGrad](#2-adagrad)
  - [3. AdaDelta](#3-adadelta)
  - [4. RMSProp](#4-rmsprop)
  - [5. Adam](#5-adam)
  - [6. NAdam](#6-nadam)
  - [7. AMSGrad](#7-amsgrad)
- [Limitations](#limitations)
- [Authors](#authors)

## Tasks list
- [X] Stochastic Gradient Descent with Momentum ([Ning Qian, 1999](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.57.5612&rep=rep1&type=pdf))
- [X] AdaGrad ([John Duchi et al, 2011](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf))
- [X] AdaDelta ([Matthew D. Zeiler, 2012](https://arxiv.org/pdf/1212.5701.pdf))
- [ ] RMSProp ([Geoffrey Hinton et al, 2012](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf))
- [X] Adam ([Diederik P. Kingma, Jimmy Lei Ba, 2015](https://arxiv.org/pdf/1412.6980.pdf))
- [ ] NAdam ([Timothy Dozat, 2016](https://openreview.net/pdf/OM0jvwB8jIp57ZJjtNEZ.pdf))
- [ ] AMSGrad ([Manzil Zaheer et al, 2018](https://openreview.net/pdf?id=ryQu7f-RZ))

## Implementations
View the full source code for each algorithm in [optimizers.py](https://github.com/hoangminhquan-lhsdt/optimizers/blob/main/optimizers.py)
### 1. Stochastic Gradient Descent with Momentum
```python
def step(self, x, y):
	g_t = self.func.df(x, y)
	self.v = self.momentum*self.v + self.lr*g_t
	return (x - self.v[0], y - self.v[1])
```
![SGD with Momentum](gifs/Rosenbrock/SGD%20with%20Momentum.gif)

### 2. AdaGrad
```python
def step(self, x, y):
	g = self.func.df(x, y)
	self.sq_grad += g**2
	v = self.lr * g / (np.sqrt(self.sq_grad)+ self.eps)
	return x - v[0], y - v[1]
```
![AdaGrad](gifs/Rosenbrock/AdaGrad.gif)

### 3. AdaDelta
```python
def step(self, x, y):
	gt = self.F.df(x, y)
	self.E_gt = self.lr*self.E_gt + (1 - self.lr)*(gt**2)
	RMS_gt = np.sqrt(self.E_gt + self.eps)
	RMS_delta = np.sqrt(self.E_delta + self.eps)
	delta = -(RMS_delta / RMS_gt)*gt
	self.E_delta = self.lr*self.E_delta + (1 - self.lr)*(delta**2)
	return (x + delta[0], y + delta[1])
```
![AdaDelta](gifs/Rosenbrock/AdaDelta.gif)

### 4. RMSProp

### 5. Adam
```python
def step(self, x, y):
	self.t += 1
	g_t = self.F.df(x, y)
	self.m = self.b1*self.m + (1-self.b1)*g_t
	self.v = self.b2*self.v + (1-self.b2)*g_t*g_t
	m_hat = self.m / (1 - self.b1**self.t)
	v_hat = self.v / (1 - self.b2**self.t)
	return (x - (self.a*m_hat[0] / (np.sqrt(v_hat[0]) + self.eps)), y - (self.a*m_hat[1] / (np.sqrt(v_hat[1]) + self.eps)))
```
![Adam](gifs/Rosenbrock/Adam.gif)

### 6. NAdam

### 7. AMSGrad

## Limitations
All algorithms are currently implemented in ℝ<sup>3</sup> space mainly for visualization purpose. ℝ<sup>N</sup> space implementation may be updated in the future.

## Authors
This repository is maintained and developed by Hoàng Minh Quân and Nguyễn Ngọc Lan Như, students at University of Science, VNU-HCM.

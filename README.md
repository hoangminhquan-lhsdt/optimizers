# Optimizers
Implementations of various optimization algorithms using Python and numerical libraries.

This repository serves as the source for visualizations and evaluations used in our thesis.
- [Tasks list](#tasks-list)
- [Implementations](#implementations)
  - [Stochastic Gradient Descent with Momentum](#stochastic-gradient-descent-with-momentum)
- [Authors](#authors)

## Tasks list
- [X] Stochastic Gradient Descent with Momentum ([Ning Qian, 1999](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.57.5612&rep=rep1&type=pdf))
- [ ] AdaGrad ([John Duchi et al, 2011](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf))
- [ ] AdaDelta ([Matthew D. Zeiler, 2012](https://arxiv.org/pdf/1212.5701.pdf))
- [ ] RMSProp ([Geoffrey Hinton et al, 2012](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf))
- [ ] Adam ([Diederik P. Kingma, Jimmy Lei Ba, 2015](https://arxiv.org/pdf/1412.6980.pdf))
- [ ] NAdam ([Timothy Dozat, 2016](https://openreview.net/pdf/OM0jvwB8jIp57ZJjtNEZ.pdf))
- [ ] AMSGrad ([Manzil Zaheer et al, 2018](https://openreview.net/pdf?id=ryQu7f-RZ))

## Implementations
View the full source code for each algorithm in [optimizers.py](https://github.com/hoangminhquan-lhsdt/optimizers/blob/main/optimizers.py)
### 1. Stochastic Gradient Descent with Momentum
```
def step(self, x, y):
	g_t = self.func.df(x, y)
	self.v = self.momentum*self.v + self.lr*g_t
	return (x - self.v[0], y - self.v[1])
```
![SGD with Momentum](gifs/momentum.gif)

## Authors
This repository is maintained and developed by Hoàng Minh Quân and Nguyễn Ngọc Lan Như, students at University of Science, VNU-HCM.
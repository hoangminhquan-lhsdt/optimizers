import numpy as np
import pandas as pd
import test_functions as F
import optimizers as opt

def distance(p1, p2):
	value = 0
	try:
		value = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
	except FloatingPointError as e:
		# print(f'Overflow at p1=({p1[0]},{p1[1]}, p2=({p2[0]},{p2[1]})')
		raise Exception
	else:
		return value

def converged(p, F):
	for minima in F.minima:
		if distance(p, minima) < 0.001:
			return True
	return False

if __name__ == '__main__':
	lr = 0.001
	funcs = [F.Booth(), F.Himmelblau(), F.Rosenbrock()]
	df = pd.DataFrame(index=['SGD', 'Momentum', 'AdaGrad', 'Adam','AMSGrad'], columns=['Booth', 'Himmelblau', 'Rosenbrock'])
	steps_df = df.copy()
	df.fillna(0, inplace=True)
	steps_df.fillna(100000, inplace=True)
	while lr < 1:  # test each learning rate configuration
		for func in funcs:
			opts = [opt.SGD(func, lr), opt.Momentum(func, lr), opt.AdaGrad(func, lr), opt.Adam(func, lr), opt.AMSGrad(func,lr)]
			print(f'\n{func.name} , lr={lr}:')
			for optimizer in opts:
				br = False  # break condition
				steps = 1
				p = optimizer.step(func.x0, func.y0)

				stop = False
				while not stop and not br:
					steps += 1
					try:
						stop = converged(p, func)
						p = optimizer.step(p[0], p[1])
					except Exception:
						br = True
					finally:
						if steps > 100000:
							br = True

				if br:
					print(f'{optimizer.name}: could not converge')
				else:
					print(f'{optimizer.name}: {steps} steps')
					if steps < steps_df.loc[optimizer.name, func.name]:
						steps_df.loc[optimizer.name, func.name] = steps
						df.loc[optimizer.name, func.name] = lr
		lr += 0.001
	print(steps_df)
	print(df)
	df.to_csv('lr.csv')
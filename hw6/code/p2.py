import numpy as np
import gd

X = np.array([[2, 0], [0, 1], [0, 0]])
y = np.array([[3], [2], [2]])


def fp(w):
    return 2 * X.T @ (X @ w - y)


stepsize = 0.1
maxiter = 1000000
w0 = np.array([[0.0], [0.0]])
w_traces = gd.gd_const_ss(fp, w0, stepsize=stepsize, maxiter=maxiter)


print(
    f'stepsize={stepsize}, number of iterations={len(w_traces)-1}')
print(f'optimal solution:\n{w_traces[-1]}')

sol = np.linalg.solve(X.T @ X, X.T @ y)
print(f'solution from np.linalg.solve:\n{sol}')

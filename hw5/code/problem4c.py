import cvxpy as cp
import numpy as np

t = 100

m = 3
n = 2

X = np.array([[2, 0], [0, 1], [0, 0]])
y = np.array([3, 2, 2])

w = cp.Variable(n)

constraints = [cp.norm2(w) ** 2 <= t]

obj = cp.Minimize(cp.norm2(X @ w - y) ** 2)


problem = cp.Problem(obj, constraints)
problem.solve()
print("status:", problem.status)
print("optimal value:", problem.value)
print("optimal w:", w.value)

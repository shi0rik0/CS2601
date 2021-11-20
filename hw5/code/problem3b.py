import cvxpy as cp
import numpy as np

m = 3
n = 2

A = np.array([[2, 1], [1, -3], [1, 2]])
b = np.array([5, 10, -5])

x = cp.Variable(n)

constraints = [cp.norm_inf(x) <= 1]

obj = cp.Minimize(cp.norm1(A @ x - b))


problem = cp.Problem(obj, constraints)
problem.solve()
print("status:", problem.status)
print("optimal value:", problem.value)
print("optimal var:", x.value)

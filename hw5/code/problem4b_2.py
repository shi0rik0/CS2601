import cvxpy as cp
import numpy as np

# 尝试把问题转化为 QP 再求解

t = 10

m = 3
n = 2

X = np.array([[2, 0], [0, 1], [0, 0]])
y = np.array([3, 2, 2])

w = cp.Variable(n)
u = cp.Variable(n)

constraints = [cp.sum(u) <= t,
               w <= u,
               -u <= w]

obj = cp.Minimize(cp.norm2(X @ w - y) ** 2)


problem = cp.Problem(obj, constraints)
problem.solve()
print("status:", problem.status)
print("optimal value:", problem.value)
print("optimal w:", w.value)
print("optimal u:", u.value)

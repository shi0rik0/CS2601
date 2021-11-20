import cvxpy as cp

x1 = cp.Variable()
x2 = cp.Variable()


constraints = [2*x1 + x2 >= 1,
               x1 + 3*x2 >= 1,
               x1 >= 0,
               x2 >= 0]

objs = [cp.Minimize(x1 + x2),
        cp.Minimize(-x1 - x2),
        cp.Minimize(x1),
        cp.Minimize(cp.maximum(x1, x2)),
        cp.Minimize(x1**2 + 9*x2**2)]


problems = [cp.Problem(obj, constraints) for obj in objs]
for p in problems:
    p.solve()
    print("status:", p.status)
    print("optimal value:", p.value)
    print("optimal var: x1={}, x2={}".format(x1.value, x2.value))

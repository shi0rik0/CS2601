import numpy as np

# def newton(fp, fpp, x0, tol=1e-5, maxiter=100000):
# 	"""
# 	fp: function that takes an input x and returns the gradient of f at x
# 	fpp: function that takes an input x and returns the Hessian of f at x
# 	x0: initial point
# 	tol: toleracne parameter in the stopping crieterion. Newton's method stops
# 	     when the 2-norm of the gradient is smaller than tol
# 	maxiter: maximum number of iterations

# 	This function should return a list of the sequence of approximate solutions
# 	x_k produced by each iteration
# 	"""
# 	x_traces = [np.array(x0)]
# 	x = np.array(x0)
# 	#   START OF YOUR CODE

# 	pass

# 	#	END OF YOUR CODE

# 	return x_traces

# def damped_newton(f, fp, fpp, x0, alpha=0.5, beta=0.5, tol=1e-5, maxiter=100000):
# 	"""
# 	f: function that takes an input x an returns the value of f at x
# 	fp: function that takes an input x and returns the gradient of f at x
# 	fpp: function that takes an input x and returns the Hessian of f at x
# 	x0: initial point in gradient descent
# 	alpha: parameter in Armijo's rule
# 				f(x + t * d) > f(x) + t * alpha * <f'(x), d>
# 	beta: constant factor used in stepsize reduction
# 	tol: toleracne parameter in the stopping crieterion. Here we stop
# 	     when the 2-norm of the gradient is smaller than tol
# 	maxiter: maximum number of iterations in gradient descent.

# 	This function should return a list of the sequence of approximate solutions
# 	x_k produced by each iteration and the total number of iterations in the inner loop
# 	"""
# 	x_traces = [np.array(x0)]
# 	stepsize_traces = []
# 	tot_num_iter = 0

# 	x = np.array(x0)

# 	for it in range(maxiter):
# 		#   START OF YOUR CODE

# 		pass

# 		#	END OF YOUR CODE
# 		x_traces.append(np.array(x))
# 		stepsize_traces.append(stepsize)

# 	return x_traces, stepsize_traces, tot_num_iter


def newton_eq(f, fp, fpp, x0, A, b, initial_stepsize=1.0, alpha=0.5, beta=0.5, tol=1e-8, maxiter=100000):
    """
    f: function that takes an input x an returns the value of f at x
    fp: function that takes an input x and returns the gradient of f at x
    fpp: function that takes an input x and returns the Hessian of f at x
    A, b: constraint A x = b
    x0: initial feasible point
    initial_stepsize: initial stepsize used in backtracking line search
    alpha: parameter in Armijo's rule 
                            f(x + t * d) > f(x) + t * alpha * f(x) @ d
    beta: constant factor used in stepsize reduction
    tol: toleracne parameter in the stopping crieterion. Gradient descent stops 
         when the 2-norm of the Newton direction is smaller than tol
    maxiter: maximum number of iterations in outer loop of damped Newton's method.

    This function should return a list of the iterates x_k
    """
    x_traces = [np.array(x0)]

    m = len(b)
    x = np.array(x0)

    for it in range(maxiter):
        A2 = np.block([[fpp(x), A.T], [A, np.zeros((m, m))]])
        b2 = np.concatenate([-fp(x), np.zeros((m,))])
        d = np.linalg.solve(A2, b2)[:-m]
        t = initial_stepsize
        while f(x + t * d) > f(x) + alpha * t * np.dot(fp(x), d):
            t *= beta
        x += t * d
        x_traces.append(x.copy())
        if np.linalg.norm(d) < tol:
            break
    return x_traces


def newton_eq_2(f, fp, fpp, x0, A, b, initial_stepsize=1.0, alpha=0.5, beta=0.5, tol=1e-8, maxiter=100000):
    """
    f: function that takes an input x an returns the value of f at x
    fp: function that takes an input x and returns the gradient of f at x
    fpp: function that takes an input x and returns the Hessian of f at x
    A, b: constraint A x = b
    x0: initial feasible point
    initial_stepsize: initial stepsize used in backtracking line search
    alpha: parameter in Armijo's rule 
                            f(x + t * d) > f(x) + t * alpha * f(x) @ d
    beta: constant factor used in stepsize reduction
    tol: toleracne parameter in the stopping crieterion. Gradient descent stops 
         when the 2-norm of the Newton direction is smaller than tol
    maxiter: maximum number of iterations in outer loop of damped Newton's method.

    This function should return a list of the iterates x_k
    """
    x_traces = [np.array(x0)]

    m = len(b)
    x = np.array(x0)

    for it in range(maxiter):
        A2 = np.block([[fpp(x), A.T], [A, np.zeros((m, m))]])
        b2 = np.concatenate([-fp(x), np.zeros((m,))])
        d = np.linalg.solve(A2, b2)[:-m]
        t = initial_stepsize
        # 防止出界
        while not (x + t * d > 0).all() or f(x + t * d) > f(x) + alpha * t * np.dot(fp(x), d):
            t *= beta
        x += t * d
        x_traces.append(x.copy())
        if np.linalg.norm(d) < tol:
            break
    return x_traces

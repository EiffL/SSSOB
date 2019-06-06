import numpy as np
import math

def eigen_max(operator, shape, num_iters=10):
    """
    Compute maximum eigenvalue of the operator by the power method.
    shape: input shape for the probe operator, usually: [batch_size, d]
    num_iters: number of power method iterations to compute the maximum
        eigenval, at each call to the model, the eigenvector is cached.
    """
    def _l2normalize(v, eps=1e-12):
        return v / np.linalg.norm(v)

    u = np.ones(shape)
    for i in range(num_iters):
        u = _l2normalize(operator(u))

    result = u.T.dot(operator(u))

    return result

def ChebyshevCoefficients(operator, a, b, n):
    """
    Returns Chebyshev coefficients up to order `n` for the function `operator`
    defined in the interval [a, b]

    WARNING: Returns 2x the coefficient c0
    TODO: Figure out how to divide c0 by 2 :-/
    """
    bma = 0.5 * (b - a)
    bpa = 0.5 * (b + a)

    xk = np.cos(math.pi * (np.arange(n+1) + 0.5)/(n + 1)) * bma + bpa
    f = operator(xk)

    i = np.arange(n+1)
    xk, _ = np.meshgrid(i, i)
    fac = 2.0 / (n+1)
    c = fac * (f * np.cos(xk.T * (math.pi * (xk + 0.5) / (n+1)))).sum(axis=1)
    return c


def StochasticChebyshevTrace(operator, shape, coeffs):
    """
    Computes the trace of the Chebyshev expansion of the function defined by
    `coeffs` and applied to `operator`, using the Hutchinson estimator

    operator: input operator
    shape: shape of input random vector to use, typically [batch_size, d]
    coeffs: Chebyshev coefficients of the function to evaluate
    n_probe: number of rademacher samples

    WARNING: Will divide coeffs[0] by two
    """
    # Sample a rademacher tensor with desired size
    v = 1 - 2*np.random.binomial(1,0.5,shape)

    # Initialize the iteration
    w0, w1 = v, operator(v)

    s = 0.5*coeffs[0]*w0 + coeffs[1]*w1

    for i in range(2, coeffs.shape[0]):
        wi = 2.*operator(w1) - w0
        s = s + coeffs[i]*wi
        w0=w1*1.0
        w1=wi*1.0

    return v.T.dot(s)

def chebyshev_spectral_sum(func, operator, a, b, shape, deg=20):
    """
    Computes the spectral sum tr(f(A)), where A is the operator,
    f is the func, [a, b] is the range of eigenvalues of A.

    shape: Shape of input vectors to the operator, typically [batch, d]
    deg: Degree of the Chebyshev approximation
    m: Number of random vectors to probe the trace

    This corresponds to Algorithm 1 in Han et al. 2017
    """

    # Compute the chebyshev coefficients of the operator
    c = ChebyshevCoefficients(func, a, b, deg)

    # Rescales the operator
    def scaled_op(x):
        return 2. * operator(x) / (b - a) - (b + a)/(b - a) * x

    Gamma = StochasticChebyshevTrace(scaled_op, shape=shape, coeffs=c)

    return Gamma


def chebyshev_logdet(operator, shape, deg=20, num_iters=10, g=1.1, eps=1e-6, m=100):
    """
    Computes the spectral sum tr(f(A)), where A is the operator,
    f is the func, [a, b] is the range of eigenvalues of A.

    shape: Shape of input vectors to the operator, typically [batch, d]
    deg: Degree of the Chebyshev approximation
    m: Number of random vectors to probe the trace

    This corresponds to Algorithm 1 in Han et al. 2017
    """
    # Find the largest eigenvalue amongst the batch
    lmax = eigen_max(operator, shape, num_iters)
    a, b = eps, g*lmax

    # Rescales the operator
    def scaled_op(x):
        return operator(x) / (a+b)

    a1 = a / (a + b)
    b1 = b / (a + b)

    # Compute the chebyshev coefficients of the operator
    c = ChebyshevCoefficients(np.log, a1, b1, deg)

    # Rescales the operator
    def scaled_op1(x):
        return 2. * scaled_op(x) / (b1 - a1) - (b1 + a1)/(b1 - a1) * x

    res = []
    for i in range(m):
        Gamma = StochasticChebyshevTrace(scaled_op1, shape=shape, coeffs=c)
        res.append( Gamma + shape[0]*np.log(a+b))
    return np.concatenate(res)

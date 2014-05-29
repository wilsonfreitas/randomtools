#!/usr/bin/env python
# encoding: utf-8

"""
randomtools.py

Created by Wilson Freitas on 2008-11-27.
Copyright (c) 2008 WelCo. All rights reserved.
"""

# from inspect import getargspec
from math import sqrt, log, ceil, exp, pi, sin, cos
from operator import mul
from inspect import getargspec


def bindergen(func):
    args_spec = getargspec(func)
    args_len = len(args_spec.args)
    def _func(*args):
        if args_len - len(args) > 0:
            return lambda *xs: _func(*(args + xs))
        else:
            def _(N):
                for i in range(N):
                    yield func(*args)
            return _
    return _func


@bindergen
def uniform(rand, a, b):
    '''
    Uniform distribution - U(a,b)
    
    Generate random numbers uniformly distributed into the interval [a , b).
    
    unif = uniform(rand, a, b)
    
    rand: uniform random number generator, generally provided by standard 
          library or some scientific package.
    a: lower bound
    b: upper bound
    '''
    return a + (b-a) * rand()


@bindergen
def clt(rand, N):
    """
    Gaussian random number generator using the Central Limit Theorem – 
    N(0.0, 1.0)
    
    Generate gaussian distributed random numbers with mean and standard 
    deviation specified. The standard output yields random numbers following 
    the standardized gaussian distribution N(0,1).
    
    The rand function can be any random number generator with known mean and
    variance (m and std stand for mean and standard deviation of the given 
    random number generator).
    This function has default parameters which represent a uniform 
    uniform distribution U(0,1). But in order to understand the way Central 
    Limit Theorem works with other distributions, for example, binomial 
    distribution, these parameters can be changed to configure properly the 
    random numbers generation.
    """
    return (sum( rand() for i in range(N) ) - N*0.5) / ( sqrt(N/12.0) )


@bindergen
def boxmuller(rand):
    """
    Gaussian random number generator using the Box & Muller (1958) 
    transformation – N(0.0, 1.0)
    
    Reference:
    http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    http://www.taygeta.com/random/gaussian.html
    """
    
    u1 = rand()
    u2 = rand()
    return sqrt(-2.0*log(u1))*cos(2*pi*u2) #, sqrt(-2.0*log(u1))*sin(2*pi*u2)


@bindergen
def marsaglia(rand):
    """
    Gaussian random number generator using a variation of the Box & Muller 
    (1958) transformation called Marsaglia polar method – N(0.0, 1.0)
    
    The Marsaglia polar method avoids the trigonometric functions
    calls used in the original implementation of Box & Muller transformation.
    These calls to trigonometric functions are said to slow down the generation 
    of random numbers.
    It might make sense for FORTRAN/C/C++ implementations, but certainly doesn't
    make sense for that Python implementation.
    It has only educational purposes.
    The references bellow contain more information on this theme.
    
    Reference:
    http://en.wikipedia.org/wiki/Marsaglia_polar_method
    http://www.taygeta.com/random/gaussian.html
    """
    
    d = 1.0
    while d >= 1.0:
        x1 = 2.0 * rand() - 1.0
        x2 = 2.0 * rand() - 1.0
        d = x1*x1 + x2*x2
    d = sqrt( (-2.0*log(d))/d )
    return x1*d # (x1*d, x2*d)


A = [ 2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637 ]
B = [ -8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833 ]
C = [ 0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
    0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
    0.0000321767881768, 0.0000002888167364, 0.0000003960315187 ]

@bindergen
def moro(rand):
    '''
    Returns the inverse of the cumulative normal distribution – N(0.0, 1.0)
    Written by B. Moro, November 1994
    '''
    U = rand()
    X = U - 0.5
    if abs(X) < 0.42:
        R = X*X
        R = X*( ( (A[3]*R + A[2])*R + A[1] )*R + A[0] ) / \
            ( ( ( (B[3]*R + B[2])*R + B[1] )*R + B[0] )*R + 1.0 )
        return R

    R = U
    if X > 0.0:
        R = 1.0 - U

    R = log( -log(R) )
    R = C[0] + R*(C[1] + R*( C[2] + R*( C[3] + R*( C[4] + R*( C[5] + R*( C[6] + R*(C[7] + R*C[8]) ) ) ) ) ))
    if X < 0.0:
        R = -R
    return R


@bindergen
def exponential(rand, lambd):
    '''
    Exponential random number generator – exp(lambda)
    '''
    u = rand()
    while u <= 1e-7:
        u = rand()
    
    return -log(u)/lambd


@bindergen
def lognormal(randgauss):
    '''
    Log-normal random number generator – ln(N(0.0, 1.0))
    '''
    return log(randgauss())


@bindergen
def weibull(rand, shape, scale):
    """
    Weibull random number generator – W(shape, scale)
    
    Reference:
    http://www.taygeta.com/random/weibull.xml
    """
    return scale*( -log( 1 - rand() ) )**(1/shape);


@bindergen
def randint(randbits, start, end):
    """
    Discrete uniform random number generator – U(start, end)
    """
    n = abs(end - start)
    x = int(ceil( log(n)/log(2) ))
    i = n + 1
    
    while i>n:
        i = randbits(x)
        
    return i + start


@bindergen
def poisson(rand, lambd):
    """
    Poisson random number generator – poisson(lambda)
    """
    U = rand()
    i = 0
    p = exp(-lambd)
    F = p
    
    while U >= F:
        p = lambd * p / (i + 1)
        F += p
        i += 1
        
    return i


@bindergen
def binomial(rand, n, p):
    """
    Binomial random number generator – binomial(n, p)
    """
    U = rand()
    c = p / (1 - p)
    i = 0
    pr = pow(1-p, n)
    F = pr
    
    while U >= F:
        pr *= c * (n - i) / (i + 1)
        F += pr
        i += 1
        
    return i


@bindergen
def gamma(rand, n, lambd):
    """
    Gamma random number generator – gamma(n, lambda)
    Good example: n=1, lambd=1
    """

    return -( 1.0/lambd )*log( reduce(mul, (rand() for i in range(int(n))), 1.0) )


# from numpy import arange, zeros, mean
# 
# def poisson(rand, lambd):
#     dt = 0.01
#     t = arange(0, 1, dt)
#     N = len(t)
#     m = zeros(N)
#     def _poisson():
#         for i in xrange(N):
#             for j in xrange(N):
#                 if 0 <= rand() <= lambd * dt:
#                     m[j] += 1
#             mu = mean(m)
#             sigma_2 = var(m)





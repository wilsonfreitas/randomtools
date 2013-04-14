#!/usr/bin/env python
# encoding: utf-8

"""
randomtools.py

Created by Wilson Freitas on 2008-11-27.
Copyright (c) 2008 WelCo. All rights reserved.

"""

# from inspect import getargspec
from math import sqrt, log, ceil, exp

LOG_2 = log(2)


class curried(object):
    def __init__(self, func, *args, **kw):
        self.func = func
        self.args = args
        self.kw = kw
    
    def __call__(self, *args, **kw):
        args = self.args + args
        kw = dict(self.kw, **kw)
        return lambda *margs, **mkw: self.func(*(args+margs), **dict(kw, **mkw))


class shortmemory(object):
    def __init__(self, func):
        self.func = func
        self.cache = { }

    def __call__(self, *args, **kw):
        key = args + tuple(kw.keys()) + tuple(kw.values())
        try:
            ret = self.cache[key]
            del self.cache[key]
            return ret[1]
        except:
            self.cache[key] = self.func(*args, **kw)
            ret = self.cache[key]
            return ret[0]

    def __repr__(self):
        return self.func.__doc__


@curried
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


@curried
def clt(rand, mu=0.0, sigma=1.0, m=0.5, std=1.0/12.0, N=12):
    """
    Gaussian random number generator using the Central Limit Theorem – 
    N(mu, sigma**2)
    
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
    s = sum( rand() for i in range(N) )
    norm = (s - N*m) / ( sqrt(N)*std )
    return mu + sigma*norm


@curried
@shortmemory
def boxmiller(rand, mu=0.0, sigma=1.0):
    """
    Gaussian random number generator using the Box & Miller (1958) 
    transformation – N(mu, sigma**2)
    
    The reference bellow proposes a different implementation of original's
    Box & Miller algorithm which uses
    
    y1 = sqrt( - 2 ln(x1) ) cos( 2 pi x2 )
    y2 = sqrt( - 2 ln(x1) ) sin( 2 pi x2 )
    
    to generate gaussian random numbers.
    That implementation is said to be slow due to many calls it does to the math
    library and also might have numerical instability when x1 is close to 
    zero.
    The proposed algorithm is a polar form of the Box & Miller algorithm.
    This polar form is interesting because it does the sine and cosine 
    geometrically without calling the math library.
    
    Reference:
    http://www.taygeta.com/random/gaussian.html
    """
    
    d = 1.0
    while d >= 1.0:
        x1 = 2.0 * rand() - 1.0
        x2 = 2.0 * rand() - 1.0
        d = x1*x1 + x2*x2
    d = sqrt( (-2.0*log(d))/d )
    return (mu + sigma*x1*d, mu + sigma*x2*d)


@curried
def exponential(rand, lambd):
    '''
    Exponential random number generator – exp(lambda)
    '''
    u = rand()
    while u <= 1e-7:
        u = rand()
    
    return -log(u)/lambd


@curried
def lognormal(randgauss, mu=0.0, sigma=1.0):
    '''
    Log-normal random number generator – ln(N(mu, sigma**2))
    '''
    return log(mu + sigma*randgauss())


@curried
def weibull(rand, shape, scale):
    """
    Weibull random number generator – W(shape, scale)
    
    Reference:
    http://www.taygeta.com/random/weibull.xml
    """
    return scale*( -log( 1 - rand() ) )**(1/shape);


@curried
def randint(randbits, start, end):
    """
    Discrete uniform random number generator – U(start, end)
    """
    n = abs(end - start)
    x = int(ceil( log(n)/LOG_2 ))
    i = n + 1
    
    while i>n:
        i = randbits(x)
        
    return i + start


@curried
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


@curried
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
            

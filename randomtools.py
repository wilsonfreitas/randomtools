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
def gaussian_clt(rand, mu=0.0, sigma=1.0, N=12):
    """
    Gaussian random number generator using the Central Limit Theorem
    
    Generate gaussian distributed random numbers with mean and standard 
    deviation specified.
    """
    s = sum( rand() for i in range(N) )
    norm = (s - N*0.5) / sqrt(N/12.0)
    return mu + sigma*norm

@curried
def gaussian_bm(rand, mu=0.0, sigma=1.0):
    """
    Gaussian random number generator using the Box & Miller algorithm
    
    http://www.taygeta.com/random/gaussian.html
    
    TODO: this function must implement some kind of memory, two random numbers
    are generated but one is not used. If it had a memory then the first number
    would be returned at the first call and at the second call the second number
    would be returned without waste time processing a new number.
    """
    
    p = [False, 0.0]
    
    if p[0]:
        p[0] = False
        y = p[1]
    else:
        d = 1.0
        while d >= 1.0:
            x1 = 2.0 * rand() - 1.0
            x2 = 2.0 * rand() - 1.0
            d = x1*x1 + x2*x2
        d = sqrt( (-2.0*log(d))/d )
        y = x1 * d
        p[1] = x2 * d
        p[0] = True
        
    return mu + sigma*y


def gaussian(impl, rand, **kwargs):
    """docstring for gaussian"""
    module = __import__('randomtools')
    func = getattr(module, 'gaussian_' + impl)
    return func(rand, **kwargs)


def randint(randbits, start, end):
    
    def _randint():
        
        n = abs(end - start)
        x = int(ceil( log(n)/LOG_2 ))
        i = n + 1
        
        while i>n:
            i = randbits(x)
            
        return i + start
    
    return _randint


def exponential(rand, lambd):
    
    def _exponential():
        
        u = rand()
        while u <= 1e-7:
            u = rand()
            
        return -log(u)/lambd
    
    return _exponential


def lognormal(randgauss):
    
    def _lognormal():
        return log(randgauss())
    
    return _lognormal


def poisson(rand, lambd):
    
    def _poisson():
        
        U = rand()
        i = 0
        p = exp(-lambd)
        F = p
        
        while U >= F:
            p = lambd * p / (i + 1)
            F += p
            i += 1
            
        return i
    
    return _poisson


def binomial(rand, n, p):
    
    def _binomial():
        
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
    
    return _binomial


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
            

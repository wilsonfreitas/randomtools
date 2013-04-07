#!/usr/bin/env python
# encoding: utf-8

"""
randomtools.py

Created by Wilson Freitas on 2008-11-27.
Copyright (c) 2008 WelCo. All rights reserved.

"""

from math import sqrt, log, ceil, exp

LOG_2 = log(2)


def currying(f, *args, **kw):
    return lambda *margs, **mkw: f(*(args+margs), **dict(kw, **mkw))


def uniform(rand, *args, **kw):
    '''
    Uniform distribution - U(a,b)
    
    Generate random numbers uniformly distributed into the interval [a , b).
    
    unif = uniform(rand, a, b)
    
    rand: uniform random number generator, generally provided by standard 
          library or some scientific package.
    a: lower bound
    b: upper bound
    '''
    def _uniform(a, b):
        return a + (b-a) * rand()
    
    return currying(_uniform, *args, **kw)


# def uniform(rand, a=0.0, b=1.0):
#     
#     def _uniform():
#         return a + (b-a) * rand()
#     
#     return _uniform


def gaussian_clt(rand, *args, **kw):
    """
    Gaussian random number generator using the Central Limit Theorem
    
    Generate gaussian distrubuted random numbers with mean and standard 
    deviation specified.
    """
    
    def _gaussian_clt(mu=0.0, sigma=1.0, N=12):
        s = sum( rand() for i in range(N) )
        norm = (s - N*0.5) / sqrt(N/12.0)
        return mu + sigma*norm
    
    return currying(_gaussian_clt, *args, **kw)


def gaussian_bm(rand, *args, **kw):
    """docstring for gaussian
    http://www.taygeta.com/random/gaussian.html
    """
    
    p = [False, 0.0]
    
    def _gaussian_bm(mu=0.0, sigma=1.0):
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
            
        return mu + sigma * y
    
    return currying(_gaussian_bm, *args, **kw)


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
            

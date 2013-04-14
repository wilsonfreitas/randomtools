#!/usr/local/bin/python
# encoding: utf-8

import unittest
import randomtools as rt
import random
from scipy import stats

class Test(unittest.TestCase):
    def setUp(self):
        self.rand = random.random
        self.N = 10000
        
    def t_pvalue(self, rand, mu_0=0.0):
        """compute t-statistic"""
        sample = [rand() for i in range(self.N)]
        return stats.ttest_1samp(sample, mu_0)[1]
        # mean = np.mean(sample)
        # serr = np.std(sample)/np.sqrt(self.N)
        # return (mean - mu_0)/serr

    def ks_pvalue(self, rand):
        """compute ks-statistic"""
        sample = [rand() for i in range(self.N)]
        st = stats.kstest(sample, 'norm')
        return st[1]

    def jarque_bera_test(self, rand):
        sample = [rand() for i in range(self.N)]
        S = float(self.N)/6*( stats.skew(sample)**2 + 
            0.25*stats.kurtosis(sample, fisher=True)**2 )
        t = stats.chi2(2).ppf(0.95)
        return S < t

    def test_t_uniform(self):
        rand = rt.uniform(self.rand, 0.0, 1.0)
        t = self.t_pvalue(rand, 0.5)
        self.assert_(t > 0.05,
            'mean test for uniform random number generator: %f' % t)
        rand = rt.uniform(self.rand, -1, 1)
        t = self.t_pvalue(rand)
        self.assert_(t > 0.05,
            'mean test for uniform random number generator: %f' % t)

    def test_t_boxmiller(self):
        rand = rt.boxmiller(self.rand, mu=10)
        t = self.t_pvalue(rand, 10.0)
        self.assertEqual(t > 0.05, True, 
            'mean test for gaussian random number generator (B&M): %f'
            % t)
        rand = rt.boxmiller(self.rand, sigma=5)
        t = self.t_pvalue(rand)
        self.assertEqual(t > 0.05, True, 
            'mean test for gaussian random number generator (B&M): %f'
            % t)
        rand = rt.boxmiller(self.rand, mu=10)
        t = self.t_pvalue(rand, 10.0)
        self.assertEqual(t > 0.05, True, 
            'mean test for gaussian random number generator (B&M): %f'
            % t)
        rand = rt.boxmiller(self.rand, mu=10, sigma=5)
        t = self.t_pvalue(rand, 10.0)
        self.assertEqual(t > 0.05, True, 
            'mean test for gaussian random number generator (B&M): %f'
            % t)

    def test_t_clt(self):
        rand = rt.clt(self.rand)
        t = self.t_pvalue(rand)
        self.assertEqual(t > 0.05, True, 
            'mean test for gaussian random number generator (CLT): %f'
            % t)
        rand = rt.clt(self.rand, sigma=9)
        t = self.t_pvalue(rand)
        self.assertEqual(t > 0.05, True, 
            'mean test for gaussian random number generator (CLT): %f'
            % t)
        rand = rt.clt(self.rand, mu=10)
        t = self.t_pvalue(rand, 10.0)
        self.assertEqual(t > 0.05, True, 
            'mean test for gaussian random number generator (CLT): %f'
            % t)
        rand = rt.clt(self.rand, mu=10, sigma=5)
        t = self.t_pvalue(rand, 10.0)
        self.assertEqual(t > 0.05, True, 
            'mean test for gaussian random number generator (CLT): %f'
            % t)

    def test_t_clt2(self):
        rand = rt.clt(self.rand, N=5)
        t = self.t_pvalue(rand)
        self.assertEqual(t > 0.05, True, 
            'mean test for gaussian random number generator (CLT): %f'
            % t)
        rand = rt.clt(self.rand, N=10)
        t = self.t_pvalue(rand)
        self.assertEqual(t > 0.05, True, 
            'mean test for gaussian random number generator (CLT): %f'
            % t)
        rand = rt.clt(self.rand, N=20)
        t = self.t_pvalue(rand)
        self.assertEqual(t > 0.05, True, 
            'mean test for gaussian random number generator (CLT): %f'
            % t)

    def test_ks_clt2(self):
        rand = rt.clt(self.rand, N=3)
        ks = self.ks_pvalue(rand)
        self.assert_(ks > 0.05,
            'ks test for gaussian random number generator (CLT): %f'
            % ks)

    def test_jb_clt2(self):
        rand = rt.clt(self.rand, N=10)
        self.assert_(self.jarque_bera_test(rand),
            'Jarque-Bera test for gaussian random number generator (CLT)')

    def test_jb_boxmiller(self):
        rand = rt.boxmiller(self.rand)
        self.assert_(self.jarque_bera_test(rand),
            'Jarque-Bera test for gaussian random number generator (CLT)')



if __name__ == '__main__':
    unittest.main()

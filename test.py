#!/usr/local/bin/python
# encoding: utf-8

import unittest
import randomtools as rt
import random
from scipy import stats

class Test(unittest.TestCase):
    def setUp(self):
        self.rand = random.random
        self.N = 1000
        self.chi2_95 = stats.chi2(2).ppf(0.95)
        
    def t_pvalue(self, rand, mu_0=0.0):
        """compute t-statistic p-value"""
        sample = [rand() for i in range(self.N)]
        return stats.ttest_1samp(sample, mu_0)[1]
        # mean = np.mean(sample)
        # serr = np.std(sample)/np.sqrt(self.N)
        # return (mean - mu_0)/serr

    def ks_pvalue(self, rand, kind='norm'):
        """compute ks-statistic p-value"""
        sample = [rand() for i in range(self.N)]
        st = stats.kstest(sample, kind)
        return st[1]

    def jb_pvalue(self, rand):
        """compute jarque-bera-statistic p-value"""
        sample = [rand() for i in range(self.N)]
        jb = float(self.N)/6*( stats.skew(sample)**2 + 
            0.25*stats.kurtosis(sample, fisher=True)**2 )
        return 1 - stats.chi2(2).cdf(jb)

    def test_t_uniform(self):
        rand = rt.uniform(self.rand, 0.0, 1.0)
        t = self.t_pvalue(rand, 0.5)
        self.assert_(t > 0.05,
            'mean test for uniform random number generator: %f' % t)
        rand = rt.uniform(self.rand, -1, 1)
        t = self.t_pvalue(rand)
        self.assert_(t > 0.05,
            'mean test for uniform random number generator: %f' % t)

    def test_t_marsaglia(self):
        rand = rt.marsaglia(self.rand)
        t = self.t_pvalue(rand)
        self.assertEqual(t > 0.05, True, 
            'mean test for gaussian random number generator (Marsaglia): %f'
            % t)

    def test_t_moro(self):
        rand = rt.moro(self.rand)
        t = self.t_pvalue(rand)
        self.assertEqual(t > 0.05, True, 
            'mean test for gaussian random number generator (Marsaglia): %f'
            % t)

    def test_t_boxmuller(self):
        rand = rt.boxmuller(self.rand)
        t = self.t_pvalue(rand)
        self.assertEqual(t > 0.05, True, 
            'mean test for gaussian random number generator (B&M): %f'
            % t)

    def test_t_clt(self):
        rand = rt.clt(self.rand)
        t = self.t_pvalue(rand)
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

    def test_t_exponential(self):
        rand = rt.exponential(self.rand, 2.0)
        t = self.t_pvalue(rand, 0.5)
        self.assertEqual(t > 0.05, True, 
            'mean test for exponential random number generator (Exponential): %f'
            % t)

    def test_t_gamma(self):
        rand = rt.gamma(self.rand)
        t = self.t_pvalue(rand, 1.0)
        self.assertEqual(t > 0.05, True, 
            'mean test for gaussian random number generator (Gamma): %f'
            % t)

    def test_ks_exponential(self):
        rand = rt.exponential(self.rand, 1.0)
        ks = self.ks_pvalue(rand, 'expon')
        self.assert_(ks > 0.05,
            'ks test for exponential random number generator (Exponential): %f'
            % ks)

    def test_ks_clt(self):
        rand = rt.clt(self.rand)
        ks = self.ks_pvalue(rand)
        self.assert_(ks > 0.05,
            'ks test for gaussian random number generator (CLT): %f'
            % ks)

    def test_ks_clt2(self):
        rand = rt.clt(self.rand, N=3)
        ks = self.ks_pvalue(rand)
        self.assert_(ks > 0.05,
            'ks test for gaussian random number generator (CLT): %f'
            % ks)

    def test_ks_boxmuller(self):
        rand = rt.boxmuller(self.rand)
        ks = self.ks_pvalue(rand)
        self.assert_(ks > 0.05,
            'ks test for gaussian random number generator (B&M): %f'
            % ks)

    def test_ks_marsaglia(self):
        rand = rt.marsaglia(self.rand)
        ks = self.ks_pvalue(rand)
        self.assert_(ks > 0.05,
            'ks test for gaussian random number generator (Marsaglia): %f'
            % ks)

    def test_ks_moro(self):
        rand = rt.moro(self.rand)
        ks = self.ks_pvalue(rand)
        self.assert_(ks > 0.05,
            'ks test for gaussian random number generator (Moro): %f'
            % ks)

    def test_jb_clt2(self):
        rand = rt.clt(self.rand, N=10)
        jb = self.jb_pvalue(rand)
        self.assert_(jb > 0.05,
            'Jarque-Bera test for gaussian random number generator (CLT): %f' % jb)

    def test_jb_boxmuller(self):
        rand = rt.boxmuller(self.rand)
        jb = self.jb_pvalue(rand)
        self.assert_(jb > 0.05,
            'Jarque-Bera test for gaussian random number generator (B&M) %f' % jb)

    def test_jb_marsaglia(self):
        rand = rt.marsaglia(self.rand)
        jb = self.jb_pvalue(rand)
        self.assert_(jb > 0.05,
            'Jarque-Bera test for gaussian random number generator (Marsaglia) %f' % jb)

    def test_jb_moro(self):
        rand = rt.moro(self.rand)
        jb = self.jb_pvalue(rand)
        self.assert_(jb > 0.05,
            'Jarque-Bera test for gaussian random number generator (Moro) %f' % jb)



if __name__ == '__main__':
    unittest.main()

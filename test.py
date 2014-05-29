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
        sample = [i for i in rand(self.N)]
        return stats.ttest_1samp(sample, mu_0)[1]
        # mean = np.mean(sample)
        # serr = np.std(sample)/np.sqrt(self.N)
        # return (mean - mu_0)/serr

    def ks_pvalue(self, rand, kind='norm'):
        """compute ks-statistic p-value"""
        sample = [i for i in rand(self.N)]
        st = stats.kstest(sample, kind)
        return st[1]

    def jb_pvalue(self, rand):
        """compute jarque-bera-statistic p-value"""
        sample = [i for i in rand(self.N)]
        jb = float(self.N)/6*( stats.skew(sample)**2 + 
            0.25*stats.kurtosis(sample, fisher=True)**2 )
        return 1 - stats.chi2(2).cdf(jb)

    def test_t_uniform(self):
        rand = rt.uniform(self.rand, -1, 1)
        t = self.t_pvalue(rand)
        self.assertGreaterEqual(t, 0.05, 'mean test for uniform RNG: %f' % t)

    def test_t_marsaglia(self):
        rand = rt.marsaglia(self.rand)
        t = self.t_pvalue(rand)
        self.assertGreaterEqual(t, 0.05, 
            'mean test for gaussian RNG (Marsaglia): %f' % t)

    def test_t_moro(self):
        rand = rt.moro(self.rand)
        t = self.t_pvalue(rand)
        self.assertGreaterEqual(t, 0.05, 
            'mean test for gaussian RNG (Marsaglia): %f' % t)

    def test_t_boxmuller(self):
        rand = rt.boxmuller(self.rand)
        t = self.t_pvalue(rand)
        self.assertGreaterEqual(t, 0.05, 
            'mean test for gaussian RNG (B&M): %f' % t)

    def test_t_clt(self):
        rand = rt.clt(self.rand, 12)
        t = self.t_pvalue(rand)
        self.assertGreaterEqual(t, 0.05, 
            'mean test for gaussian RNG (CLT): %f' % t)

    def test_t_clt_small_sample(self):
        rand = rt.clt(self.rand, 3)
        t = self.t_pvalue(rand)
        self.assertGreaterEqual(t, 0.05, 
            'mean test for gaussian RNG (CLT small-sample): %f' % t)

    def test_t_exponential(self):
        rand = rt.exponential(self.rand, 2.0)
        t = self.t_pvalue(rand, 0.5)
        self.assertGreaterEqual(t, 0.05, 
            'mean test for exponential RNG (Exponential): %f' % t)

    def test_t_gamma(self):
        rand = rt.gamma(self.rand, 1.0, 1.0)
        t = self.t_pvalue(rand, 1.0)
        self.assertGreaterEqual(t, 0.05, 
            'mean test for gaussian RNG (Gamma): %f' % t)

    def test_ks_exponential(self):
        rand = rt.exponential(self.rand, 1.0)
        t = self.ks_pvalue(rand, 'expon')
        self.assertGreaterEqual(t, 0.05,
            'ks test for exponential RNG (Exponential): %f' % t)

    def test_ks_clt(self):
        rand = rt.clt(self.rand, 12)
        t = self.ks_pvalue(rand)
        self.assertGreaterEqual(t, 0.05,
            'ks test for gaussian RNG (CLT): %f' % t)

    def test_ks_clt_small_sample(self):
        rand = rt.clt(self.rand, 3)
        t = self.ks_pvalue(rand)
        self.assertGreaterEqual(t, 0.05,
            'ks test for gaussian RNG (CLT small-sample): %f' % t)

    def test_ks_boxmuller(self):
        rand = rt.boxmuller(self.rand)
        t = self.ks_pvalue(rand)
        self.assertGreaterEqual(t, 0.05,
            'ks test for gaussian RNG (B&M): %f' % t)

    def test_ks_marsaglia(self):
        rand = rt.marsaglia(self.rand)
        t = self.ks_pvalue(rand)
        self.assertGreaterEqual(t, 0.05,
            'ks test for gaussian RNG (Marsaglia): %f' % t)

    def test_ks_moro(self):
        rand = rt.moro(self.rand)
        t = self.ks_pvalue(rand)
        self.assertGreaterEqual(t, 0.05,
            'ks test for gaussian RNG (Moro): %f' % t)

    def test_jb_clt_small_sample(self):
        rand = rt.clt(self.rand, 10)
        t = self.jb_pvalue(rand)
        self.assertGreaterEqual(t, 0.05,
            'Jarque-Bera test for gaussian RNG (CLT small-sample): %f' % t)

    def test_jb_boxmuller(self):
        rand = rt.boxmuller(self.rand)
        t = self.jb_pvalue(rand)
        self.assertGreaterEqual(t, 0.05,
            'Jarque-Bera test for gaussian RNG (B&M) %f' % t)

    def test_jb_marsaglia(self):
        rand = rt.marsaglia(self.rand)
        t = self.jb_pvalue(rand)
        self.assertGreaterEqual(t, 0.05,
            'Jarque-Bera test for gaussian RNG (Marsaglia) %f' % t)

    def test_jb_moro(self):
        rand = rt.moro(self.rand)
        t = self.jb_pvalue(rand)
        self.assertGreaterEqual(t, 0.05,
            'Jarque-Bera test for gaussian RNG (Moro) %f' % t)



if __name__ == '__main__':
    unittest.main(verbosity=2)

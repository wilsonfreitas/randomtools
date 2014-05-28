![Bitdeli Badge](https://d2weczhvl823v0.cloudfront.net/wilsonfreitas/randomtools/trend.png)

> Instead of writing an one page document with all random tools it should be interesting post some articles discussing each examples and its formulae.


# randomtools

**randomtools** is a python module which has educational purposes.
It's mainly concerned with helping people understand how to easyly generate random variables for some of the well known distributions (like gaussian and poison) – and some others less popular (like power laws and Levy).
It will also help with understanding how to perform tests on generated random samples.
Here it follows a list with some distributions I'd like to implement.
The *emphasized* distribution names have already been implemented.
I also intend to show how to use **randomtools** to simulate some stochastic processes.

Continuous Distributions:

*	*uniform*
*	gaussian
	*	*boxmuller* – Box & Muller transformation
	*	*marsaglia* – Marsaglia polar method
	*	*clt* – Central Limit Theorem
	*	*moro* – Moro Inversion
	*	interp – gaussian interpolation
*	*exponential*
*	*lognormal*
*	*weibull*
*	*gamma*
*	levy
*	power law (Pareto)
*	beta

Discrete Distributions:

*	*poisson*
*	*binomial*
*	negative binomial
*	geometric

Stochastic Processes:

*	Geometric Brownian Motion
*	Arithmetic Brownian Motion
*	Mean reversion
*	Mean reversion with jumps

<!-- *	noise decorators -->

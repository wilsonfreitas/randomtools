Title: The design of random variables  
Author: Wilson Freitas  
HTML Header: <script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: { inlineMath: [ ['$','$'],['\\(','\\)'] ] }  }); </script>  

# The design of random variables  

[random]: http://docs.python.org/2/library/random.html
[scipy-stats]: http://docs.scipy.org/doc/scipy/reference/stats.html
[scipy]: http://www.scipy.org/
[MT]: http://en.wikipedia.org/wiki/Mersenne_twister

Nowadays, it is much common to pick up a random number generator (RNG) from a library and use it extensively without any prior understanding of how the magic happens.
In my opinion, the comprehension of how random numbers are generated is quite essential for anyone who works with simulation and take it seriously.
We have at hand great libraries offering all sorts of RNGs.
It's easy to use and people use it naively.
I would say that it's fairly relevant to know exactly what are you doing, but in some cases I also admit that you must have faith in the library's implementation and just work on to get your hypothesis validated.

The Python language that comes with batteries included has the [`random`](random) module which brings several random number generators for the most popular distributions and also has the powerful Mersenne Twister as its default random number generator for uniformly random floats in the semi-open range $[0, 1)$.
Besides the `random` module, we have the [Scipy](scipy) framework brought by the Python scientific community.
This framework has the [`stats`](scipy-stats) package that covers a wider range of distributions than `random`.
The list of covered distributions is so extensive that is worth a visit at the project homepage.
Despite of having a great list of distributions and being well tested and structured I feel myself lost every time I try to use `stats`, personally I don't like its structure.
I used to use R for simulations before moving to Python and, in my opinion, the functions names of distributions in R are much intuitive.
Scipy `stats` tries an OO approach which standardizes the distribution parameters and confuses me a lot.
Fortunately, Scipy's legibility is not the subject here, 
I recommend Scipy for anyone interested in scientific development, it is an amazing toolset that any scientific developer must know deeply.

I consider understanding of how a RNG works quite relevant.
Most importantly I consider all probability theory around random variables completely enjoyable.
So, to learn algorithms to generate and test RNGs I have been developing **randomtools**.
It is not a Python module focused on performance, it has educational purposes. Of course it can be used in real experiments but I wouldn't recommend it for proving convergence theorems or computing *pi* with high precision.
All algorithms implemented in **randomtools** have been developed to be as clear as possible for anyone to understand.
In the next sections we will see how random number generators for several distributions can be built and tested.

## Building block

The RNG for uniformly random floats in the semi-open range $[0, 1)$ is the fundamental building block of the random number generators theory.
Once you have that RNG you can create a huge amount of RNGs to many discrete and continuous random variables.
Usually, any language has one of that RNG implemented, we have many available.
The Python language comes with the [Mersenne Twister](MT) RNG which is very good and sufficient for as many cases as I can think of.
The Mersenne Twister RNG has a very long period of $2^19937 - 1$ and it passes in all tests of randomness, as far as I know.

Since we have a fairly RNG for random floats into $[0, 1)$, we can think of simulating an unbiased coin tossing, for example.
Given that both head and tail have the same probability we can simply split our range into $2$ equally spaced ranges.
As the RNG outputs are uniformly distributed into $[0, 1)$ range, we have that averagely, both head and tail are equally likely.
We only must consider that in the sortation of random numbers, those not greater than $0.5$ represent a head outcome and otherwise a tail.
This approach can be extended to reproduce an unbiased roulette.
Instead of splitting the range into 2 ranges we can just split it into $N$ equally spaced intervals.

## Gaussian random variables




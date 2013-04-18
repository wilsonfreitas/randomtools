
# The design of random variables

[random]: http://docs.python.org/2/library/random.html
[scipy-stats]: http://docs.scipy.org/doc/scipy/reference/stats.html
[scipy]: http://www.scipy.org/

Nowadays, it is much common to pick up a random number generator (RNG) from a library and use it extensively without any prior understanding of how the magic happens.
In my opinion, the comprehension of how random numbers are generated is quite essential for anyone who works with simulation and take it seriously.
We have at hand great libraries offering all sorts of RNGs.
It's easy to use and people are using it naively.
I would say that it's fairly relevant to know exactly what are you doing, how and why, but in some cases I also admit that you must belive in what is implemented and all you have to do is pray for the best.

The Python language that comes with batteries included has the [`random`](random) module which brings several random number generators for the most popular distributions and also has the powerful Mersenne Twister as its default random number generator for uniformly random floats in the semi-open range [0.0, 1.0).
Besides the `random` module, we have the [Scipy](scipy) framework brought by the Python scientific community.
This framework has the [`stats`](scipy-stats) package that covers a wider range of distributions than `random`.
The list of covered distributions is so extensive that a visit at the project homepage is certainly worthy.
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




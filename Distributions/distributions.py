import math
import numpy as np
import scipy.stats as st
from distbase import Distribution
import normdist as Normal

class UniformDist(Distribution):
    def __init__(self, a, b):
        if a >= b:
            raise ValueError("a must be less than b!")

        self.a = a
        self.b = b

        super().__init__((a + b) / 2, ((b - a) ** 2) / 12)

    def pdf(self, x): 
        return (1.0 / (self.b - self.a)) if self.a <= x <= self.b else 0

    #Defined seperately to avoid an unneccessary O(n) operation
    def cdf(self, x):
        if x < self.a:
            return 0.0
        elif x >= self.b:
            return 1.0
        else:
            return (x - self.a) / (self.b - self.a)
        
    def plot_pdf(self):
        xp = [i for i in range(self.a, self.b + 1)]
        yp = [1.0 / (self.b - self.a) for i in range(self.a, self.b + 1)]

        self.plot(xp, yp)

    def plot_cdf(self):
        xp = [i for i in range(self.a, self.b + 1)]
        yp = [self.cdf(i) for i in range(self.a, self.b + 1)]

        self.plot(xp, yp)
        

class BinomialDist(Distribution):
    def __init__(self, n: int, p: float):
        if n < 1:
            raise ValueError("n must be an integer greater than 0!")
        elif p < 0.0 or p > 1.0:
            raise ValueError("p must be between 0.0 and 1.0!")

        self.n = n
        self.p = p
        super().__init__(n*p, n*p*(1.0 - p))

    def pdf(self, x):
        if x < 0 or x > self.n:
            return 0

        return math.comb(self.n, x) * (self.p ** x) * ((1.0 - self.p) ** (self.n - x))
    
    def plot_pdf(self):
        xp = [i for i in range(self.n)]
        yp = [self.pdf(i) for i in range(self.n)]

        self.plot(xp, yp)

    def plot_cdf(self):
        xp = [i for i in range(self.n)]
        yp = [self.cdf(i) for i in range(self.n)]

        self.plot(xp, yp)
    
class HyperGeoDist(Distribution):
    def __init__(self, N, M, n):
        if M > N:
            raise ValueError("Can't have more successes in population than population size itself!")
        elif n > N:
            raise ValueError("Sample size must be less than population size!")
        elif n < 0 or M < 0 or N < 0:
            raise ValueError("All values must be positive!")
        
        self.N = N
        self.M = M
        self.n = n

        bin_approx = BinomialDist(n, M / N)
        super().__init__(bin_approx.expected_val, bin_approx.variance)

    def pdf(self, x):
        if 0 <= x <= self.n:
            c1 = math.comb(self.M, x)
            c2 = math.comb(self.N - self.M, self.n - x)
            c3 = math.comb(self.N, self.n)

            return (c1 * c2) / c3
        else:
            return 0
        
    def plot_pdf(self):
        xp = [i for i in range(self.n)]
        yp = [self.pdf(i) for i in range(self.n)]

        self.plot(xp, yp)

    def plot_cdf(self):
        xp = [i for i in range(self.n)]
        yp = [self.cdf(i) for i in range(self.n)]

        self.plot(xp, yp)

#X is the number of failures before r successes
class NegBinomialDist(Distribution):
    def __init__(self, r, p):
        if r < 1:
            raise ValueError("r must be an integer greater than 0!")
        elif not (0.0 <= p <= 1.0):
            raise ValueError("p must be between 0.0 and 1.0")
        
        self.r = r
        self.p = p

        rq = r * (1.0 - p)

        super().__init__(rq / p, rq / (p ** 2))

    def pdf(self, x):
        if x < 0:
            return 0
        
        c = math.comb(x + self.r - 1, self.r - 1)
        return c * (self.p ** self.r) * ((1.0 - self.p) ** x)
    
    def plot_pdf(self, max):
        xp = [i for i in range(max)]
        yp = [self.pdf(i) for i in xp]

        self.plot(xp, yp)

    def plot_cdf(self, max):
        xp = [i for i in range(max)]
        yp = [self.cdf(i) for i in xp]

        self.plot(xp, yp)
    

class PoissonDist(Distribution):
    def __init__(self, l):
        self.l = l
        super().__init__(l, l)

    def pdf(self, x):
        if x < 0:
            return 0
        
        return (math.exp(-1 * self.l) * (self.l ** x)) / math.factorial(x)
    
    def plot_pdf(self, max):
        xp = [i for i in range(max)]
        yp = [self.pdf(i) for i in xp]

        self.plot(xp, yp)

    def plot_cdf(self, max):
        xp = [i for i in range(max)]
        yp = [self.cdf(i) for i in xp]

        self.plot(xp, yp)


d = Normal.NormalDistribution(mean=50, sd=10, n=250)
d.bell_curve(51)

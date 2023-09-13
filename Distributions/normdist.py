import math
import numpy as np
import scipy.stats as st
import plotly.graph_objects as go

class NormalDistribution():
    def __init__(self, vals=None, mean=0.0, sd=1.0, n=1.0):
        if vals:
            #if len(vals) < 30:
                #raise ValueError("Less than 30 values detected, use a t-distribution!")

            self.vals = vals
            self.n = len(vals)
            self.mean = sum(vals) / self.n
            var = sum([(self.mean - xi) ** 2 for xi in vals]) / (self.n - 1)
            self.sd = var ** 0.5
            self.sample_sd = self.sd / (self.n ** 0.5)
        else:
            self.vals = None
            self.mean = mean
            self.sd = sd
            self.n = n
            self.sample_sd = self.sd / (self.n ** 0.5)

    def z_score(self, x):
        z = (x - self.mean) / self.sample_sd
        return st.norm.pdf(z)
    
    def l_area(self, x):
        z = (x - self.mean) / self.sample_sd
        return st.norm.cdf(z)
    
    def hyp_test(self, x, method='<'):
        if method == "<":
            return self.l_area(x)
        elif method == ">":
            return 1 - self.l_area(x)

    def bell_curve(self, v):
        l = self.mean - 6*self.sample_sd
        r = self.mean + 6*self.sample_sd

        x = np.linspace(l, self.mean, num=100, endpoint=False)
        x = np.append(x, self.mean)
        x = np.append(x, np.linspace(self.mean, r, num=100, endpoint=False))
        x = np.append(x, v)

        yp = [self.z_score(xi) for xi in x]

        lbls = ["P(X <= x) = {}".format(self.l_area(xi)) for xi in x]

        fig = go.Figure(go.Scatter(x=x, y=yp, line_shape='spline', hovertext=lbls, selectedpoints=[-1], selected={'marker': {'color': 'red', 'size': 12}}))
        fig.show()


x1 = [1927,	2548,	2825,	1921,	1628,	2175	,2112,	2621,	1843,	2544]
x2 = [2125,	2885,	2895,	1944,	1750,	2182,	2164,	2626,	2006,	2626
]

d = [x-y for x, y in zip(x2, x1)]

n = NormalDistribution(vals=d)
print(n.mean, n.sd, n.sample_sd)
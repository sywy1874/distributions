import math
import plotly.graph_objects as go
import numpy as np

class Distribution():
    def __init__(self, exp_val, var):
        self.expected_val = exp_val
        self.variance = var

    def std_dev(self):
        return math.sqrt(self.variance)
    
    def cdf(self, x):
        return sum([self.pdf(i) for i in range(x)])
    
    def plot(self, x, y):
        """
        def mouse_event(event):
            nearest_x = round(event.xdata)
            plt.text(nearest_x, y[nearest_x], y[nearest_x])
            fig.canvas.draw()

        cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
        """

        xp = np.array(x)
        yp = np.array(y)

        fig = go.Figure(go.Scatter(x=xp, y=yp, line_shape='spline', hovertemplate="x=%{x} P: %{y:.8f}"))
        fig.update_traces(mode="markers+lines")
        fig.show()

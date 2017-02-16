import numpy as np

class GridInterpolator(object):
    def __init__(self, xi, yi, interpolator="mlab"):
        self.xi = xi
        self.yi = yi
        self.xx, self.yy = np.meshgrid(xi, yi)
        self._interpolator = interpolator

    def __call__(self, xl, yl, zl):
        if self._interpolator == "scipy":
            from scipy.interpolate import griddata
            x_sample = 256
            z_gridded = griddata(np.array([yl*x_sample, xl]).T,
                                 np.array(zl),
                                 (self.yy*x_sample, self.xx),
                                 method="linear")
        elif self._interpolator == "mlab":
            from matplotlib.mlab import griddata
            try:
                import mpl_toolkits.natgrid
            except ImportError:
                z_gridded = griddata(xl, yl, zl, self.xi, self.yi,
                                     interp="linear")
            else:
                z_gridded = griddata(xl, yl, zl, self.xi, self.yi)

        return z_gridded

import numpy as np

try:
    from stsci.image import median as stsci_median
except ImportError:
    stsci_median = None

if stsci_median is None:
    def stsci_median(arrs, badmasks=None):
        dd1 = np.ma.array(arrs, mask=badmasks)
        ddm = np.ma.median(dd1, axis=0)
        return ddm

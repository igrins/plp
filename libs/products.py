import os
from json_helper import json_dump
import json

import astropy.io.fits as pyfits

class PipelineProducts(dict):
    def __init__(self, desc, **kwargs):
        self.desc = desc
        dict.__init__(self, **kwargs)

    def save(self, mastername, masterhdu):

        fn, ext = os.path.splitext(mastername)
        dict_to_save = dict()
        for k, d in self.items():
            if hasattr(d, "shape") and (d.shape == masterhdu.data.shape):
                if d.dtype == bool:
                    d = d.astype("i8")
                hdu = pyfits.PrimaryHDU(header=masterhdu.header,
                                        data=d)
                fn0 = "".join([fn, ".", k, ".fits"])
                hdu.writeto(fn0, clobber=True)
                dict_to_save[k+".fits"] = os.path.basename(fn0)
            else:
                dict_to_save[k] = d
        if dict_to_save:
            json_dump(dict_to_save, open(mastername, "w"))

    @classmethod
    def load(self, inname, load_fits=True):
        d = json.load(open(inname))
        if load_fits:
            dirname = os.path.dirname(inname)
            for k, v in d.items():
                if k.endswith(".fits"):
                    hdu = pyfits.open(os.path.join(dirname, v))[0]
                    k2 = k[:-5]
                    if k2.endswith("mask"):
                        d[k2] = hdu.data.astype(bool)
                    else:
                        d[k2] = hdu.data
        desc = d.get("desc", "")

        return PipelineProducts(desc, **d)

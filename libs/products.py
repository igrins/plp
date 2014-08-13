import os
from json_helper import json_dump
import json

import astropy.io.fits as pyfits

class PipelineProducts(dict):
    def __init__(self, desc, **kwargs):
        self.desc = desc
        dict.__init__(self, **kwargs)

    def save(self, mastername, masterhdu=None):

        fn, ext = os.path.splitext(mastername)
        dict_to_save = dict()
        for k, d in self.items():
            if hasattr(d, "shape") and (d.shape == masterhdu.data.shape):
                if d.dtype == bool:
                    d = d.astype("i8")
                if masterhdu is not None:
                    hdu = pyfits.PrimaryHDU(header=masterhdu.header,
                                            data=d)
                else:
                    hdu = pyfits.PrimaryHDU(data=d)
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


class ProductPath(object):
    def __init__(self, igr_path, source_filename):
        self.igr_path = igr_path
        self.basename = os.path.basename(os.path.splitext(source_filename)[0])

    def get_secondary_path(self, ext, subdir_prefix=None):
        outname = "%s.%s" % (self.basename, ext)
        if subdir_prefix is not None:
            subdir = "%s_%s" % (self.basename, subdir_prefix)
        else:
            subdir = None
        return self.igr_path.get_secondary_calib_filename(outname, subdir)


class ProductDB(object):
    def __init__(self, dbpath):
        self.dbpath = dbpath

    def update(self, band, basename):
        with open(self.dbpath, "a") as myfile:
            myfile.write("%s %s\n" % (band, basename))

    def query(self, band, obsid):
        import numpy as np
        with open(self.dbpath, "r") as myfile:
            obsid_list = []
            basename_list = []
            for l0 in myfile.readlines():
                b_l1 = l0.strip().split()
                if len(b_l1) != 2: continue
                b, l1 = b_l1
                if b != band: continue
                obsid_list.append(int(l1.strip().split("_")[-1]))
                basename_list.append(l1.strip())

            # return last one with minimum distance
            obsid_dist = np.abs(np.array(obsid_list) - obsid)
            i = np.where(obsid_dist == np.min(obsid_dist))[0][-1]
            return basename_list[i]


def WavelenthSolutions(object):
    def __init__(self, orders, solutions):
        self.orders = orders
        self.solutions = solutions

    def to_json(self):
        return dict(orders=self.orders,
                    solutions=self.solutions)

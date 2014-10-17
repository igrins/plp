import os
from json_helper import json_dump
import json
import numpy as np

import astropy.io.fits as pyfits

from pyfits import Card

class PipelineImage(object):
    def __init__(self, header, *data_list):
        self.header = header
        self.data_list = data_list
        self.data = data_list[0]

    def store(self, fn, masterhdu=None):
        d_list = self.data_list

        d_list2 = []
        for d in d_list:
            if d.dtype.kind == "b":
                d = d.astype("uint8")

            if hasattr(d, "filled"): # if masked array
                if d.dtype.kind in  "ui": # "u" or "i"
                    d = d.filled(0) # not sure if this is safe
                elif d.dtype.kind == "f":
                    d = d.astype("d").filled(np.nan)
                else:
                    raise ValueError("unsupported dtype :  %s" % str(d.dtype))
            d_list2.append(d)

        if masterhdu is not None:
            def get_primary_hdu(d, header=masterhdu.header):
                return pyfits.PrimaryHDU(header=header,
                                         data=d)
            def get_image_hdu(d, header=masterhdu.header):
                return pyfits.ImageHDU(header=header,
                                       data=d)
        else:
            def get_primary_hdu(d):
                return pyfits.PrimaryHDU(data=d)
            def get_image_hdu(d):
                return pyfits.ImageHDU(data=d)

        hdu0 = get_primary_hdu(d_list2[0])
        hdu_rest = [get_image_hdu(d_) for d_ in d_list2[1:]]

        hdu = pyfits.HDUList([hdu0] + hdu_rest)

        #fn0 = "".join([fn, ".fits"])
        hdu.writeto(fn, clobber=True)


class PipelineDict(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, **kwargs)

    def store(self, fn, masterhdu):
        json_dump(self, open(fn, "w"))

class PipelineProducts(dict):
    def __init__(self, desc):
        self.desc = desc
        dict.__init__(self)

    def add(self, desc, value):
        self[desc] = value


class PipelineStorage(object):
    def __init__(self, igr_path):
        self.igr_path = igr_path
        self._cache = {}

    def get_path(self, desc, mastername):
        section, prefix, ext = desc
        fn0 = prefix + os.path.basename(mastername) + ext
        fn = self.igr_path.get_section_filename_base(section, fn0)

        return fn


    def load(self, product_descs, mastername):
        mastername, ext_ = os.path.splitext(mastername)

        r = PipelineProducts("")
        for (section, prefix, ext) in product_descs:
            fn0 = prefix + os.path.basename(mastername) + ext
            fn = self.igr_path.get_section_filename_base(section, fn0)

            if fn in self._cache:
                print "loading (cached)", fn
                r[(section, prefix, ext)] = self._cache[fn]
            else:
                print "loading", fn
                v = self.load_one(fn)
                r[(section, prefix, ext)] = v
                self._cache[fn] = v

            #self.save_one(fn, v, masterhdu)
        return r

    def load1(self, product_desc, mastername):
        return self.load([product_desc], mastername)[product_desc]

    def store(self, products, mastername, masterhdu=None, cache=True):
        mastername, ext_ = os.path.splitext(mastername)

        for (section, prefix, ext), v in products.items():
            fn0 = prefix + os.path.basename(mastername) + ext
            fn = self.igr_path.get_section_filename_base(section, fn0)

            print "store", fn
            if cache:
                self._cache[fn] = v
            self.save_one(fn, v, masterhdu)

    def save_one(self, fn, v, masterhdu=None):

        v.store(fn, masterhdu)

    def load_one(self, fn):

        if fn.endswith("json"):
            return json.load(open(fn))
        elif fn.endswith("mask.fits"):
            hdu = pyfits.open(fn)[0]
            return PipelineImage(hdu, hdu.data.astype(bool))
        elif fn.endswith("fits"):
            hdu = pyfits.open(fn)[0]
            return hdu
        else:
            raise RuntimeError("Unknown file extension")
        # #fn, ext = os.path.splitext(mastername)
        # #dict_to_save = dict()
        # for k, d in self.items():
        #     if hasattr(d, "shape") and (d.shape == masterhdu.data.shape):
        #         if d.dtype == bool:
        #             d = d.astype("i8")
        #         if masterhdu is not None:
        #             hdu = pyfits.PrimaryHDU(header=masterhdu.header,
        #                                     data=d)
        #         else:
        #             hdu = pyfits.PrimaryHDU(data=d)
        #         fn0 = "".join([fn, ".", k, ".fits"])
        #         hdu.writeto(fn0, clobber=True)
        #         dict_to_save[k+".fits"] = os.path.basename(fn0)
        #     else:
        #         dict_to_save[k] = d
        # if dict_to_save:
        #     json_dump(dict_to_save, open(mastername, "w"))


    def save_old(self, mastername, masterhdu=None):

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


class PipelineProductsDeprecated(dict):
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

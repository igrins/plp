import os
from json_helper import json_dump
import json
import numpy as np

import libs.fits as pyfits
#from astropy.io.fits import Card

class PipelineImage(object):
    Header = pyfits.Header
    def __init__(self, header, data):
        self.header = self.Header(header)
        self.data = data


class PipelineImageBase(object):
    def __init__(self, header, *data_list, **kwargs):
        self.header = header
        self.header_list = [header] * len(data_list)
        self.data_list = data_list
        self.data = data_list[0]

        self.masterhdu = kwargs.get("masterhdu", None)

    def __getitem__(self, i):
        return type(self)(self.header_list[i], self.data_list[i])

    def iter_header_data(self):
        for h, d in zip(self.header_list, self.data_list):
            h = pyfits.Header()
            yield h, d

    def store(self, fn, masterhdu=None):

        if masterhdu is None:
            masterhdu = self.masterhdu

        d_list2 = []
        for h, d in self.iter_header_data():
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
        from itertools import izip
        for hdu1, (h, d) in izip(hdu, self.iter_header_data()):
            hdu1.header.extend(h)

        #fn0 = "".join([fn, ".fits"])
        hdu.writeto(fn, clobber=True)


class PipelineImages(PipelineImageBase):
    def __init__(self, hdu_list, **kwargs):
        #PipelineImageBase
        self.hdu_list = hdu_list
        self.masterhdu = kwargs.get("masterhdu", None)

    def iter_header_data(self):
        for hdu in self.hdu_list:
            yield hdu.header, hdu.data



class PipelineDict(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, **kwargs)

    def store(self, fn, masterhdu=None):
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
        self._hdu_cache = self._cache

    @classmethod
    def from_utdate(cls, utdate, config=None):
        from libs.path_info import IGRINSPath

        if config is None:
            from libs.igrins_config import IGRINSConfig
            config = IGRINSConfig()

        igr_path = IGRINSPath(config, utdate)
        igr_storage = cls(igr_path)
        return igr_storage

    def get_path(self, desc, mastername):
        section, prefix, ext = desc
        fn0 = prefix + os.path.basename(mastername) + ext
        fn = self.igr_path.get_section_filename_base(section, fn0)

        return fn


    def get_masterhdu(self, mastername):
        if mastername in self._hdu_cache:
            return self._hdu_cache[mastername][0]
        else:
            hdu_list = pyfits.open(mastername)
            self._hdu_cache[mastername] = hdu_list
            return hdu_list[0]

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

    def get_item_path(self, product_desc, mastername):
        mastername, ext_ = os.path.splitext(mastername)

        section, prefix, ext = product_desc

        fn0 = prefix + os.path.basename(mastername) + ext
        fn = self.igr_path.get_section_filename_base(section, fn0)

        return fn


    def load_item_from_path(self, item_path):

        fn = item_path

        if fn in self._cache:
            print "loading (cached)", fn
            v = self._cache[fn]
        else:
            print "loading", fn
            v = self.load_one(fn)
            self._cache[fn] = v

            #self.save_one(fn, v, masterhdu)
        return v

    def load_item(self, product_desc, mastername):

        fn = self.get_item_path(product_desc, mastername)

        if fn in self._cache:
            print "loading (cached)", fn
            v = self._cache[fn]
        else:
            print "loading", fn
            v = self.load_one(fn)
            self._cache[fn] = v

            #self.save_one(fn, v, masterhdu)
        return v

    def store_item(self, product_desc, mastername, item):

        fn = self.get_item_path(product_desc, mastername)
        print "saving %s" % fn

        item.store(fn)

        self._cache[fn] = item


    def load1(self, product_desc, mastername,
              return_hdu_list=False):
        product1 = self.load([product_desc], mastername)[product_desc]
        if not return_hdu_list and isinstance(product1,
                                              pyfits.HDUList):
            return product1[0]
        else:
            return product1


    def store(self, products, mastername, masterhdu=None, cache=True):
        mastername, ext_ = os.path.splitext(mastername)

        for (section, prefix, ext), v in products.items():
            fn0 = prefix + os.path.basename(mastername) + ext
            fn = self.igr_path.get_section_filename_base(section, fn0)

            print "store", fn
            if cache:
                self._cache[fn] = v
            self.save_one(fn, v, masterhdu=masterhdu)

    def save_one(self, fn, v, masterhdu=None):

        v.store(fn, masterhdu=masterhdu)

    def load_one(self, fn):

        if fn.endswith("json"):
            return json.load(open(fn))
        elif fn.endswith("mask.fits"):
            hdu = pyfits.open(fn)[0]
            return PipelineImageBase(hdu, hdu.data.astype(bool))
        elif fn.endswith("fits"):
            hdu = pyfits.open(fn)
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
        if os.path.exists(self.dbpath):
            mode = "a"
        else:
            mode = "w"

        with open(self.dbpath, mode) as myfile:
                myfile.write("%s %s\n" % (band, basename))


    def query(self, band, obsid):
        import numpy as np
        import os
        if not os.path.exists(self.dbpath):
            return ""

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

            if obsid_list:
                # return last one with minimum distance
                obsid_dist = np.abs(np.array(obsid_list) - obsid)
                i = np.where(obsid_dist == np.min(obsid_dist))[0][-1]
                return basename_list[i]
            else:
                return None


def WavelenthSolutions(object):
    def __init__(self, orders, solutions):
        self.orders = orders
        self.solutions = solutions

    def to_json(self):
        return dict(orders=self.orders,
                    solutions=self.solutions)

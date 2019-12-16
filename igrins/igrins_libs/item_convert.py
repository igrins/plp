"""
conversion between buffer and custom data format (fits, json, etc)
to_item, from_item is the basic interface.
"""

from io import BytesIO
import astropy.io.fits as pyfits
import json
from ..utils.json_helper import json_dumps

from ..resource_manager.base_storage import StorageBase

def null_function(buf):
    return buf


def fits_add_primary_hdu(hdul, mode="use_first_image_hdu"):

    if mode not in ["empty_if_multi",
                    "use_first_image_hdu"]:
        raise ValueError("unrecognized mode value")

    phdu = None
    if ((mode == "empty_if_multi" and (len(hdul) == 1)) or
        (mode == "use_first_image_hdu" and
         isinstance(hdul[0], (pyfits.ImageHDU,
                              pyfits.PrimaryHDU)))):
        phdu = pyfits.PrimaryHDU(header=hdul[0].header,
                                 data=hdul[0].data)
        hdul = [phdu] + hdul[1:]
    else:
        phdu = pyfits.PrimaryHDU()
        if isinstance(hdul[0], pyfits.PrimaryHDU):
            hdul[0] = pyfits.ImageHDU(hdeader=hdul[0].header,
                                      data=hdul[0].data)
            # hdul[0].verify(option="fix")
            # print("to image")
        hdul = [phdu] + hdul
        # print("added empty HDU")

    # print(len(hdul))
    # if len(hdul) > 1:
    #     hdul[1].verify(option="fix")

    return pyfits.HDUList(hdul)


def fits_remove_empty_primary_hdu(hdul):
    if hdul[0].data is None:
        return hdul[1:]
    else:
        return hdul


class ItemConverterBase(StorageBase):
    def __init__(self, storage):
        self.storage = storage

    def exists(self, section, fn):
        return self.storage.exists(section, fn)

    def load(self, section, fn, item_type=None, check_candidate=None):
        if item_type is None:
            item_type = self.guess_item_type(fn)

        d = self.storage.load(section, fn, item_type=item_type,
                              check_candidate=check_candidate)

        if item_type is not None:
            convert_to = self.get_to_item(item_type)
            d = convert_to(d)

        return d

    def store(self, section, fn, data, item_type=None):

        if item_type is None:
            item_type = self.guess_item_type(fn)

        if item_type is not None:
            convert_from = self.get_from_item(item_type)
            buf = convert_from(data)
        else:
            buf = data

        self.storage.store(section, fn, buf, item_type=item_type)

    @classmethod
    def get_to_item(cls, item_type):
        return null_function

    @classmethod
    def get_from_item(cls, item_type):
        return null_function

    @classmethod
    def to_item(cls, item_type, buf):
        return cls.get_to_item(item_type)(buf)

    @classmethod
    def from_item(cls, item_type, data):
        return cls.get_from_item(item_type)(data)

    @staticmethod
    def guess_item_type(item_desc):
        return None


class ItemConverter(ItemConverterBase):
    def __init__(self, storage, fits_mode="use_first_image_hdu"):
        fits_mode = "empty_if_multi"
        ItemConverterBase.__init__(self, storage)
        self.fits_mode = fits_mode

    @classmethod
    def get_to_item(cls, item_type):
        return dict(fits=cls._to_fits,
                    mask=cls._to_mask,
                    json=cls._to_json,
                    raw=null_function)[item_type]

    def get_from_item(self, item_type):
        return dict(fits=self._from_fits,
                    mask=self._from_mask,
                    json=self._from_json,
                    raw=null_function)[item_type]

    # @classmethod
    # def to_item(cls, item_type, buf):
    #     return cls.get_to_item(item_type)(buf)

    # @classmethod
    # def from_item(cls, item_type, data):
    #     return cls.get_from_item(item_type)(data)

    @staticmethod
    def guess_item_type(item_desc):
        item_type = None

        if isinstance(item_desc, tuple):
            fname = item_desc[-1]
        else:
            fname = item_desc

        if fname.endswith("mask.fits"):
            item_type = "mask"
        elif fname.endswith(".fits"):
            item_type = "fits"
        elif fname.endswith(".json"):
            item_type = "json"
        return item_type

    @staticmethod
    def _to_fits(buf):
        hdul = pyfits.HDUList.fromstring(buf)
        hdul = fits_remove_empty_primary_hdu(hdul)
        return hdul

    def _from_fits(self, hdul):

        hdul = fits_add_primary_hdu(hdul, self.fits_mode)
        print(hdul.info())
        buf = BytesIO()
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'Card is too long')

            hdul.writeto(buf)  #  , output_verify="fix")
        return buf.getvalue()

    @staticmethod
    def _to_json(buf):
        return json.loads(buf)

    @staticmethod
    def _from_json(j):
        return json_dumps(j)

    @classmethod
    def _to_mask(cls, buf):
        from ..utils.load_fits import get_first_science_hdu

        hdulist = cls._to_fits(buf)
        hdu = get_first_science_hdu(hdulist)
        return hdu.data.astype(bool)

    def _from_mask(self, m):
        d = m.astype("uint8")
        # hdul = pyfits.HDUList([pyfits.PrimaryHDU(data=d)])
        hdul = [pyfits.ImageHDU(data=d)]
        return self._from_fits(hdul)

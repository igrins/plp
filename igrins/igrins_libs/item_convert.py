"""
conversion between buffer and custom data format (fits, json, etc)
to_item, from_item is the basic interface.
"""

from io import BytesIO
import astropy.io.fits as pyfits
import json
from ..utils.json_helper import json_dumps


def null_function(buf):
    return buf


class ItemConverterBase(object):
    def __init__(self, storage):
        self.storage = storage

    def load(self, section, fn, item_type=None):
        if item_type is None:
            item_type = self.guess_item_type(fn)

        d = self.storage.load(section, fn, item_type=item_type)

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
    @classmethod
    def get_to_item(cls, item_type):
        return dict(fits=cls._to_fits,
                    mask=cls._to_mask,
                    json=cls._to_json)[item_type]

    @classmethod
    def get_from_item(cls, item_type):
        return dict(fits=cls._from_fits,
                    mask=cls._from_mask,
                    json=cls._from_json)[item_type]

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
        hdulist = pyfits.HDUList.fromstring(buf)
        return hdulist

    @staticmethod
    def _from_fits(hdulist):
        buf = BytesIO()
        hdulist.writeto(buf)
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

    @classmethod
    def _from_mask(cls, m):
        d = m.astype("uint8")
        return cls._from_fits(pyfits.PrimaryHDU(data=d))

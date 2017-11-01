# import os

# from .products import ProductDB

# from .storage_descriptions import (load_resource_def,
#                                    load_descriptions,
#                                    DB_Specs)

# from .load_fits import get_first_science_hdu
# from .cal_db_resources import ResourceManager

# from . import cal_db

import astropy.io.fits as pyfits

import json
from io import BytesIO


# class Storage(object):
#     def get_stream(self, section, fn):
#         pass

#     def get_fileobject(self, section, fn):

#         s = self.get_stream(section, fn)

#         buf = BytesIO()
#         for d in s.stream():
#             buf.write(d)

#         buf.seek(0)

#         return buf

#     def put_stream(self, section, fn, stream):
#         pass

#     def put_fileobject(self, section, fn, stream):
#         pass

from cal_db_context import CalDBContextStack
# from item_convert import ItemConverterBase


class CalDB(object):

    """
    categories:
      config
      refdata (or master cal data)
      resource
      items
    """

    def __init__(self, storage, caldb_spec,
                 item_converter_class=None,
                 qa_generator=None):

        self._caldb_spec = caldb_spec

        if item_converter_class is None:
            self.storage = storage
        else:
            self.storage = item_converter_class(storage)

        self.qa_generator = qa_generator

        self.context_stack = CalDBContextStack(self.storage)

    # Context
    def new_context(self, context_name):
        self.context_stack.new_context(context_name)

    def close_context(self, context_name=None):
        if context_name is not None:
            assert self.context_stack.current.name == context_name

        self.context_stack.close_context()

    # # conversion
    # def get_convert_to(self, item_desc, item_type):
    #     item_type = item_type if item_type else \
    #                 self.item_converter.guess_item_type(item_desc)

    #     return self.item_converter.get_to_item(item_type)

    # def get_convert_from(self, item_desc, item_type):
    #     item_type = item_type if item_type else \
    #                 self.item_converter.guess_item_type(item_desc)

    #     return self.item_converter.get_from_item(item_type)

    def get_section_n_fn(self, basename, item_desc, postfix=""):
        (section, tmpl) = item_desc
        fn = tmpl.format(basename=basename, postfix=postfix)

        return section, fn

    def load(self, basename, item_desc, item_type=None, postfix=""):

        section, fn = self.get_section_n_fn(basename, item_desc, postfix)

        d = self.context_stack.load(section, fn, item_type=item_type)
        return d

    def store(self, basename, item_desc, data, item_type=None,
              postfix=""):

        section, fn = self.get_section_n_fn(basename, item_desc, postfix)

        self.context_stack.store(section, fn, data, item_type=item_type)

    # resources


if __name__ == "__main__":

    utdate = "20150525"
    band = "H"
    obsids = [52]
    master_obsid = obsids[0]

    from recipe_helper import RecipeHelper
    helper = RecipeHelper("../recipe.config", utdate)

    caldb = helper.get_caldb()
    resource = caldb.load_resource_for((band, master_obsid),
                                       resource_type="aperture_definition")

    basename = caldb.db_query_basename("flat_on", (band, master_obsid))

    resource = caldb.load_item_from(basename,
                                    "FLATCENTROID_SOL_JSON")

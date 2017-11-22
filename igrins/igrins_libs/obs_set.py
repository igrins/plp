"""
ObsSet: Helper class for a single obsid, and its derived products.
"""

import astropy.io.fits as pyfits

from .. import DESCS
from ..utils.load_fits import get_first_science_hdu


class ObsSet(object):
    def __init__(self, resource_stack, recipe_name, obsids, frametypes,
                 groupname=None, recipe_entry=None,
                 reset_read_cache=False, basename_postfix=""):
        self.rs = resource_stack
        self.recipe_name = recipe_name
        self.obsids = obsids
        self.frametypes = frametypes
        if groupname is None:
            groupname = str(self.obsids[0])
        self.groupname = groupname
        self.recipe_entry = recipe_entry
        self._reset_read_cache = reset_read_cache
        self.basename_postfix = basename_postfix

        # self.basename = self.caldb._get_basename((self.band, groupname))
        # # this is for query
        # self.basename_for_query = self.caldb._get_basename((self.band,
        #                                                     obsids[0]))

    # def get_config(self):
    #     return self.caldb.get_config()

    # def get(self, name):
    #     return self.caldb.get(self.basename_for_query, name)

    # def get_base_info(self):
    #     return self.caldb.get_base_info(self.band, self.obsids)

    # def get_frames(self):
    #     pass
    def set_basename_postfix(self, basename_postfix):
        self.basename_postfix = basename_postfix

    # context related
    def get_descriptions(self):
        desc = dict(recipe_name=self.recipe_name,
                    obsids=self.obsids,
                    frametypes=self.frametypes,
                    groupname=self.groupname,
                    basename_postfix=self.basename_postfix)  # ,
                    # recipe_entry=self.recipe_entry)

        return desc

    def new_context(self, context_name):
        self.rs.new_context(context_name,
                            reset_read_cache=self._reset_read_cache)

    def close_context(self, context_name):
        self.rs.close_context(context_name)

    def abort_context(self, context_name):
        self.rs.abort_context(context_name)

    def get_obsids(self, frametype=None):
        if frametype is None:
            return self.obsids
        else:
            obsids = [o for o, f in zip(self.obsids, self.frametypes)
                      if f == frametype]

            return obsids

    def get_subset(self, frametype):
        obsids = [o for o, f in zip(self.obsids, self.frametypes)
                  if f == frametype]
        frametypes = [frametype] * len(obsids)

        return ObsSet(self.rs, self.recipe_name, obsids, frametypes)

    # ref_data related
    def load_ref_data(self, kind, get_path=False):
        return self.rs.load_ref_data(kind, get_path=get_path)

    # ResourceStack Interface
    def load_fits_sci_hdu(self, item_desc, postfix=""):
        hdul = self.load(item_desc, item_type="fits", postfix=postfix)
        return get_first_science_hdu(hdul)

    def load(self, item_desc, item_type=None, postfix=""):

        if not isinstance(item_desc, tuple):
            item_desc = DESCS[item_desc]

        r = self.rs.load(self.groupname, item_desc,
                         item_type=item_type, postfix=postfix)
        return r

    def store(self, item_desc, data, item_type=None,
              postfix=None, cache_only=False):

        if not isinstance(item_desc, tuple):
            item_desc = DESCS[item_desc]

        if postfix is None:
            postfix = self.basename_postfix

        self.rs.store(self.groupname, item_desc, data,
                      item_type=item_type, postfix=postfix,
                      cache_only=cache_only)

    def add_to_db(self, db_name):
        self.rs.update_db(db_name, self.groupname)

    def query_resource_for(self, resource_type, postfix=""):

        resource_basename, item_desc = self.rs.query_resource_for(self.groupname,
                                                                  resource_type,
                                                                  postfix=postfix)

        return resource_basename, item_desc

    def load_resource_for(self, resource_type,
                          item_type=None, postfix="",
                          resource_postfix="",
                          check_self=False):
        r = self.rs.load_resource_for(self.groupname, resource_type, item_type=item_type,
                                      postfix=postfix, resource_postfix=resource_postfix,
                                      check_self=check_self)

        return r

    def load_resource_sci_hdu_for(self, resource_type,
                                  item_type=None, postfix="",
                                  resource_postfix="",
                                  check_self=False):
        r = self.rs.load_resource_for(self.groupname, resource_type, item_type=item_type,
                                      postfix=postfix, resource_postfix=resource_postfix,
                                      check_self=check_self)
        hdu = get_first_science_hdu(r)
        return hdu

    # load as hdu list
    def get_hdus(self):

        hdus = []
        for obsid in self.get_obsids():
            hdul = self.rs.load(obsid, DESCS["RAWIMAGE"], item_type="fits")
            hdu = get_first_science_hdu(hdul)
            hdus.append(hdu)

        return hdus

    # to store
    def get_template_hdul(self, *hdu_type_list):
        hdul = self.rs.load(self.get_obsids()[0],
                            DESCS["RAWIMAGE"], item_type="fits")
        master_header = get_first_science_hdu(hdul).header

        hdul = []
        for hdu_type in hdu_type_list:
            if hdu_type == "primary":
                hdu_class = pyfits.PrimaryHDU(header=master_header)
            elif hdu_type == "image":
                hdu_class = pyfits.ImageHDU(header=master_header)
            else:
                msg = "hdu_type of {hdu_type} is not supported"
                raise ValueError(msg.format(hdu_type=hdu_type))
            hdul.append(hdu_class)

        return pyfits.HDUList(hdul)

    def get_hdul_to_write(self, *card_data_list):
        hdu_type_list = ["primary"] + (["image"] * (len(card_data_list) - 1))
        hdul = self.get_template_hdul(*hdu_type_list)
        for hdu, (cards, data) in zip(hdul, card_data_list):
            hdu.header.update(cards)
            if data.dtype.name == "bool":
                data = data.astype("uint8")
            hdu.data = data

        return hdul

    # # ref data
    # def load_ref_data(self, kind, get_path=False):
    #     return self.rs.load_ref_data(kind, get_path=get_path)


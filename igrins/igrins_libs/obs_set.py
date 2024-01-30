"""
ObsSet: Helper class for a single obsid, and its derived products.
"""

import re
import fnmatch
import glob

import astropy.io.fits as pyfits

from .. import DESCS
from ..utils.load_fits import get_first_science_hdu
from ..procedures.clean_pattern import clean_detector_pattern


class ObsSet(object):
    def __init__(self, resource_stack, recipe_name, obsids, frametypes,
                 groupname=None, recipe_entry=None,
                 reset_read_cache=False, basename_postfix="",
                 runner_config=None):
        self.rs = resource_stack
        self.recipe_name = recipe_name
        self.obsids = obsids
        self.frametypes = frametypes
        if groupname is None:
            groupname = str(self.obsids[0]) if self.obsids else ""
        self.master_obsid = None if len(obsids) == 0 else obsids[0]
        self.groupname = groupname
        self.recipe_entry = recipe_entry
        self._reset_read_cache = reset_read_cache
        self.basename_postfix = basename_postfix
        self._recipe_parameters = {}

        self.default_cards = []

        if runner_config is None:
            runner_config = {}

        self.runner_config = runner_config

        # self.basename = self.caldb._get_basename((self.band, groupname))
        # # this is for query
        # self.basename_for_query = self.caldb._get_basename((self.band,
        #                                                     obsids[0]))

    def get_resource_spec(self):
        return self.rs.get_resource_spec()

    # def get_config(self):
    #     return self.caldb.get_config()

    # def get(self, name):
    #     return self.caldb.get(self.basename_for_query, name)

    # def get_base_info(self):
    #     return self.caldb.get_base_info(self.band, self.obsids)

    # def get_frames(self):
    #     pass

    def set_recipe_parameters(self, **kwargs):
        self._recipe_parameters.update(kwargs)

    def get_recipe_parameter(self, parname):
        return self._recipe_parameters.get(parname)

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

    def ensure_recipe_name(self, recipe_pattern):
        f_pattern = re.compile(fnmatch.translate(recipe_pattern))

        for recipe_name in self.recipe_name.split("|"):
            if f_pattern.match(recipe_name):
                self.recipe_name = recipe_name
                break
        else:
            raise ValueError("no matching recipe {} found: {}".
                             format(recipe_pattern, self.recipe_name))

    def get_obsids(self, frametype=None):
        if frametype is None:
            return self.obsids
        else:
            obsids = [o for o, f in zip(self.obsids, self.frametypes)
                      if f == frametype]

            return obsids

    def get_subset(self, *frametypes):
        ofs = [(o, f) for o, f in zip(self.obsids, self.frametypes)
               if f in frametypes]
        obsids = [o for o, f in ofs]
        frametypes = [f for o, f in ofs]
        subset = ObsSet(self.rs, self.recipe_name, obsids, frametypes)
        subset._recipe_parameters = self._recipe_parameters #Propogate recipe parameters to subsets
        #return ObsSet(self.rs, self.recipe_name, obsids, frametypes)
        return subset

    # ref_data related
    def load_ref_data(self, kind, get_path=False):
        return self.rs.load_ref_data(kind, get_path=get_path)

    # ResourceStack Interface
    def load_fits_sci_hdu(self, item_desc, postfix=""):
        hdul = self.load(item_desc, item_type="fits", postfix=postfix)
        return get_first_science_hdu(hdul)

    def locate(self, item_desc, item_type=None, postfix=""):

        if not isinstance(item_desc, tuple):
            item_desc = DESCS[item_desc]

        postfix = "" if postfix is None else postfix

        r = self.rs.locate(self.groupname, item_desc,
                           item_type=item_type, postfix=postfix)
        return r

    def load(self, item_desc, item_type=None, postfix=""):

        if not isinstance(item_desc, tuple):
            item_desc = DESCS[item_desc]

        postfix = "" if postfix is None else postfix

        r = self.rs.load(self.groupname, item_desc,
                         item_type=item_type, postfix=postfix)
        return r

    def store(self, item_desc, data, item_type=None,
              postfix=None, cache_only=False):

        if not isinstance(item_desc, tuple):
            item_desc = DESCS[item_desc]

        if postfix is None:
            postfix = self.basename_postfix
            postfix = "" if postfix is None else postfix

        self.rs.store(self.groupname, item_desc, data,
                      item_type=item_type, postfix=postfix,
                      cache_only=cache_only)

    def store_under(self, item_desc, filename, data, item_type=None,
                    postfix=None, cache_only=False):

        if not isinstance(item_desc, tuple):
            item_desc = DESCS[item_desc]

        if postfix is None:
            postfix = self.basename_postfix
            postfix = "" if postfix is None else postfix

        self.rs.store_under(self.groupname, item_desc, filename, data,
                            item_type=item_type, postfix=postfix,
                            cache_only=cache_only)

    def add_to_db(self, db_name):
        self.rs.update_db(db_name, self.groupname)

    def query_resource_basename(self, db_name):
        return self.rs.query_resource_basename(db_name, self.groupname)

    def query_resource_for(self, resource_type, postfix=""):

        resource_basename, item_desc = self.rs.query_resource_for(self.master_obsid,
                                                                  resource_type,
                                                                  postfix=postfix)

        return resource_basename, item_desc

    def load_resource_for(self, resource_type,
                          item_type=None, postfix="",
                          resource_postfix="",
                          check_self=False):
        r = self.rs.load_resource_for(self.master_obsid,
                                      resource_type, item_type=item_type,
                                      postfix=postfix, resource_postfix=resource_postfix,
                                      check_self=check_self)

        return r

    def load_resource_sci_hdu_for(self, resource_type,
                                  item_type=None, postfix="",
                                  resource_postfix="",
                                  check_self=False):
        r = self.rs.load_resource_for(self.master_obsid, resource_type, item_type=item_type,
                                      postfix=postfix, resource_postfix=resource_postfix,
                                      check_self=check_self)
        hdu = get_first_science_hdu(r)
        return hdu

    # load as hdu list
    def get_hdus(self, obsids=None):

        if obsids is None:
            obsids = self.get_obsids()

        hdus = []

        # try:
        #     date, band = self.get_resource_spec()
        #     filename = glob.glob('calib/primary/'+date+'/SDC'+band+'_'+date+'*pattern_mask.fits')[0] #Load order map
        #     bias_mask = pyfits.getdata(filename)

        #     #bias_mask = self.load_resource_for(DESCS["PATTERNMASK_FITS"])
        #     #bias_mask = self.load_resource_for("bias_mask")
        #     #bias_mask = fits.getdata()


        # except:
        #     breakpoint()
            
        #     bias_mask = None

        for obsid in obsids:
            hdul = self.rs.load(obsid, DESCS["RAWIMAGE"], item_type="fits")


            hdu = get_first_science_hdu(hdul)

            hdul[0].data = clean_detector_pattern(hdul[0].data) #Clean repeating detector battern from raw frames

            # if bias_mask is not None: #Check if bias mask exists
            #     print('LOOKS LIKE IT IS WORKING!!!!!')
            #     #breakpoint()
            #     hdul[0].data = clean_detector_pattern(hdul[0].data, bias_mask) #Clean repeating detector battern from raw frames
            # else:
            #     print('BIAS MASK NOT FOUND YET!!!  NEED TO MAKE IT AGAIN')

            hdus.append(hdu)

        return hdus

    def extend_cards(self, cards):
        self.default_cards.extend(cards)

    # to store
    def get_template_hdul(self, *hdu_type_list, convention=None):
        hdul = self.rs.load(self.get_obsids()[0],
                            DESCS["RAWIMAGE"], item_type="fits")
        master_header = get_first_science_hdu(hdul).header

        if convention == "gemini":
            secondary_header = pyfits.Header()
        else:
            secondary_header = master_header

        hdul = []
        for hdu_type in hdu_type_list:
            if hdu_type == "primary":
                hdu_class = pyfits.PrimaryHDU(header=master_header)
            elif hdu_type == "image":
                hdu_class = pyfits.ImageHDU(header=secondary_header)
            else:
                msg = "hdu_type of {hdu_type} is not supported"
                raise ValueError(msg.format(hdu_type=hdu_type))
            hdul.append(hdu_class)

        return pyfits.HDUList(hdul)

    def get_hdul_to_write(self, *card_data_list, convention=None):
        # hdu_type_list = ["primary"] + (["image"] * (len(card_data_list) - 1))
        if convention == "gemini":
            hdu_type_list = ["primary"] + ["image"] * len(card_data_list)
            card_data_list = (([], None),) + card_data_list
        else:
            hdu_type_list = ["image"] * len(card_data_list)

        hdul = self.get_template_hdul(*hdu_type_list, convention=convention)
        for hdu, (cards, data) in zip(hdul, card_data_list):
            hdu.header.update(self.default_cards + list(cards))
            if data is not None and data.dtype.name == "bool":
                data = data.astype("uint8")
            hdu.data = data

        return hdul

    # # ref data
    # def load_ref_data(self, kind, get_path=False):
    #     return self.rs.load_ref_data(kind, get_path=get_path)


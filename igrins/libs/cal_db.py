from .products import ProductDB

from .storage_descriptions import (load_resource_def,
                                   load_descriptions,
                                   DB_Specs)

from .load_fits import get_first_science_hdu, open_fits
from .cal_db_resources import ResourceManager

class CalDB(object):

    def __init__(self, helper, utdate, db_names=[]):
        self.helper = helper
        self.utdate = utdate
        self.db_dict = {}

        self.DB_Specs = DB_Specs

        self.DESC_DICT = load_descriptions()

        self.RESOURCE_DICT = load_resource_def()

        for db_name in db_names:
            self.load_db(db_name)

        # define resource manager
        self.resource_manager = ResourceManager(self)

    def _repr_html_(self):
        import pandas as pd

        d = self.get_config_dict()
        s = pd.DataFrame(dict(key=d.keys(), value=d.values()))
        
        return s._repr_html_()

    def get_config(self):
        return self.helper.config

    def get_config_dict(self):

        config = self.get_config()

        from collections import OrderedDict
        d = OrderedDict()

        for k, v in config.config.defaults().iteritems():
            d[k] = config.get_value(k, self.utdate)

        return d


    def get_master_cal_dict(self, band):

        assert band in "HK"

        config = self.get_config()

        from collections import OrderedDict
        d = OrderedDict()

        refdate = config.config.get("MASTER_CAL", "refdate")
        for k, v in config.config._sections["MASTER_CAL"].iteritems():
            if k == "__name__":
                pass
            d[k] = config.get("MASTER_CAL", k, band=band, refdate=refdate)

        return d

    # resource manager related method
    def get(self, basename, name):
        return self.resource_manager.get(basename, name)

    def get_base_info(self, band, obsids):
        return self.helper.get_base_info(band, obsids)

    # def get_section_filename_base(self, db_spec_path, db_spec_name,
    #                               subdir=None):
    #     return self.helper.get_section_filename_base(db_spec_path,
    #                                                  db_spec_name,
    #                                                  subdir)

    def load_db(self, db_name):
        if db_name not in self.db_dict:
            db_spec_path, db_spec_name = self.DB_Specs[db_name]
            # db_path = self.helper.get_section_filename_base(db_spec_path,
            #                                                 db_spec_name)
            db_path = self.helper.get_item_path((db_spec_path, db_spec_name),
                                                basename=None)

            db = ProductDB(db_path)
            self.db_dict[db_name] = db
        else:
            db = self.db_dict[db_name]
        return db

    def db_query_basename(self, db_name, basename):
        band, master_obsid = self._get_band_masterobsid(basename)
        db = self.load_db(db_name)
        resource_basename = db.query(band, int(master_obsid))
        return resource_basename


    def query_item_path(self, basename, item_type_or_desc,
                        basename_postfix=None, subdir=None):
        """
        this queries db to find relavant basename and load the related resource.
        """

        if not isinstance(basename, str):
            band, master_obsid = basename
            # basename = self.helper.get_basename(band, master_obsid)
            basename = self.helper.get_basename_with_groupname(band,
                                                               master_obsid)

        if basename_postfix is not None:
            basename += basename_postfix

        if isinstance(item_type_or_desc, str):
            item_desc = self.DESC_DICT[item_type_or_desc.upper()]
        else:
            item_desc = item_type_or_desc

        prevent_split=(basename_postfix is not None)
        path = self.helper.get_item_path(item_desc, basename,
                                         prevent_split=prevent_split,
                                         subdir=subdir)

        return path


    def load_item_from_path(self, item_path):
        """
        this queries db to find relavant basename and load the related resource.
        """

        return self.helper.load_item_from_path(item_path)


    def load_item_from(self, basename, item_type_or_desc,
                       basename_postfix=None):
        """
        this queries db to find relavant basename and load the related resource.
        """

        item_path = self.query_item_path(basename, item_type_or_desc,
                                         basename_postfix)
        return self.load_item_from_path(item_path)


    def load_item(self, band, master_obsid, item_type):
        """
        this queries db to find relavant basename and load the related resource.
        """

        if isinstance(master_obsid, str):
            basename = master_obsid
        else:
            basename = self.helper.get_basename(band, master_obsid)

        item_desc = self.DESC_DICT[item_type.upper()]
        resource = self.helper.load_item(item_desc, basename)

        return resource

    def load_image(self, basename, item_type):
        pipeline_image = self.load_item_from(basename, item_type)
        if hasattr(pipeline_image, "data"):
            return pipeline_image.data
        else:
            return pipeline_image[0].data

    def _get_basename_old(self, band, master_obsid):
        if isinstance(master_obsid, str):
            basename = master_obsid
        else:
            basename = self.helper.get_basename(band, master_obsid)
        return basename

    def _get_basename(self, basename):
        if not isinstance(basename, str):
            band, groupname = basename
            basename = self.helper.get_basename_with_groupname(band,
                                                               groupname)

        return basename

    def _get_band_masterobsid(self, basename):
        if isinstance(basename, str):
            sdc_, utdate, masterobsid = basename.split("_")
            band = sdc_[-1]
            # masterobsid = int(masterobsid_)
        else:
            band, masterobsid = basename

        return band, masterobsid

    def _get_master_hdu(self, basename, master_hdu):
        if master_hdu is None:
            band, master_obsid = self._get_band_masterobsid(basename)
            mastername = self.helper.get_filename(band, master_obsid)
            master_hdu = self.helper.get_masterhdu(mastername)
        elif isinstance(master_hdu, int):
            band, master_obsid = self._get_band_masterobsid(basename)
            # interprete master_hdu as master_obsid
            master_obsid = master_hdu
            master_hdu = self.helper.get_masterhdu(mastername)
        else:
            pass

        return master_hdu

    def store_image(self, basename, item_type, data,
                    master_hdu=None,
                    header=None, card_list=None):
        from products import PipelineImageBase

        item_desc = self.DESC_DICT[item_type.upper()]

        # band, master_obsid = self._get_band_masterobsid(basename)
        # basename = self._get_basename(basename)

        # mastername = self.helper.get_filenames(band, [master_obsid])[0]

        # hdu = self.helper.get_masterhdu(mastername)

        master_hdu = self._get_master_hdu(basename, master_hdu)

        # if header is not None:
        #     master_hdu.header = header

        if card_list is not None:
            master_hdu.header.extend(card_list)

        pipeline_image = PipelineImageBase([], data,
                                           masterhdu=master_hdu)

        self.helper.store_item(item_desc, basename,
                               pipeline_image)

    def store_multi_image(self, basename,
                          item_type, hdu_list,
                          master_hdu=None,
                          basename_postfix=None):
        # basename = self._get_basename(basename)

        item_desc = self.DESC_DICT[item_type.upper()]

        master_hdu = self._get_master_hdu(basename, master_hdu)

        from products import PipelineImages
        pipeline_image = PipelineImages(hdu_list,
                                        masterhdu=master_hdu)

        self.helper.store_item(item_desc, basename,
                               pipeline_image,
                               basename_postfix=basename_postfix)

    def store_dict(self, basename, item_type, data):
        basename = self._get_basename(basename)
        item_desc = self.DESC_DICT[item_type.upper()]

        from products import PipelineDict
        pipeline_dict = PipelineDict(**data)
        self.helper.store_item(item_desc, basename,
                               pipeline_dict)

    def query_resource_for(self, basename, resource_type):
        """
        query resource from the given master_obsid.
        """

        try:
            db_name, item_desc = self.RESOURCE_DICT.get(resource_type,
                                                        resource_type)
        except ValueError as e:
            raise e  # it would be good if we can modify the message

        resource_basename = self.db_query_basename(db_name, basename)

        return resource_basename, item_desc

    def load_resource_for(self, basename, resource_type,
                          get_science_hdu=False):
        """
        this load resource from the given master_obsid.
        """

        resource_basename, item_desc = self.query_resource_for(basename,
                                                               resource_type)

        resource = self.load_item_from(resource_basename, item_desc)

        if get_science_hdu:
            resource = get_first_science_hdu(resource)

        return resource

    def store_resource_for(self, basename, resource_type, data):
        """
        this load resource from the given master_obsid.
        """

        db_name, item_desc = self.RESOURCE_DICT[resource_type]

        resource_basename = self.db_query_basename(db_name, basename)

        # FIX THIS
        self.store_dict(band, resource_basename, item_desc, data)

        return resource

    def get_ref_data_path(self, band, kind):
        from igrins.libs.master_calib import get_ref_data_path
        return get_ref_data_path(self.get_config(), band, kind)

    def load_ref_data(self, band, kind):
        from igrins.libs.master_calib import load_ref_data
        f = load_ref_data(self.get_config(), band=band,
                          kind=kind)
        return f

    def fetch_ref_data(self, band, kind):
        from igrins.libs.master_calib import fetch_ref_data
        fn, d = fetch_ref_data(self.get_config(), band=band,
                          kind=kind)
        return fn, d

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

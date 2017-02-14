
import storage_descriptions as DESCS
from products import ProductDB
import os

def load_storage_descriptions():
    import storage_descriptions
    desc_list = [n for n in dir(storage_descriptions) if n.endswith("_DESC")]
    desc_dict = dict((n[:-5].upper(),
                      getattr(storage_descriptions, n)) for n in desc_list)

    return desc_dict

caldb_resource = '''
aperture_definition=("flat_on", "FLATCENTROID_SOL_JSON")
orders=("flat_on", "FLATCENTROID_ORDERS_JSON")
wvlsol=("sky", "SKY_WVLSOL_JSON")
hotpix_mask=("flat_off", "HOTPIX_MASK")
deadpix_mask=("flat_on", "DEADPIX_MASK")
bias_mask=("flat_on", "BIAS_MASK")
wvlsol_v0=("thar", "WVLSOL_V0_JSON")
flat_off=("flat_off", "FLAT_OFF")
'''

class CalDB(object):

    DB_Specs = dict(flat_on=("PRIMARY_CALIB_PATH", "flat_on.db"),
                    flat_off=("PRIMARY_CALIB_PATH", "flat_off.db"),
                    sky=("PRIMARY_CALIB_PATH", "sky.db"),
                    a0v=("OUTDATA_PATH", "a0v.db"),
                    thar=('PRIMARY_CALIB_PATH', 'thar.db')
                    )

    DESC_DICT = load_storage_descriptions()

    def _load_resource_dict(self):
        resource_dict = {}
        import cal_db
        reload(cal_db)
        import ast
        for l in cal_db.caldb_resource.split("\n"):
            _ = l.strip()
            if not _: continue
            _ = _.split("=")
            if len(_) != 2:
                print "unknown resource definition : " + l.strip()
            else:
                v, e_ = _
                e = ast.literal_eval(e_)
                e0 = e[0]
                e1 = getattr(DESCS, e[1] + "_DESC")
                resource_dict[v] = (e0, e1)

        return resource_dict

    def __init__(self, helper, utdate, db_names=[]):
        self.helper = helper
        self.utdate = utdate
        self.db_dict = {}

        self.RESOURCE_DICT = self._load_resource_dict()

        for db_name in db_names:
            self.load_db(db_name)


    def load_db(self, db_name):
        if db_name not in self.db_dict:
            db_spec_path, db_spec_name = self.DB_Specs[db_name]
            db_path = self.helper.igr_path.get_section_filename_base(db_spec_path,
                                                                     db_spec_name)
            db = ProductDB(db_path)
            self.db_dict[db_name] = db
        else:
            db = self.db_dict[db_name]
        return db

    def db_query_basename(self, db_name, band, master_obsid):
        db = self.load_db(db_name)
        basename = db.query(band, master_obsid)
        return basename


    def query_item_path(self, basename, item_type_or_desc):
        """
        this queries db to find relavant basename and load the related resource.
        """

        if not isinstance(basename, str):
            band, master_obsid = basename
            basename = self.helper.get_basename(band, master_obsid)


        if isinstance(item_type_or_desc, str):
            item_desc = self.DESC_DICT[item_type_or_desc.upper()]
        else:
            item_desc = item_type_or_desc

        path = self.helper.igr_storage.get_item_path(item_desc, basename)

        return path


    def load_item_from_path(self, item_path):
        """
        this queries db to find relavant basename and load the related resource.
        """

        return self.helper.igr_storage.load_item_from_path(item_path)


    def load_item_from(self, basename, item_type_or_desc):
        """
        this queries db to find relavant basename and load the related resource.
        """

        item_path = self.query_item_path(basename, item_type_or_desc)
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
        resource = self.helper.igr_storage.load_item(item_desc, basename)

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
            band, master_obsid = basename
            basename = self.helper.get_basename(band, master_obsid)

        return basename

    def _get_band_masterobsid(self, basename):
        if isinstance(basename, str):
            sdc_, utdate, masterobsid_ = basename.split("_")
            band = sdc_[-1]
            masterobsid = int(masterobsid_)
        else:
            band, masterobsid = basename

        return band, masterobsid

    def store_image(self, basename, item_type, data,
                    header=None, card_list=None):
        band, master_obsid = self._get_band_masterobsid(basename)
        basename = self._get_basename(basename)

        item_desc = self.DESC_DICT[item_type.upper()]

        from products import PipelineImageBase
        mastername = self.helper.get_filenames(band, [master_obsid])[0]

        hdu = self.helper.igr_storage.get_masterhdu(mastername)
        if header is not None:
            hdu.header = header
        if card_list is not None:
            hdu.header.extend(card_list)

        pipeline_image = PipelineImageBase([], data,
                                           masterhdu=hdu)

        self.helper.igr_storage.store_item(item_desc, basename,
                                           pipeline_image)

    def store_multi_image(self, basename, item_type, hdu_list):
        band, master_obsid = self._get_band_masterobsid(basename)
        basename = self._get_basename(basename)

        item_desc = self.DESC_DICT[item_type.upper()]

        from products import PipelineImages
        mastername = self.helper.get_filenames(band, [master_obsid])[0]
        hdu = self.helper.igr_storage.get_masterhdu(mastername)

        pipeline_image = PipelineImages(hdu_list,
                                        masterhdu=hdu)

        self.helper.igr_storage.store_item(item_desc, basename,
                                           pipeline_image)

    def store_dict(self, basename, item_type, data):
        basename = self._get_basename(basename)
        item_desc = self.DESC_DICT[item_type.upper()]

        from products import PipelineDict
        pipeline_dict = PipelineDict(**data)
        self.helper.igr_storage.store_item(item_desc, basename,
                                           pipeline_dict)

    def query_resource_for(self, basename, resource_type):
        """
        query resource from the given master_obsid.
        """

        band, master_obsid = self._get_band_masterobsid(basename)

        try:
            db_name, item_desc = self.RESOURCE_DICT.get(resource_type,
                                                        resource_type)
        except ValueError as e:
            raise e  # it would be good if we can modify the message

        resource_basename = self.db_query_basename(db_name, band,
                                                   master_obsid)

        return resource_basename, item_desc

    def load_resource_for(self, basename, resource_type):
        """
        this load resource from the given master_obsid.
        """

        resource_basename, item_desc = self.query_resource_for(basename,
                                                               resource_type)

        resource = self.load_item_from(resource_basename, item_desc)

        return resource

    def store_resource_for(self, basename, resource_type, data):
        """
        this load resource from the given master_obsid.
        """

        db_name, item_desc = self.RESOURCE_DICT[resource_type]

        band, master_obsid = self._get_band_masterobsid(basename)
        resource_basename = self.db_query_basename(db_name, band, master_obsid)

        # FIX THIS
        self.store_dict(band, resource_basename, item_desc, data)

        return resource


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

    basename = caldb.db_query_basename("flat_on", band, master_obsid)

    resource = caldb.load_item_from(basename,
                                    "FLATCENTROID_SOL_JSON")

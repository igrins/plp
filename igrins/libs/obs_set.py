class ObsSet(object):
    def __init__(self, caldb, band, recipe_name, obsids, frametypes,
                 groupname=None):
        self.caldb = caldb
        self.recipe_name = recipe_name
        self.band = band
        self.obsids = obsids
        self.frametypes = frametypes
        if groupname is None:
            groupname = str(self.obsids[0])
        self.basename = self.caldb._get_basename((self.band, groupname))
        # this is for query
        self.basename_for_query = self.caldb._get_basename((self.band, 
                                                            obsids[0]))

    def get_config(self):
        return self.caldb.get_config()

    def get(self, name):
        return self.caldb.get(self.basename_for_query, name)

    def get_base_info(self):
        return self.caldb.get_base_info(self.band, self.obsids)

    def get_frames(self):
        pass

    def get_obsids(self, frametype=None):
        if frametype is None:
            return self.obsids
        else:
            obsids = [obsid for o, f \
                      in zip(self.obsids, self.frametypes) if f == frametype]

            return obsids

    def get_subset(self, frametype):
        obsids = [o for o, f in zip(self.obsids, self.frametypes)
                  if f == frametype]
        frametypes = [frametype] * len(obsids)

        return ObsSet(self.caldb, self.band, self.recipe_name, 
                      obsids, frametypes)

    def get_hdu_list(self):

        _ = self.get_base_info()
        filenames = _[0]

        from igrins.libs.load_fits import load_fits_data
        hdu_list = [load_fits_data(fn_) for fn_ in filenames]

        return hdu_list

    def get_data_list(self):

        hdu_list = self.get_hdu_list()

        return [hdu.data for hdu in hdu_list]

    def load_db(self, db_name):
        return self.caldb.load_db(db_name)


    def query_item_path(self, item_type_or_desc,
                        basename_postfix=None, subdir=None):
        return self.caldb.query_item_path(self.basename_for_query, 
                                          item_type_or_desc,
                                          basename_postfix=basename_postfix,
                                          subdir=subdir)

    def load_item(self, itemtype,
                  basename_postfix=None):
        return self.caldb.load_item_from(self.basename, itemtype,
                                         basename_postfix=basename_postfix)

    def load_data_frame(self, itemtype, orient="split",
                        basename_postfix=None):
        import pandas as pd
        coeffs_path = self.query_item_path(itemtype, 
                                           basename_postfix=basename_postfix)
        df = pd.read_json(coeffs_path, orient=orient)

        return df

    def load_image(self, item_type):
        "similar to load_item, but returns the image as a numpy.array"
        return self.caldb.load_image(self.basename, item_type)

    def _load_item_from(self, item_type_or_desc,
                        basename_postfix=None):
        return self.caldb.load_item_from(self.basename,
                                         item_type_or_desc,
                                         basename_postfix=basename_postfix)

    def store_data_frame(self, itemtype, df, orient="split",
                        basename_postfix=None):
        # using df.to_dict and self.store_dict does not work as it may
        # writes 'nan' which is not readable from
        # pandas.read_json. So, we use df.to_json for now which write
        # 'null'.

        coeffs_path = self.query_item_path(itemtype, 
                                           basename_postfix=basename_postfix)

        df.to_json(coeffs_path, orient=orient)


    def store_dict(self, item_type, data):
        return self.caldb.store_dict(self.basename, item_type, data)

    def store_image(self, item_type, data, 
                    header=None, card_list=None):
        return self.caldb.store_image(self.basename,
                                      item_type=item_type, data=data,
                                      header=header, card_list=card_list)

    def store_multi_images(self, item_type, hdu_list,
                           basename_postfix=None):
        self.caldb.store_multi_image(self.basename,
                                     item_type, hdu_list,
                                     basename_postfix=basename_postfix)

    def load_resource_for(self, resource_type,
                          get_science_hdu=False):
        return self.caldb.load_resource_for(self.basename_for_query,
                                            resource_type,
                                            get_science_hdu=get_science_hdu)

    # Ref data related

    def get_ref_data_path(self, kind):
        return self.caldb.get_ref_data_path(self.band, kind)

    def load_ref_data(self, kind):
        return self.caldb.load_ref_data(self.band, kind)

    def fetch_ref_data(self, kind):
        return self.caldb.fetch_ref_data(self.band, kind)

    def get_ref_spec_name(self, recipe_name=None):

        if recipe_name is None:
            recipe_name = self.recipe_name

        if (recipe_name in ["SKY"]) or recipe_name.endswith("_AB"):
            ref_spec_key = "SKY_REFSPEC_JSON"
            ref_identified_lines_key = "SKY_IDENTIFIED_LINES_V0_JSON"

        elif recipe_name in ["THAR"]:
            ref_spec_key = "THAR_REFSPEC_JSON"
            ref_identified_lines_key = "THAR_IDENTIFIED_LINES_V0_JSON"

        else:
            raise ValueError("Recipe name of '%s' is unsupported." % recipe_name)

        return ref_spec_key, ref_identified_lines_key




def process_band(utdate, recipe_name, band,
                 obsids, frametypes,
                 config, interactive=True):

    # utdate, recipe_name, band, obsids, config = "20150525", "A0V", "H", [63, 64], "recipe.config"

    from igrins.libs.recipe_helper import RecipeHelper
    helper = RecipeHelper(config, utdate, recipe_name)
    caldb = helper.get_caldb()

    from igrins.libs.load_fits import get_hdus, get_combined_image
    hdus = get_hdus(helper, band, obsids)

    a_and_b = dict()
    for frame, hdu in zip(frametypes, hdus):
        a_and_b.setdefault(frame.upper(), []).append(hdu)

    # print a_and_b.keys()

    a = get_combined_image(a_and_b["A"]) / len(a_and_b["A"])
    b = get_combined_image(a_and_b["B"]) / len(a_and_b["B"])

    sky_data_ = a+b - abs(a-b)

    
    from igrins.libs.get_destripe_mask import get_destripe_mask
    destripe_mask = get_destripe_mask(helper, band, obsids)

    from igrins.libs.image_combine import destripe_sky
    sky_data = destripe_sky(sky_data_, destripe_mask, subtract_bg=False)

    # from igrins.libs.destriper import destriper
    # sky_data = destriper.get_destriped(sky_data_)

    basename = helper.get_basename(band, obsids[0])
    store_output(caldb, basename, hdus[0], sky_data)

    # a = get_combined_image(a_list)
    # b = get_combined_image(b_list)

    # master_obsid = obsids[0]
    # desc = "SPEC_FITS_FLATTENED"
    # blaze_corrected=True
    # src_filename = caldb.query_item_path((band, master_obsid),
    #                                      desc)


def store_output(caldb, basename, master_hdu, sky_data):

    from igrins.libs.products import PipelineImage as Image
    from igrins.libs.products import PipelineImages

    image_list = [Image([("EXTNAME", "SKY")], sky_data)]

    product = PipelineImages(image_list, masterhdu=master_hdu)
    item_type = "SKY_GENERATED_FITS"
    item_desc = caldb.DESC_DICT[item_type]
    caldb.helper.igr_storage.store_item(item_desc, basename,
                                        product)


from igrins.libs.recipe_base import RecipeBase

class RecipeSkyMaker(RecipeBase):

    def run_selected_bands_with_recipe(self, utdate, selected, bands):
        interactive = self.kwargs.get("interactive", True)

        for band in bands:
            for s in selected:
                recipe_name = s[0].strip()
                obsids = s[1]

                obsids = s[1]
                frametypes = s[2]

                aux_infos = s[3]
                exptime = float(aux_infos[4])

                if recipe_name.endswith("_AB"):
                    print recipe_name, obsids, frametypes, exptime

                    process_band(utdate, recipe_name, band, 
                                 obsids, frametypes, aux_infos,
                                 self.config,
                                 interactive)
                # target_type = recipe_name.split("_")[0]

                # if target_type not in ["A0V", "STELLAR", "EXTENDED"]:
                #     print "Unsupported recipe : %s" % recipe_name
                #     continue

                #print (utdate, recipe_name, band, obsids, self.config)


def make_sky(utdate, refdate=None, bands="HK",
             starting_obsids=None, interactive=False,
             recipe_name = "*",
             config_file="recipe.config"):

    recipe = RecipeSkyMaker(interactive=interactive)
    recipe.set_recipe_name(recipe_name)
    recipe.process(utdate, bands,
                   starting_obsids, config_file)

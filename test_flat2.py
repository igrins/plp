from igrins.driver import get_obsset, apply_steps
import igrins.recipes.recipe_flat2 as flat

def test_main():
    config_name = "recipe.config.igrins128"

    obsdate = "20150120"
    obsids = list(range(6, 26))

    frametypes = (["OFF"] * 10) + (["ON"] * 10)

    recipe_name = "FLAT"

    band = "H"

    obsset = get_obsset(obsdate, recipe_name, band,
                        obsids, frametypes, config_name)

    apply_steps(obsset, flat.steps)


if False:
    config_name = "recipe.config.igrins128"

    obsdate = "20150120"
    obsids = list(range(11, 31))
    frametypes = (["OFF"] * 10) + (["ON"] * 10)

    recipe_name = "FLAT"

    band = "H"


    from igrins.libs.igrins_config import IGRINSConfig
    from igrins.libs.resource_manager import get_igrins_resource_manager
    from igrins import get_obsset
    # caldb = get_caldb(config_name, utdate, ensure_dir=True)

    config = IGRINSConfig(config_name)

    resource_manager = get_igrins_resource_manager(config, (obsdate, band))

    print(resource_manager)
    from igrins.libs.obs_set2 import ObsSet
    obsset = ObsSet(resource_manager, recipe_name, obsids, frametypes)

    from igrins.libs.storage_descriptions import load_descriptions
    descs = load_descriptions()

    from igrins.libs.load_fits import get_first_science_hdu
    hdul = obsset.rs.load(10, descs["RAWIMAGE"], item_type="fits")
    hdu = get_first_science_hdu(hdul)
    print (hdu)
    # import astropy.io.fits as pyfits
    # print(pyfits.HDUList.fromstring(d))

if __name__ == "__main__":
    test_main()


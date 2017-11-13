from igrins.driver import get_obsset, apply_steps
import igrins.recipes.recipe_extract4 as extract

def test_main():
    config_name = "recipe.config.igrins128"

    obsdate = "20150120"
    obsids = [45, 46]

    frametypes = "AB"

    recipe_name = "STELLAR_AB"

    band = "H"

    context_name = "test_context_extract.pickle"

    if False: # rerun from saved
        nskip = 3
        save_context_name = None
        saved_context_name = context_name
    else:
        nskip = 0
        save_context_name = None # context_name
        saved_context_name = None

    obsset = get_obsset(obsdate, recipe_name, band,
                        obsids, frametypes, config_name,
                        saved_context_name=saved_context_name)

    apply_steps(obsset, extract.steps,
                nskip=nskip,
                save_context_name=save_context_name)


if __name__ == "__main__":
    test_main()


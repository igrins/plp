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



if __name__ == "__main__":
    test_main()


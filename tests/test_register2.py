from igrins.driver import get_obsset, apply_steps
import igrins.recipes.recipe_register2 as register_sky

def test_main():
    config_name = "recipe.config.igrins128"

    obsdate = "20150120"
    obsids = [65, 68]

    # frametypes = ["ON", "ON"]
    frametypes = None

    recipe_name = "SKY"

    band = "H"

    obsset = get_obsset(obsdate, recipe_name, band,
                        obsids, frametypes, config_name)

    apply_steps(obsset, register_sky.steps)



if __name__ == "__main__":
    test_main()


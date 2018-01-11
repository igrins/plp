def get_pipeline_steps(recipe_name):
    from . import (recipe_flat,
                   recipe_register,
                   recipe_wvlsol as recipe_wvlsol_sky,
                   recipe_extract_sky,
                   recipe_a0v_onoff,
                   recipe_a0v_ab,
                   recipe_stellar_onoff,
                   recipe_stellar_ab,
                   recipe_extended_onoff,
                   recipe_extended_ab)

    # m = p_extract.match(recipe_name)
    # if m:
    #     recipe_name = m.group(1)

    steps = {"flat": recipe_flat.steps,
             "register-sky": recipe_register.steps,
             "wvlsol-sky": recipe_wvlsol_sky.steps,
             "extract-sky": recipe_extract_sky.steps,
             "extended-ab": recipe_extended_ab.steps,
             "extended-onoff": recipe_extended_onoff.steps,
             "stellar-ab": recipe_stellar_ab.steps,
             "stellar-onoff": recipe_stellar_onoff.steps,
             "a0v-ab": recipe_a0v_ab.steps,
             "a0v-onoff": recipe_a0v_onoff.steps
    }

    return steps[recipe_name]

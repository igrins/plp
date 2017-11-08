class Step():
    def __init__(self, name, f, **kwargs):
        self.name = name
        self.f = f
        self.kwargs = kwargs

    def __call__(self, obsset):
        self.f(obsset, **self.kwargs)


def apply_steps(obsset, steps):

    # STEP 1 :
    ## make combined image

    n_steps = len(steps)

    print("[{}]".format(obsset.obsids[0]))
    for context_id, step in enumerate(steps):
        if hasattr(step, "name"):
            context_name = step.name
        else:
            context_name = "Undefined Context {}".format(context_id)

        obsset.rs.new_context(context_name, reset_read_cache=True)
        print("  * ({}/{}) {}...".format(context_id + 1,
                                         n_steps, context_name))
        step(obsset)
        # obsset.rs.describe_context()
        obsset.rs.close_context(context_name)

    import pickle
    pickle.dump(obsset.rs, open("test_context.pickle", "wb"))

def get_obsset(utdate, recipe_name, band,
               obsids, frametypes, config_name):

    from .libs.igrins_config import IGRINSConfig
    from .libs.resource_manager import get_igrins_resource_manager
    # from igrins import get_obsset
    # caldb = get_caldb(config_name, utdate, ensure_dir=True)

    config = IGRINSConfig(config_name)

    resource_manager = get_igrins_resource_manager(config, (utdate, band))

    from .libs.obs_set2 import ObsSet
    obsset = ObsSet(resource_manager, recipe_name, obsids, frametypes)

    return obsset


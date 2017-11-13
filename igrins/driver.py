""" Pipeline Driver """


class Step():
    def __init__(self, name, f, **kwargs):
        self.name = name
        self.f = f
        self.kwargs = kwargs

    def __call__(self, obsset):
        self.f(obsset, **self.kwargs)


def apply_steps(obsset, steps, nskip=0, save_context_name=None):

    # STEP 1 :
    ## make combined image

    n_steps = len(steps)

    print("[{}]".format(obsset.obsids[0]))
    for context_id, step in enumerate(steps):
        if hasattr(step, "name"):
            context_name = step.name
        else:
            context_name = "Undefined Context {}".format(context_id)

        if context_id < nskip:
            continue

        obsset.rs.new_context(context_name, reset_read_cache=True)
        print("  * ({}/{}) {}...".format(context_id + 1,
                                         n_steps, context_name))
        try:
            step(obsset)
        except:
            obsset.rs.abort_context(context_name)
            if save_context_name is not None:
                import cPickle as pickle
                pickle.dump(obsset.rs, open(save_context_name, "wb"))
            raise
        else:
            obsset.rs.close_context(context_name)


def get_obsset(utdate, recipe_name, band,
               obsids, frametypes, config_name,
               saved_context_name=None):

    from .libs.igrins_config import IGRINSConfig
    from .libs.resource_manager import get_igrins_resource_manager
    # from igrins import get_obsset
    # caldb = get_caldb(config_name, utdate, ensure_dir=True)

    config = IGRINSConfig(config_name)

    if saved_context_name is not None:
        import cPickle as pickle
        resource_manager = pickle.load(open(saved_context_name, "rb"))
    else:
        resource_manager = get_igrins_resource_manager(config, (utdate, band))

    from .libs.obs_set2 import ObsSet
    obsset = ObsSet(resource_manager, recipe_name, obsids, frametypes)

    return obsset


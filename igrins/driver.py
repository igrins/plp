""" Pipeline Driver """


class Step():
    def __init__(self, name, f, **kwargs):
        self.name = name
        self.f = f
        self.kwargs = kwargs

    def apply(self, obsset, kwargs):
        kwargs0 = self.kwargs.copy()
        for k in kwargs0:
            if k in kwargs:
                kwargs0[k] = kwargs[k]

        self.f(obsset, **kwargs0)

    def __call__(self, obsset):
        self.f(obsset, **self.kwargs)


def apply_steps(obsset, steps, kwargs=None, nskip=0):

    if kwargs is None:
        kwargs = {}

    n_steps = len(steps)

    print("[{}]".format(obsset))
    for context_id, step in enumerate(steps):
        if hasattr(step, "name"):
            context_name = step.name
        else:
            context_name = "Undefined Context {}".format(context_id)

        if context_id < nskip:
            continue

        obsset.new_context(context_name)
        print("  * ({}/{}) {}...".format(context_id + 1,
                                         n_steps, context_name))
        try:
            # step(obsset)
            step.apply(obsset, kwargs)
        except:
            obsset.abort_context(context_name)
            raise
        else:
            obsset.close_context(context_name)

    # if save_context_name is not None:
    #     obsset.rs.save_pickle(open(save_context_name, "wb"))


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


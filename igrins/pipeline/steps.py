from collections import OrderedDict
from .argh_helper import argh, arg, wrap_multi


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


# STEPS = {}


# def get_pipeline(pipeline_name):
#     steps = STEPS[pipeline_name]
#     return create_pipeline(pipeline_name, steps)


class PipelineKwargs(object):
    def __init__(self, steps):
        self.master_kwargs = OrderedDict()
        for step in steps:
            if hasattr(step, "kwargs"):
                self.master_kwargs.update(step.kwargs)

    def check(self, kwargs):
        for k in kwargs:
            if k not in self.master_kwargs:
                msg = ("{} is invalid keyword argyment for this function"
                       .format(k))
                raise TypeError(msg)

    def generate_docs(self, pipeline_name):
        descs = ["{}={}".format(k, v)for k, v in self.master_kwargs.items()]
        return "{}(obsset, {})".format(pipeline_name,
                                       ", ".join(descs))

    def generate_argh(self):
        args = []
        for k, v in self.master_kwargs.items():
            s = "--" + k.replace("_", "-")
            a = arg(s, default=v)
            args.append(a)

        return args


def create_pipeline_from_steps(pipeline_name, steps):
    pipeline_kwargs = PipelineKwargs(steps)

    def _f(obsset, nskip=0, *kwargs):
        pipeline_kwargs.check(kwargs)
        apply_steps(obsset, steps, nskip=nskip, kwargs=kwargs)

    _f.__doc__ = pipeline_kwargs.generate_docs(pipeline_name)
    _f.__name__ = pipeline_name

    return _f


def create_argh_command_from_steps(recipe_name, steps,
                                   driver_func, driver_args,
                                   recipe_name_fnmatch=None):

    pipeline_kwargs = PipelineKwargs(steps)
    args = pipeline_kwargs.generate_argh()

    if recipe_name_fnmatch is None:
        recipe_name_fnmatch = recipe_name.upper().replace("-", "_")

    def _func(obsdate, **kwargs):
        driver_func(steps, recipe_name_fnmatch, obsdate, **kwargs)

    func = wrap_multi(_func, args)
    func = wrap_multi(func, driver_args)
    func = argh.decorators.named(recipe_name)(func)

    return func

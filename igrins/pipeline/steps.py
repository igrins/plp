from collections import OrderedDict
try:
    from tqdm.autonotebook import tqdm
except ImportError:
    tqdm = None

from .argh_helper import argh, arg, wrap_multi

from ..igrins_libs.logger import info
from ..igrins_libs.recipes import get_pmatch_from_fnmatch


class ArghFactoryBase(object):
    def __init__(self, v):
        self.v = v

    def __call__(self, k):
        s = "--" + k.replace("_", "-")
        a = arg(s, default=self.v)
        return a


class ArghFactoryWithShort(ArghFactoryBase):
    def __init__(self, v, shortened=None):
        ArghFactoryBase.__init__(self, v)
        self.shortened = shortened

    def __call__(self, k):
        s = "--" + k.replace("_", "-")

        if self.shortened is None:
            shortened = k[0]
        else:
            shortened = self.shortened[0]

        a = arg("-" + shortened, s, default=self.v)
        return a


class Step():
    def __init__(self, name, f, **kwargs):
        self.name = name
        self.f = f
        self.kwargs = dict((k, v.v if isinstance(v, ArghFactoryBase) else v)
                           for (k, v) in kwargs.items())
        self.argh_helpers = dict((k, v ) for (k, v) in kwargs.items()
                                 if isinstance(v, ArghFactoryBase))

    def apply(self, obsset, kwargs):
        kwargs0 = self.kwargs.copy()
        for k in kwargs0:
            if k in kwargs:
                kwargs0[k] = kwargs[k]

        self.f(obsset, **kwargs0)

    def __call__(self, obsset):
        self.f(obsset, **self.kwargs)


def apply_steps(obsset, steps, kwargs=None, step_slice=None, on_raise=None,
                progress_mode="terminal"):
    """
    progress_mode : terminal, tqdm, notebook, etc
    """

    if kwargs is None:
        kwargs = {}

    n_steps = len(steps)
    step_range = range(n_steps)
    if step_slice is not None:
        step_range = step_range[step_slice]

    obsdate_band = str(obsset.rs.get_resource_spec())
    if progress_mode == "terminal":
        if obsset.basename_postfix:
            info("[{} {}: {} {}]".format(obsdate_band,
                                         obsset.recipe_name,
                                         obsset.groupname,
                                         obsset.basename_postfix))
        else:
            info("[{} {}: {}]".format(obsdate_band,
                                      obsset.recipe_name, obsset.groupname))

    if tqdm and progress_mode == "tqdm":
        _it = tqdm(enumerate(steps), total=len(steps))
    else:
        _it = enumerate(steps)

    for context_id, step in _it:
        if hasattr(step, "name"):
            context_name = step.name
        else:
            context_name = "Undefined Context {}".format(context_id)

        if context_id not in step_range:
            continue

        obsset.new_context(context_name)
        if progress_mode == "terminal":
            info("  * ({}/{}) {}...".format(context_id + 1,
                                            n_steps, context_name))
        try:
            # step(obsset)
            step.apply(obsset, kwargs)
            obsset.close_context(context_name)
        except Exception:
            obsset.abort_context(context_name)
            if on_raise is not None:
                on_raise(obsset, context_id)
            raise

    # if save_context_name is not None:
    #     obsset.rs.save_pickle(open(save_context_name, "wb"))
    del obsset.rs #Fix memory leak


# STEPS = {}


# def get_pipeline(pipeline_name):
#     steps = STEPS[pipeline_name]
#     return create_pipeline(pipeline_name, steps)


class PipelineKwargs(object):
    def __init__(self, steps):
        self.master_kwargs = OrderedDict()
        self.master_argh_helpers = OrderedDict()

        for step in steps:
            if hasattr(step, "kwargs"):
                self.master_kwargs.update(step.kwargs)
            if hasattr(step, "argh_helpers"):
                self.master_argh_helpers.update(step.argh_helpers)

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
            if k in self.master_argh_helpers:
                a = self.master_argh_helpers[k](k)
            else:
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


def create_argh_command_from_steps(command_name, steps,
                                   driver_func, driver_args,
                                   recipe_name_fnmatch=None,
                                   recipe_name_exclude=None):

    pipeline_kwargs = PipelineKwargs(steps)
    args = pipeline_kwargs.generate_argh()

    if recipe_name_fnmatch is None:
        recipe_name_fnmatch = command_name.upper().replace("-", "_")

    p_match = get_pmatch_from_fnmatch(recipe_name_fnmatch,
                                      recipe_name_exclude)

    def _func(obsdate, **kwargs):
        driver_func(command_name, steps, p_match, obsdate,
                    **kwargs)

    func = wrap_multi(_func, args)
    func = wrap_multi(func, driver_args)
    func = argh.decorators.named(command_name)(func)

    return func

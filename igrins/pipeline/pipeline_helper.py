from collections import OrderedDict
from ..driver import apply_steps



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


STEPS = {}


def get_pipeline(pipeline_name):
    steps = STEPS[pipeline_name]
    return create_pipeline(pipeline_name, steps)


def create_pipeline(pipeline_name, steps):
    pipeline_kwargs = PipelineKwargs(steps)

    def _f(obsset, nskip=0, *kwargs):
        pipeline_kwargs.check(kwargs)
        apply_steps(obsset, steps, nskip=nskip, kwargs=kwargs)

    _f.__doc__ = pipeline_kwargs.generate_docs(pipeline_name)
    _f.__name__ = pipeline_name

    return _f

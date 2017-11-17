from ..external import argh
# from igrins.external.argh.constants import ATTR_ARGS
# from igrins.external.argh.assembling import _fix_compat_issue29

# from functools import wraps


def arg(*args, **kwargs):
    r = dict(option_strings=args, **kwargs)
    return r


def wrap_multi(func, args):
    declared_args = getattr(func, argh.constants.ATTR_ARGS, [])
    setattr(func, argh.constants.ATTR_ARGS, list(args) + declared_args)
    argh.assembling._fix_compat_issue29(func)
    return func



from igrins.recipes.argh_helper import argh
import sys

from igrins.driver import Step, apply_steps


def step1(obsset):
    print(1)


def step2(obsset, lacosmic_thresh):
    print(2, lacosmic_thresh)


steps = [Step("step 1", step1, args0=True),
         Step("step 2", step2,
              lacosmic_thresh=0.)]

def get_flat_steps(lacosmic_thresh=0.):
    steps = [Step("step 1", step1),
             Step("step 2", step2,
                  lacosmic_thresh=lacosmic_thresh)]
    return steps


STEPS = {}


import igrins.external.argh as argh
from igrins.external.argh.constants import ATTR_ARGS
from igrins.external.argh.assembling import _fix_compat_issue29

from functools import wraps


# def test2():
def arg(*args, **kwargs):
    r = dict(option_strings=args, **kwargs)
    return r

args_drive = [arg("-b", "--bands", default="HK"),
              arg("-g", "--groupname", default=None),
              arg("-d", "--debug", default=False)]

args = [arg("--lacosmic-thresh", default=2.),
        arg("--args0", default=True)]

def wrap_multi(func, args):
    declared_args = getattr(func, ATTR_ARGS, [])
    setattr(func, ATTR_ARGS, list(args) + declared_args)
    _fix_compat_issue29(func)
    return func


def flat(utdate, **kwargs):
    print(kwargs)
    # steps = get_flat_steps()
    # for s in steps:
        # s(None)

flat = wrap_multi(flat, args)
flat = wrap_multi(flat, args_drive)

parser = argh.ArghParser()
a = argh.add_commands(parser, [flat])
print(sys.argv)
argh.dispatch(parser, argv=sys.argv[1:])

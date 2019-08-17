from collections import OrderedDict
import inspect


def get_default_values(driver_args):
    default_map = OrderedDict()
    for a in driver_args:
        if "default" not in a:
            continue

        for k in a["option_strings"]:
            if k.startswith("--"):
                default_map[k[2:].replace("-", "_")] = a["default"]

    return default_map


def _merge_signature(s1, s2):

    filtered_ss = (list(s1.parameters.values())[:-1]
                    + [_s for _s in s2.parameters.values()
                        if ((_s.default is not inspect._empty) or
                            (_s.kind is inspect.Parameter.VAR_KEYWORD))])

    return inspect.Signature(filtered_ss)


def merge_signature(f1, f2):
    s1 = inspect.signature(f1)
    s2 = inspect.signature(f2)

    return _merge_signature(s1, s2)

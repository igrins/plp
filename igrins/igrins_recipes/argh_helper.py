from collections import OrderedDict


def get_default_values(driver_args):
    default_map = OrderedDict()
    for a in driver_args:
        if "default" not in a:
            continue

        for k in a["option_strings"]:
            if k.startswith("--"):
                default_map[k[2:].replace("-", "_")] = a["default"]

    return default_map

import numpy as np


def encode_array(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "dtype"):
        return np.asscalar(obj)
    else:
        raise TypeError(repr(obj) + " is not JSON serializable")

import json

def json_dump(obj, f, *kl, **kw):
    if "default" not in kw:
        kw["default"] = encode_array

    json.dump(obj, f, *kl, **kw)

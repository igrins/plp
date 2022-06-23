import numpy as np
import numpy.polynomial as P

import simplejson

def encode_array(obj):
    if hasattr(obj, "to_json"):
        return obj.to_json()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "dtype"):
        return obj.item()
    # # check if numpy polynomial. Need to be improved
    # elif hasattr(obj, "convert"):
    #     p = obj.convert(kind=P.Polynomial)
    #     return ["polynomial", p.coef]
    else:
        raise TypeError(repr(obj) + " is not JSON serializable")

import json

# def json_dump(obj, f, *kl, **kw):
#     if "default" not in kw:
#         kw["default"] = encode_array

    # return simplejson.dumps(o, ignore_nan=True)


def json_dumps(obj, *kl, **kw):
    if "default" not in kw:
        kw["default"] = encode_array

    kw["ignore_nan"] = True
    return simplejson.dumps(obj, *kl, **kw)
    # return json.dumps(obj, *kl, **kw)

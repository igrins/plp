import numpy as np
import numpy.polynomial as P

def encode_array(obj):
    if hasattr(obj, "to_json"):
        return obj.to_json()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "dtype"):
        return np.asscalar(obj)
    # # check if numpy polynomial. Need to be improved
    # elif hasattr(obj, "convert"):
    #     p = obj.convert(kind=P.Polynomial)
    #     return ["polynomial", p.coef]
    else:
        raise TypeError(repr(obj) + " is not JSON serializable")

import json

def json_dump(obj, f, *kl, **kw):
    if "default" not in kw:
        kw["default"] = encode_array

    json.dump(obj, f, *kl, **kw)

def _filter_a0v(a0v, a0v_obsid, group2):

    # print master_obsid, a0v_obsid
    if a0v is not None:
        if a0v_obsid is not None:
            raise ValueError("a0v-obsid option is not allowed "
                             "if a0v opption is used")
        elif str(a0v).upper() == "GROUP2":
            a0v = group2
    else:
        if a0v_obsid is not None:
            a0v = a0v_obsid
        else:
            # a0v, a0v_obsid is all None. Keep it as None
            pass

    return a0v

import re
p = re.compile(r"0+1")

def get_group2(obsset):
    group2 = obsset.recipe_entry.get("group2", None) if obsset.recipe_entry is not None else None
    if group2 == "1":
        return None
    elif p.match(group2):
        return "1"
    else:
        return group2


def get_a0v_obsid(obsset, a0v, a0v_obsid):
    group2 = get_group2(obsset)
    a0v_obsid = _filter_a0v(a0v, a0v_obsid, group2)

    return a0v_obsid

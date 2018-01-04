# for k, group in d3.groupby(groupby_keys, sort=False):
#     objname, objtype, group1, group2, exptime = k
#     recipe_name = get_recipe_name(objtype)

#     obsids = " ".join(group["OBSID"].apply(str))

from itertools import groupby as _groupby


def groupby(df, keys):

    def _k(row):
        return tuple(row[1][keys])

    for k, l in _groupby(df.iterrows(), key=_k):
        indices = [t[0] for t in l]

        yield k, df.loc[indices]

def test():
    for k, subgroup in groupby(d3, ["OBJNAME", "OBJTYPE"]):
        print(k, subgroup.index)

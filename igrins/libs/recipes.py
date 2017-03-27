import numpy as np



def load_recipe_list_numpy(fn, allow_duplicate_groups=False):
    dtype=[('OBJNAME', 'S128'), ('OBJTYPE', 'S128'),
           ('GROUP1', 'S128'), ('GROUP2', 'S128'),
           ('EXPTIME', 'f'), ('RECIPE', 'S128'),
           ('OBSIDS', 'S1024'),  ('FRAMETYPES', 'S1024')]
    d = np.genfromtxt(fn, delimiter=",", names=True, comments="#",
                      dtype=dtype)
    recipe_list = []
    for row in d:
        recipe_name = row["RECIPE"].strip()
        obsids  = map(int, row["OBSIDS"].strip().split())
        frametypes  = row["FRAMETYPES"].strip().split()
        recipe_list.append((recipe_name, obsids, frametypes, row))

    return recipe_list

def load_recipe_list_pandas(fn, allow_duplicate_groups=False):
    dtypes= [('OBJNAME', 'S128'), ('OBJTYPE', 'S128'),
             ('GROUP1', 'S128'), ('GROUP2', 'S128'),
             ('EXPTIME', 'f'), ('RECIPE', 'S128'),
             ('OBSIDS', 'S1024'),  ('FRAMETYPES', 'S1024')]

    names = [_[0] for _ in dtypes]
    df = pd.read_csv(fn, skiprows=0, dtype=dtypes, comment="#", 
                     # names=names, 
                     escapechar="\\", skipinitialspace=True)

    update_group1 = True

    if update_group1:
        msk = (df["GROUP1"] == "1")
        s_obsids = [(r.split()[0] if m else g) for r,g, m
                    in zip(df["OBSIDS"], df["GROUP1"], msk)]

        if np.any(msk):
            # print "RECIPE: repacing group1 value of 1 with 1st obsids."
            # print [s for s, m in zip(s_obsids, msk) if m ]

            df["GROUP1"] = s_obsids

    for i, row in df.iterrows(): 
        if row["OBJTYPE"] != "TAR":
            if row["GROUP1"] != row["OBSIDS"].split()[0]:
                raise ValueError("GROUP1 should be identical to "
                                 "1st OBSIDS unless the OBJTYPE is "
                                 "TAR")

    # df["OBJNAME"] = [s.replace(",", "\\,") for s in df["OBJNAME"]]
    if len(np.unique(df["GROUP1"])) != len(df["GROUP1"]):
        if not allow_duplicate_groups:
            raise ValueError("Dupicate group names in the recipe file.")
        else:
            pass

    d = df.to_records(index=False)

    # d = np.genfromtxt(fn, delimiter=",", names=True, comments="#",
    #                   dtype=dtype)
    recipe_list = []
    for row in d:
        recipe_name = row["RECIPE"].strip()
        obsids  = map(int, row["OBSIDS"].strip().split())
        frametypes  = row["FRAMETYPES"].strip().split()
        recipe_list.append((recipe_name, obsids, frametypes, row))

    return recipe_list

load_recipe_list = load_recipe_list_pandas

def make_recipe_dict(recipe_list):
    recipe_dict = {}
    for recipe_name, obsids, frametypes, row in recipe_list:
        recipe_dict.setdefault(recipe_name, []).append((obsids, frametypes, row))
    return recipe_dict

def get_multi_fnmatch_pattern(fnmatch_list):

    import re, fnmatch
    p_list = []
    for fnmatch1 in fnmatch_list:
        p = re.compile(fnmatch.translate(fnmatch1))
        p_list.append(p)

    def p_match(s, p_list=p_list):
        for p in p_list:
            if p.match(s): return True
        return False

    return p_match


class Recipes(object):
    def __init__(self, fn, allow_duplicate_groups=False):
        self._fn = fn
        self.recipe_list = load_recipe_list(fn, 
                                            allow_duplicate_groups)
        self.recipe_dict = make_recipe_dict(self.recipe_list)

    def select_multi(self, recipe_names, starting_obsids=None):
        selected = []
        for recipe_name in recipe_names:
            _ = self.select_fnmatch(recipe_name, starting_obsids)
            selected.extend(_)

    def select_fnmatch(self, recipe_fnmatch, starting_obsids=None):

        if isinstance(recipe_fnmatch, str):
            recipe_fnmatch_list = [recipe_fnmatch]
        else:
            recipe_fnmatch_list = recipe_fnmatch

        p_match = get_multi_fnmatch_pattern(recipe_fnmatch_list)

        from collections import OrderedDict
        dict_by_1st_obsid = OrderedDict((recipe_item[1][0], recipe_item) \
                                        for recipe_item in self.recipe_list \
                                        if p_match(recipe_item[0]))

        if starting_obsids is None:
            starting_obsids = dict_by_1st_obsid.keys()

        selected = [dict_by_1st_obsid[s1] for s1 in starting_obsids]

        return selected

    def select_fnmatch_by_groups(self, recipe_fnmatch, groups=None):

        if isinstance(recipe_fnmatch, str):
            recipe_fnmatch_list = [recipe_fnmatch]
        else:
            recipe_fnmatch_list = recipe_fnmatch

        p_match = get_multi_fnmatch_pattern(recipe_fnmatch_list)

        from collections import OrderedDict
        dict_by_group = OrderedDict((recipe_item[-1]["GROUP1"], recipe_item) \
                                    for recipe_item in self.recipe_list \
                                    if p_match(recipe_item[0]))

        if groups is None:
            groups = dict_by_group.keys()

        selected = [dict_by_group[s1] for s1 in groups]

        return selected

import pandas as pd

def load_recipe_as_dict_numpy(fn):
    dtype=[('OBJNAME', 'S128'), ('OBJTYPE', 'S128'), ('GROUP1', 'S128'), ('GROUP2', 'S128'), ('EXPTIME', 'f'), ('RECIPE', 'S128'), ('OBSIDS', 'S1024'),  ('FRAMETYPES', 'S1024')]
    d = np.genfromtxt(fn, delimiter=",", names=True, comments="#",
                      dtype=dtype)

    for k in ["RECIPE", "OBJNAME", "OBJTYPE"]:
        d[k] = [n.strip() for n in d[k]]

    obsids  = [map(int, row["OBSIDS"].strip().split()) for row in d]
    frametypes  = [row["FRAMETYPES"].strip().split() for row in d]
    starting_obsids = [o[0] for o in obsids]

    r = dict(starting_obsid=starting_obsids,
             objname=d["OBJNAME"],
             obstype=d["OBJTYPE"],
             group1=d["GROUP1"],
             group2=d["GROUP2"],
             exptime=d["EXPTIME"],
             recipe=d["RECIPE"],
             obsids=obsids,
             frametypes=frametypes)

    return r


def load_recipe_as_dict_pandas(fn):

    dtypes= [('OBJNAME', 'S128'), ('OBJTYPE', 'S128'),
             ('GROUP1', 'S128'), ('GROUP2', 'S128'),
             ('EXPTIME', 'f'), ('RECIPE', 'S128'),
             ('OBSIDS', 'S1024'),  ('FRAMETYPES', 'S1024')]

    df = pd.read_csv(fn, skiprows=0, dtype=dtypes, comment="#", 
                     escapechar="\\", skipinitialspace=True)
    d = df.to_records(index=False)

    for k in ["RECIPE", "OBJNAME", "OBJTYPE"]:
        d[k] = [n.strip() for n in d[k]]

    obsids  = [map(int, row["OBSIDS"].strip().split()) for row in d]
    frametypes  = [row["FRAMETYPES"].strip().split() for row in d]
    starting_obsids = [o[0] for o in obsids]

    group1 = d["GROUP1"]
    try:
        if np.all(d["GROUP1"].astype("i") == 1):
            group1 = starting_obsids
    except ValueError:
        pass

    r = dict(starting_obsid=starting_obsids,
             objname=d["OBJNAME"],
             obstype=d["OBJTYPE"],
             group1=group1, # d["GROUP1"],
             group2=d["GROUP2"],
             exptime=d["EXPTIME"],
             recipe=d["RECIPE"],
             obsids=obsids,
             frametypes=frametypes)

    return r

load_recipe_as_dict = load_recipe_as_dict_pandas

class RecipeLog(pd.DataFrame):
    def __init__(self, fn):
        d = load_recipe_as_dict(fn)

        columns = ["starting_obsid", "objname", "obstype", 
                   "recipe", "obsids", "frametypes",
                   "exptime", "group1", "group2"]
        super(RecipeLog, self).__init__(d, columns=columns)
        #self.set_index("starting_obsid", inplace=True)

    def subset(self, **kwargs):
        """
        You can index the data frame by theie values, 
        but obsids and frametypes are not allowed as they are inherently list.
        """
        bad_k = [k for k in kwargs.iterkeys()
                 if k in ["obsids", "frametypes"]]

        if bad_k:
            raise ValueError("keyname %s cannot be selected." % bad_k)

        from collections import Iterable

        m_reversed = np.ones(len(self.index), dtype=bool)

        for k, v in kwargs.iteritems():
            if isinstance(v, str):
                m = (self[k] == v)
            elif isinstance(v, Iterable):
                m = self[k].isin(v)
            else:
                m = (self[k] == v)

            m_reversed &= m

        #print m_reversed
        return self.iloc[m_reversed]


def _test1():
    fn = "../../recipe_logs/20141023.recipes"
    # r = load_recipe(fn)
    r = RecipeDataFrame(fn)
    r2 = Recipes(fn)

    r.subset()
    r.subset(starting_obsid=105)
    r.subset(starting_obsid=[11, 105])
    r.subset(recipe=["SKY", "EXTENDED_ONOFF"])

def _test2():
    fn = "../../recipe_logs/20141023.recipes"
    # r = load_recipe(fn)
    r2 = Recipes(fn)

    s1 = r2.select_deprecate("A0V_AB")
    s2 = r2.select_fnmatch("A0V_AB")

if __name__ == "__main__":
    pass

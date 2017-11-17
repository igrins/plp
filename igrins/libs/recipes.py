from __future__ import print_function

import numpy as np
import pandas as pd


def load_recipe_list_numpy(fn, allow_duplicate_groups=False):
    dtype = [('OBJNAME', 'S128'), ('OBJTYPE', 'S128'),
             ('GROUP1', 'S128'), ('GROUP2', 'S128'),
             ('EXPTIME', 'f'), ('RECIPE', 'S128'),
             ('OBSIDS', 'S1024'),  ('FRAMETYPES', 'S1024')]
    d = np.genfromtxt(fn, delimiter=",", names=True, comments="#",
                      dtype=dtype)
    recipe_list = []
    for row in d:
        recipe_name = row["RECIPE"].strip()
        obsids = [int(v) for v in row["OBSIDS"].strip().split()]
        frametypes = row["FRAMETYPES"].strip().split()
        recipe_list.append((recipe_name, obsids, frametypes, row))

    return recipe_list


def load_recipe_list_pandas(fn, allow_duplicate_groups=False):
    dtypes = [('OBJNAME', 'U128'), ('OBJTYPE', 'U128'),
              ('GROUP1', 'U128'), ('GROUP2', 'U128'),
              ('EXPTIME', 'f'), ('RECIPE', 'U128'),
              ('OBSIDS', 'U1024'),  ('FRAMETYPES', 'U1024')]

    # names = [_[0] for _ in dtypes]
    df = pd.read_csv(fn, skiprows=0, dtype=dict(dtypes), comment="#",
                     escapechar="\\", skipinitialspace=True,
                     engine="python")

    update_group1 = True

    if update_group1:
        msk = (df["GROUP1"] == "1")
        s_obsids = [(r.split()[0] if m else g) for r, g, m
                    in zip(df["OBSIDS"], df["GROUP1"], msk)]

        if np.any(msk):
            # print "RECIPE: repacing group1 value of 1 with 1st obsids."
            # print [s for s, m in zip(s_obsids, msk) if m ]

            df["GROUP1"] = s_obsids

    for i, row in df.iterrows():
        if row["OBJTYPE"] != "TAR":
            if row["GROUP1"] != row["OBSIDS"].split()[0]:
                msg = ("GROUP1 should be identical to "
                       "1st OBSIDS unless the OBJTYPE is "
                       "TAR : "
                       "GROUP1=%s, OBSID[0]=%s")
                raise ValueError(msg % (row["GROUP1"],
                                        row["OBSIDS"].split()[0]))

    if not allow_duplicate_groups:
        for i, row in df.groupby(["GROUP1", "RECIPE"]):
            if len(row) > 1:
                msg = "Dupicate group names with same recipe found: "
                msg += "GROUP1={} OBJTYPE={}".format(row["GROUP1"],
                                                     row["RECIPE"])
                raise ValueError(msg)

    d = df.to_records(index=False)

    # d = np.genfromtxt(fn, delimiter=",", names=True, comments="#",
    #                   dtype=dtype)
    recipe_list = []
    for row in d:
        recipe_name = row["RECIPE"].strip()
        obsids = [int(v) for v in row["OBSIDS"].strip().split()]
        frametypes = row["FRAMETYPES"].strip().split()
        recipe_list.append((recipe_name, obsids, frametypes, row))

    return recipe_list


load_recipe_list = load_recipe_list_pandas


def make_recipe_dict(recipe_list):
    recipe_dict = {}
    for recipe_name, obsids, frametypes, row in recipe_list:
        _ = (obsids, frametypes, row)
        recipe_dict.setdefault(recipe_name, []).append(_)
    return recipe_dict


def get_multi_fnmatch_pattern(fnmatch_list):

    import re
    import fnmatch
    p_list = []
    for fnmatch1 in fnmatch_list:
        p = re.compile(fnmatch.translate(fnmatch1))
        p_list.append(p)

    def p_match(s, p_list=p_list):
        for p in p_list:
            if p.match(s):
                return True
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
        dict_by_1st_obsid = OrderedDict((recipe_item[1][0], recipe_item)
                                        for recipe_item in self.recipe_list
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

        _ = []
        for recipe_item in self.recipe_list:
            for recipe_name in recipe_item[0].split("|"):
                if p_match(recipe_name):
                    recipe_item_new = (recipe_name, ) + recipe_item[1:]
                    _.append((recipe_item[-1]["GROUP1"],
                              recipe_item_new))

        # from collections import OrderedDict
        # dict_by_group = OrderedDict(_)

        if groups is None:
            selected = [s1[1] for s1 in _]
        else:
            selected = [s1[1] for s1 in _ if s1[0] in groups]

        return selected


class Recipes2(object):
    def __init__(self, fn, allow_duplicate_groups=False):
        self._fn = fn
        self.recipe_list = load_recipe_list(fn,
                                            allow_duplicate_groups)

    # def select_multi(self, recipe_names, starting_obsids=None):
    #     selected = []
    #     for recipe_name in recipe_names:
    #         _ = self.select_fnmatch(recipe_name, starting_obsids)
    #         selected.extend(_)

    # def select_fnmatch(self, recipe_fnmatch, starting_obsids=None):

    #     if isinstance(recipe_fnmatch, str):
    #         recipe_fnmatch_list = [recipe_fnmatch]
    #     else:
    #         recipe_fnmatch_list = recipe_fnmatch

    #     p_match = get_multi_fnmatch_pattern(recipe_fnmatch_list)

    #     from collections import OrderedDict
    #     dict_by_1st_obsid = OrderedDict((recipe_item[1][0], recipe_item)
    #                                     for recipe_item in self.recipe_list
    #                                     if p_match(recipe_item[0]))

    #     if starting_obsids is None:
    #         starting_obsids = dict_by_1st_obsid.keys()

    #     selected = [dict_by_1st_obsid[s1] for s1 in starting_obsids]

    #     return selected

    def select_fnmatch_by_groups(self, recipe_fnmatch, groups=None):

        if isinstance(recipe_fnmatch, str):
            recipe_fnmatch_list = [recipe_fnmatch]
        else:
            recipe_fnmatch_list = recipe_fnmatch

        p_match = get_multi_fnmatch_pattern(recipe_fnmatch_list)

        _ = []
        for recipe_item in self.recipe_list:
            for recipe_name in recipe_item[0].split("|"):
                if p_match(recipe_name):
                    recipe_item_new = (recipe_name, ) + recipe_item[1:]
                    _.append((recipe_item[-1]["GROUP1"],
                              recipe_item_new))

        # from collections import OrderedDict
        # dict_by_group = OrderedDict(_)

        if groups is None:
            selected = [s1[1] for s1 in _]
        else:
            selected = [s1[1] for s1 in _ if s1[0] in groups]

        return selected


def load_recipe_as_dict_numpy(fn):
    dtype = [('OBJNAME', 'S128'),
             ('OBJTYPE', 'S128'),
             ('GROUP1', 'S128'),
             ('GROUP2', 'S128'),
             ('EXPTIME', 'f'),
             ('RECIPE', 'S128'),
             ('OBSIDS', 'S1024'),
             ('FRAMETYPES', 'S1024')]

    d = np.genfromtxt(fn, delimiter=",", names=True, comments="#",
                      dtype=dtype)

    for k in ["RECIPE", "OBJNAME", "OBJTYPE"]:
        d[k] = [n.strip() for n in d[k]]

    obsids = [[int(v) for v in  row["OBSIDS"].strip().split()] for row in d]
    frametypes = [row["FRAMETYPES"].strip().split() for row in d]
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

    dtypes = [('OBJNAME', 'S128'), ('OBJTYPE', 'S128'),
              ('GROUP1', 'S128'), ('GROUP2', 'S128'),
              ('EXPTIME', 'f'), ('RECIPE', 'S128'),
              ('OBSIDS', 'S1024'),  ('FRAMETYPES', 'S1024')]

    df = pd.read_csv(fn, skiprows=0, dtype=dtypes, comment="#",
                     escapechar="\\", skipinitialspace=True)
    d = df.to_records(index=False)

    for k in ["RECIPE", "OBJNAME", "OBJTYPE"]:
        d[k] = [n.strip() for n in d[k]]

    obsids = [[int(v) for v in row["OBSIDS"].strip().split()] for row in d]
    frametypes = [row["FRAMETYPES"].strip().split() for row in d]
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
             group1=group1,  # d["GROUP1"],
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
        # self.set_index("starting_obsid", inplace=True)

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

        # print m_reversed
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


def _test3():
    fn = "../../recipe_logs/20150120.recipes"
    # r = load_recipe(fn)
    recipes = Recipes(fn)

    recipe_name = "SKY*"
    groups_parsed = None
    selected = recipes.select_fnmatch_by_groups(recipe_name,
                                                groups_parsed)
    print(selected)


def _test4():
    # names = [_[0] for _ in dtypes]
    fn = "../../recipe_logs/20150120.recipes"
    dtypes = [('OBJNAME', 'U128'), ('OBJTYPE', 'U128'),
              ('GROUP1', 'U128'), ('GROUP2', 'U128'),
              ('EXPTIME', 'f'), ('RECIPE', 'U128'),
              ('OBSIDS', 'U1024'),  ('FRAMETYPES', 'U1024')]

    df = pd.read_csv(fn, skiprows=0, dtype=dict(dtypes), comment="#",
                     escapechar="\\", skipinitialspace=True,
                     engine="python")
    print(df)

if __name__ == "__main__":
    _test3()

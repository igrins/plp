import numpy as np

from itertools import cycle


def make_tuple(row):
    return [(row["GROUP1"],
             row["EXPTIME"],
             row["FRAMETYPES"].count("OFF"),
             row["FRAMETYPES"].count("ON"),
             len(row["OBSIDS"]))]


def in_between(v, vmin, vmax):
    return vmin <= v <= vmax


def get_first_group(group_exptime_frametypes,
                    exptime_min_max=None,
                    off_count_min_max=None, on_count_min_max=None,
                    obsid_count_min_max=None):
    for (g1, exptime, off_count, on_count,
         obsid_count) in group_exptime_frametypes:
        if(exptime_min_max is not None and
           not in_between(exptime, *exptime_min_max)):
            continue
        elif (off_count_min_max is not None and
              not in_between(off_count, *off_count_min_max)):
            continue
        elif (on_count_min_max is not None and
              not in_between(on_count, *on_count_min_max)):
            continue
        elif (obsid_count_min_max is not None and
              not in_between(obsid_count, *obsid_count_min_max)):
            continue

        group = g1
        break
    else:
        group = ""

    return group


def split_obsids_frametypes(df_recipe_logs, copy=True):
    if copy:
        df_recipe_logs = df_recipe_logs.copy()

    ss = df_recipe_logs["OBSIDS"].astype("U").str.split()
    df_recipe_logs["OBSIDS"] = [list(map(int, s)) for s in ss]
    df_recipe_logs["FRAMETYPES"] = df_recipe_logs["FRAMETYPES"].str.split()

    return df_recipe_logs


class RecipeSummary(object):
    def __init__(self, obsdate, df):
        self.obsdate = obsdate
        self.df = df  # split_obsids_frametypes(df)
        self.grouped = self.df.groupby("RECIPE")

    def get_recipe_count(self):
        recipe_count = dict(A0V=0,
                            A0V_AB=0,
                            DARK=0,
                            DEFAULT=0,
                            FLAT=0,
                            STELLAR=0,
                            EXTENDED=0,
                            ARC=0,
                            ARC_THAR=0,
                            ARC_UNE=0,
                            SKY=0)

        k = self.grouped["RECIPE"].count().to_dict()

        for recipe_root in ["A0V", "STELLAR", "EXTENDED"]:
            k[recipe_root] = (k.get(recipe_root + "_AB", 0) +
                              k.get(recipe_root + "_ONOFF", 0))
        recipe_count.update(k)

        return recipe_count

    def get_group(self, recipe_name):
        if recipe_name not in self.grouped.groups:
            _r = []
        else:
            _r = self.grouped.get_group(recipe_name).apply(make_tuple,
                                                           axis=1).sum()
            # flats.sort()

        return _r

    def sky_off(self):
        df = self.df
        msk = ((df["OBJNAME"] == "SKY") |
               (df["OBJTYPE"] == "SKY"))

        return df[msk]

    @staticmethod
    def _frametype_ab(row):
        return all(a == b for (a, b) in zip(row, cycle("ABBA")))

    def sky_ab(self, obstype="STD", min_exptime=30, max_exptime=150):
        df = self.df
        msk = (((df["OBJTYPE"] == obstype))  # | (df["OBJTYPE"] == "TAR"))
               & (df["EXPTIME"] >= min_exptime)
               & (df["EXPTIME"] <= max_exptime)
               & (df["OBSIDS"].apply(len) % 2 == 0)
               & df["FRAMETYPES"].apply(self._frametype_ab))

        return df[msk]

    def select_master_dark(self, exptime_min_max=(20, 40),
                           obsid_count_min_max=(3, np.inf)):
        _r = self.get_group("DARK")

        group1 = get_first_group(_r, exptime_min_max=exptime_min_max,
                                 obsid_count_min_max=obsid_count_min_max)

        return "DARK", group1

    def select_master_flat(self, exptime_min_max=(5, 400),
                           off_count_min_max=(5, np.inf),
                           on_count_min_max=(5, np.inf)):
        _r = self.get_group("FLAT")

        group1 = get_first_group(_r, exptime_min_max=exptime_min_max,
                                 off_count_min_max=off_count_min_max,
                                 on_count_min_max=on_count_min_max)

        return "FLAT", group1

    def select_master_sky(self, obstype="STD",
                          min_exptime=30, max_exptime=150):
        df_sky = self.sky_off()
        if len(df_sky):
            sky_recipe = "SKY"
        else:
            df_sky = self.sky_ab(obstype, min_exptime, max_exptime)
            if len(df_sky):
                sky_recipe = "SKY_AB"
            else:
                return "", 0

        i = len(df_sky) // 2
        sky_group = df_sky.iloc[i]["GROUP1"]

        return sky_recipe, sky_group

    def select_arc(self):
        """empirical way of selecting ThAr & UNe lamps"""

        arc_recipes = {}

        # arc_msk = (df["RECIPE"] == "ARC")
        for rownum, arc_row in self.df.iterrows():
            if(arc_row["RECIPE"] == "ARC"
               and arc_row["EXPTIME"] == 15
               and len(arc_row["OBSIDS"]) == 6):
                arc_recipes["THAR"] = arc_row["OBSIDS"][:3]
                arc_recipes["UNE"] = arc_row["OBSIDS"][3:]
                print(arc_row["OBSDATE"])
            # elif arc_row["RECIPE"] == "ARC":
            #       print(arc_row)
            elif arc_row["RECIPE"] == "ARC_THAR":
                arc_recipes["THAR"] = arc_row["OBSIDS"]
            elif arc_row["RECIPE"] == "ARC_UNE":
                arc_recipes["UNE"] = arc_row["OBSIDS"]
            else:
                pass

        return arc_recipes

    def get_summary(self):

        arc_recipes = self.select_arc()

        return dict(obsdate=self.obsdate,
                    dark_group=self.select_master_dark(),
                    darks=self.get_group("DARK"),
                    flat_group=self.select_master_flat(),
                    flats=self.get_group("FLAT"),
                    sky_group=self.select_master_sky(),
                    sky_abs=list(self.sky_ab()["GROUP1"]),
                    arc_thar=arc_recipes.get("THAR", []),
                    arc_une=arc_recipes.get("UNE", []),
                    **self.get_recipe_count())

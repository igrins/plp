import os
# from contextlib import chdir
from pathlib import Path
import re

import igrins
from igrins.igrins_libs.recipes import RecipeLog
from igrins.igrins_libs.igrins_config import IGRINSConfig as IGRINSConfig_
from ..external import argh

p_group1 = re.compile(r"(\d+)(.*)")

class IGRINSConfig(IGRINSConfig_):
    def load_recipe_log(self, obsdate):
        return igrins.load_recipe_log(obsdate, config_file=self)

    def save_recipe_log(self, obsdate: str, recipe_log: RecipeLog):

        fnout = self.get_value('RECIPE_LOG_PATH', utdate=obsdate)
        recipe_log.write_to_file(fnout, lower_colnames=True)

    def get_obsset(self, obsdate, band, recipe_name_or_entry,
                   obsids=None, frametypes=None,
                   groupname=None, recipe_entry=None):

        return igrins.get_obsset(obsdate, band, recipe_name_or_entry,
                                 obsids=obsids, frametypes=frametypes,
                                 groupname=groupname, recipe_entry=recipe_entry,
                                 config_file=self)


def _link_calib(utdate_src: str, utdate_dst: str, recipe_name: str, s_obsid: int=9000,
                igrins_config: IGRINSConfig | None =None):
    # UTDATE_SRC, UTDATE_DST = "20240429", "20240427"
    # s_obsid = 9000
    # recipe_name = "FLAT"

    if igrins_config is None:
        igrins_config = IGRINSConfig()

    rl_source = igrins_config.load_recipe_log(utdate_src)
    rl_target = igrins_config.load_recipe_log(utdate_dst)


    # with chdir()
    utdate = igrins_config.get("DEFAULT", "INDATA_PATH", UTDATE=utdate_src)
    utdate2 = igrins_config.get("DEFAULT", "INDATA_PATH", UTDATE=utdate_dst)
    utdate_rel = os.path.relpath(utdate, utdate2)

    # rows = []
    for i, row in rl_source.subset(recipe=recipe_name).iterrows():
        row2 = row.copy()
        row2["obsids"] = [str(s_obsid + int(o)) for o in row["obsids"]]
        row2["starting_obsid"] = str(s_obsid + int(row["starting_obsid"]))
        if row["group1"] != "1":
            if (m := p_group1.match(row["group1"])):
                obsid, other = m.groups()
                row2["group1"] = str(s_obsid + int(obsid)) + other
            else:
                print("group1 not changed")

        rl_target = rl_target._append(row2, ignore_index=True)

        # make links

        for band in "HK":
            for obsid, obsid2 in zip(row["obsids"], row2["obsids"]):
                # for i in range(11, 31):
                fin = f"SDC{band}_{utdate_src}_{int(obsid):04d}.fits"
                fout = f"SDC{band}_{utdate_dst}_{int(obsid2):04d}.fits"
                src, dst = Path(utdate_rel) / fin, Path(utdate2) / fout
                os.symlink(src, dst)
                print(src, "->", dst)

    igrins_config.save_recipe_log(utdate_dst, rl_target)


def link_calib(utdate_src: str, utdate_dst: str, recipe_name: str, s_obsid: int=9000,
               config_file="recipe.config"):
    igrins_config = IGRINSConfig(config_file)

    _link_calib(utdate_src, utdate_dst, recipe_name, s_obsid,
                igrins_config)



def main():
    import sys
    utdate_src, utdate_dst, recipe_name, s_obsid = sys.argv[1:]
    igrins_config = IGRINSConfig()
    _link_calib(utdate_src, utdate_dst, recipe_name, s_obsid=int(s_obsid),
                igrins_config=igrins_config)


if __name__ == '__main__':
    main()

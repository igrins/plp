import os
import numpy as np

def prepare_recipe_logs(utdate, config_file="recipe.config"):

    from libs.igrins_config import IGRINSConfig
    config = IGRINSConfig(config_file)

    fn0 = config.get_value('INDATA_PATH', utdate)
    fn = os.path.join(fn0, "IGRINS_DT_Log_%s-1_H.txt" % (utdate,))

    # p_end_comma = re.compile(r",\s$")
    # s = "".join(p_end_comma.sub(",\n", l) for l in lines)

    #s = "".join(lines)

    # dtype=[('FILENAME', 'S128'), ('OBSTIME', 'S128'), ('GROUP1', 'i'), ('GROUP2', 'i'), ('OBJNAME', 'S128'), ('OBJTYPE', 'S128'), ('FRAMETYPE', 'S128'), ('EXPTIME', 'd'), ('ROTPA', 'd'), ('RA', 'S128'), ('DEC', 'S128'), ('AM', 'd')]

    # log file format for March and May, July is different.
    dtype_=[('FILENAME', 'S128'),
            ('OBSTIME', 'S128'),
            ('GROUP1', 'i'),
            ('GROUP2', 'i'),
            ('OBJNAME', 'S128'),
            ('OBJTYPE', 'S128'),
            ('FRAMETYPE', 'S128'),
            ('EXPTIME', 'd'),
            ('ROTPA', 'd'),
            ('RA', 'S128'),
            ('DEC', 'S128'),
            ('AM', 'd'),
            ('OBSDATE', 'S128'),
            ('SEQID1', 'i'),
            ('SEQID2', 'i'),
            ('ALT', 'd'),
            ('AZI', 'd'),
            ('OBSERVER', 'S128'),
            ('EPOCH', 'S128'),
            ('AGPOS', 'S128'),
            ]
    dtype_map = dict(dtype_)

    dtype_replace = dict(SEQID1="GROUP1", SEQID2="GROUP2")

    lines = open(fn).readlines()
    stripped_lines = [s1.strip() for s1 in lines[1].split(",")]
    dtype = [(dtype_replace.get(s1, s1), dtype_map[s1]) for s1 in stripped_lines if s1]

    l = np.genfromtxt(fn,
                      skip_header=2, delimiter=",", dtype=dtype)

    from itertools import groupby

    groupby_keys = ["OBJNAME", "OBJTYPE", "GROUP1", "GROUP2", "EXPTIME"]
    def keyfunc(l1):
        return tuple(l1[k] for k in groupby_keys)

    s_list = []
    for lll in groupby(l, keyfunc):
        grouper = list(lll[1])
        obsids = [int(lll1[0].split(".")[0].split("_")[-1]) for lll1 in grouper]
        frametypes = [lll1["FRAMETYPE"]  for lll1 in grouper]

        objtype = lll[0][1]
        if objtype.lower() == "flat":
            recipe = "FLAT"
        elif objtype.lower() == "std":
            recipe = "A0V_AB"
        elif objtype.lower() == "tar":
            recipe = "STELLAR_AB"
        else:
            recipe = "DEFAULT"

        s1 = "%s, %s, %d, %d, %f," % lll[0]
        s2 = "%s, %s, %s\n" % (recipe,
                               " ".join(map(str,obsids)),
                               " ".join(frametypes),
                               )
        s_list.append(s1+" "+s2)


    headers = groupby_keys + ["RECIPE", "OBSIDS", "FRAMETYPES"]

    recipe_log_name = config.get_value('RECIPE_LOG_PATH', utdate)
    fn_out = recipe_log_name + ".tmp"

    fout = open(fn_out, "w")
    fout.write(", ".join(headers) + "\n")
    fout.write("# Avaiable recipes : FLAT, THAR, SKY, A0V_AB, A0V_ONOFF, STELLAR_AB, STELLAR_ONOFF, EXTENDED_AB, EXTENDED_ONOFF\n")

    fout.writelines(s_list)
    fout.close()

    print "A draft version of the recipe log is written to '%s'." % (fn_out,)
    print "Make an adjusment and rename it to '%s'." % (recipe_log_name,)

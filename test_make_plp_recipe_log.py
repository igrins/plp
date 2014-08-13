import numpy as np

fn = "indata/20140525/IGRINS_DT_Log_20140525-1_H.txt"

dtype=[('FILENAME', 'S128'), ('OBSTIME', 'S128'), ('GROUP1', 'i'), ('GROUP2', 'i'), ('OBJNAME', 'S128'), ('OBJTYPE', 'S128'), ('FRAMETYPE', 'S128'), ('EXPTIME', 'd'), ('ROTPA', 'd'), ('RA', 'S128'), ('DEC', 'S128'), ('AM', 'd')]

l = np.genfromtxt(fn, names=True, skip_header=1, delimiter=",", dtype=dtype)

from itertools import groupby

groupby_keys = ["OBJNAME", "OBJTYPE", "GROUP1", "GROUP2", "EXPTIME"]
def keyfunc(l1):
    return tuple(l1[k] for k in groupby_keys)

s_list = []
for lll in groupby(l, keyfunc):
    obsids = [int(lll1[0].split(".")[0].split("_")[-1]) for lll1 in lll[1]]
    objtype = lll[0][1]
    if objtype.lower() == "std" and len(obsids) == 4:
        recipe = "A0V_ABBA"
    elif len(obsids) == 4:
        recipe = "STELLAR_ABBA"
    elif len(obsids) == 2:
        recipe = "EXTENDED_AB"
    else:
        recipe = "DEFAULT"

    s1 = "%s, %s, %d, %d, %f," % lll[0]
    s2 = "%s, %s\n" % (recipe, " ".join(map(str,obsids)))
    s_list.append(s1+s2)


headers = groupby_keys + ["RECIPE", "OBSIDS"]

fout = open("20140525.recipes.tmp", "w")
fout.write(" ".join(headers) + "\n")
fout.writelines(s_list)
fout.close()

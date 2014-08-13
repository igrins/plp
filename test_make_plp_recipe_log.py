import numpy as np

utdate = "20140525"

fn = "indata/%s/IGRINS_DT_Log_%s-1_H.txt" % (utdate, utdate)

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

fout = open("%s.recipes.tmp" % utdate, "w")
fout.write(", ".join(headers) + "\n")
fout.write("# Avaiable recipes : FLAT_OFF, FLAT_ON, THAR, SKY, A0V_ABBA, STELLAR_ABBA, EXTENDED_AB")

fout.writelines(s_list)
fout.close()

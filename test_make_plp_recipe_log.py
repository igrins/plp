import numpy as np
import re

utdate = "20140713"

fn = "indata/%s/IGRINS_DT_Log_%s-1_H.txt" % (utdate, utdate)
p_end_comma = re.compile(r",$")
s = "".join(p_end_comma.sub("", l) for l in open(fn))

dtype=[('FILENAME', 'S128'), ('OBSTIME', 'S128'), ('GROUP1', 'i'), ('GROUP2', 'i'), ('OBJNAME', 'S128'), ('OBJTYPE', 'S128'), ('FRAMETYPE', 'S128'), ('EXPTIME', 'd'), ('ROTPA', 'd'), ('RA', 'S128'), ('DEC', 'S128'), ('AM', 'd')]

from StringIO import StringIO
l = np.genfromtxt(StringIO(s),
                  names=True, skip_header=1, delimiter=",", dtype=dtype)

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

fout = open("%s.recipes.tmp" % utdate, "w")
fout.write(", ".join(headers) + "\n")
fout.write("# Avaiable recipes : FLAT, THAR, SKY_WVLSOL, A0V_AB, STELLAR_AB, EXTENDED_AB, EXTENDED_ONOFF\n")

fout.writelines(s_list)
fout.close()

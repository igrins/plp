import numpy as np
import re

utdate = "20140710"

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

#s_list = []

import datetime

today = datetime.date.today()
last_end_time = None

for lll in groupby(l, keyfunc):
    grouper = list(lll[1])
    obsids = [int(lll1[0].split(".")[0].split("_")[-1]) for lll1 in grouper]
    frametypes = [lll1["FRAMETYPE"]  for lll1 in grouper]

    objtype = lll[0][1]

    abba_start_time_ = datetime.time(*map(int, grouper[0][1].split(":")))
    abba_end_time_ = datetime.time(*map(int, grouper[-1][1].split(":")))


    abba_start_time = datetime.datetime.combine(today, abba_start_time_)
    abba_end_time = datetime.datetime.combine(today, abba_end_time_)

    if len(grouper) > 1:
        abba_end2_time_ = datetime.time(*map(int, grouper[-2][1].split(":")))
        abba_end2_time = datetime.datetime.combine(today, abba_end2_time_)

        exptime = abba_end_time - abba_end2_time
    else:
        exptime = datetime.timedelta(seconds=float(grouper[0][7]))

    abba_end_time_real = abba_end_time + exptime

    #print grouper[0][4], abba_start_time, abba_end_time

    if objtype.lower() in ["std", "tar"]:
        #print grouper[0][4], abba_start_time, abba_end_time

        if last_end_time:
            print grouper[0][4], abba_start_time - last_end_time

    last_end_time = abba_end_time_real

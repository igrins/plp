import os
import pandas as pd

import glob

from ..igrins_libs import logger


def convert_group_values(groups):

    _cached_g = None

    new_groups = []

    for g in groups:
        # we assume that g is string
        try:
            from_cached = int(g) < 0
        except ValueError:
            from_cached = False

        if from_cached:
            if _cached_g is None:
                raise RuntimeError("Negative group values are "
                                   "only allowed if previous "
                                   "group value has been defined")
        else:
            _cached_g = g

        new_groups.append(_cached_g)

    return new_groups


def _load_data_pandas(fn):
    dtype_ = [('FILENAME', str),
              ('OBSTIME', str),
              ('GROUP1', str),
              ('GROUP2', str),
              ('OBJNAME', str),
              ('OBJTYPE', str),
              ('FRAMETYPE', str),
              ('EXPTIME', 'd'),
              ('ROTPA', 'd'),
              ('RA', str),
              ('DEC', str),
              ('AM', 'd'),
              ('OBSDATE', str),
              ('SEQID1', 'i'),
              ('SEQID2', 'i'),
              ('ALT', 'd'),
              ('AZI', 'd'),
              ('OBSERVER', str),
              ('EPOCH', str),
              ('AGPOS', str)]

    dtype_map = dict(dtype_)
    dtype_replace = dict(SEQID1="GROUP1", SEQID2="GROUP2")

    lines = open(fn).readlines()
    stripped_lines = [s1.strip() for s1 in lines[1].split(",")]
    dtype = [(dtype_replace.get(s1, s1), dtype_map[s1])
             for s1 in stripped_lines if s1]

    names = [_[0] for _ in dtype]
    dtypes = dict(dtype)

    from six import StringIO
    import re
    p = re.compile("PY_VAR\d+")
    ss = p.sub("NaN", open(fn).read())
    open("ttt.txt", "w").write(ss)
    s = StringIO(ss)

    df = pd.read_csv(s, skiprows=2, dtype=dtypes, comment="#",
                     names=names, escapechar="\\", na_values=["PY_VAR0"])

    df["OBJNAME"] = [(s.replace(",", "\\,") if (s == s) else s)
                     for s in df["OBJNAME"]]

    df["GROUP1"] = convert_group_values(df["GROUP1"])
    df["GROUP2"] = convert_group_values(df["GROUP2"])

    # l = df.to_records(index=False)
    # l = np.genfromtxt(fn,
    #                   skip_header=2, delimiter=",", dtype=dtype)
    return df


_load_data = _load_data_pandas


def load_from_fn_list(obsdate, fn_list):
    l_list = [_load_data(fn) for fn in fn_list]
    l = pd.concat(l_list, ignore_index=True)

    obsids = get_unique_obsid(obsdate, l)
    l["OBSID"] = obsids
    l["OBSDATE"] = obsdate

    return l


def load_from_dir(obsdate, dir, debug=False):
    # there could be two log files!
    fn_list = glob.glob(os.path.join(dir, "IGRINS_DT_Log_*-1_H.txt"))
    fn_list.sort()

    logger.info("loading DT log files: {}".format(fn_list))

    return load_from_fn_list(obsdate, fn_list)


def get_unique_obsid(obsdate, l, bands="HK"):
    obsdate0 = int(obsdate)

    def parse_obsdate_obsid(fn):
        obsdate1, obsid = fn.split(".")[0].split("_")[-2:]
        return int(obsdate1), int(obsid)

    obsdate_obsid_list = [parse_obsdate_obsid(l1[0])
                          for i, l1 in l.iterrows()]

    maxobsid = max(obsid for obsdate, obsid in obsdate_obsid_list
                   if obsdate == obsdate0)

    from itertools import count
    new_obsid_count = count(maxobsid + 1)
    new_obsid = []
    for obsdate, obsid in obsdate_obsid_list:
        if obsdate == obsdate0:
            new_obsid.append(obsid)
        else:
            new_obsid.append(next(new_obsid_count))

    return new_obsid


def get_obsid_map(obsdate, l, bands="HK"):
    # prepare the convertsion map between obsids with different obsdate
    # to the current obsdate.
    obsdate0 = int(obsdate)
    obsid_link_list = []

    maxobsid = 0

    for i, l1 in l.iterrows():
        try:
            fn = l1[0]
            obsdate1, obsid = fn.split(".")[0].split("_")[-2:]

            if int(obsdate1) != obsdate0:
                obsid_link_list.append((obsdate1, int(obsid), fn))
            else:
                maxobsid = max(maxobsid, int(obsid))
        except:
            logger.logger.error("Error on : {}".format(l1))
            raise

    obsid_map = {}
    for i, (obsdate1, obsid, fn) in enumerate(obsid_link_list, start=1):
        obsid2 = maxobsid + i
        obsid_map[(obsdate1, obsid)] = obsid2

    return obsid_map


def update_file_links(obsdate, l, bands="HK"):
    # prepare the convertsion map between obsids with different obsdate
    # to the current obsdate.
    obsid_map = get_obsid_map(obsdate, l, bands=bands)

    if obsid_map:
        print("trying to make soft links for files of different obsdates")
        from igrins.libs.load_fits import find_fits
        old_new_list = []

        for band in bands:
            for k, v in obsid_map.items():
                obsdate1, obsid = k
                tmpl = "SDC%s_%s_%04d" % (band, obsdate1, obsid)
                # old_fn = os.path.join(fn0, tmpl+".fits")
                old_fn = tmpl + ".fits"
                old_fn = find_fits(old_fn)

                newtmpl = "SDC%s_%s_%04d" % (band, obsdate, v)
                _newfn = os.path.basename(old_fn).replace(tmpl, newtmpl)
                new_fn = os.path.join(os.path.dirname(old_fn),
                                      _newfn)

                old_new_list.append((old_fn, new_fn))

        for old_fn, new_fn in old_new_list:
            if os.path.exists(new_fn):
                if os.path.islink(new_fn):
                    os.unlink(new_fn)
                else:
                    raise RuntimeError("file already exists: %s" % new_fn)

        for old_fn, new_fn in old_new_list:
            os.symlink(os.path.basename(old_fn), new_fn)



# if False:
#     rootdir = "/home/jjlee/annex/igrins/"
#     fn_list = [os.path.join(rootdir,
#                             "20170315/IGRINS_DT_Log_20170315-1_H.txt")]
#     d2 = load_from_fn_list(fn_list)

def test():
    # obsdate = "20140208"
    # dirname = "2014T2/20140208"
    dirname = '2014T3/20141009'
    dirname = '2015T1/20150106'
    dirname = '2016T2/20160723'
    dirname = '2016T3/20160926'
    dirname = '2016T3/20161003'
    dirname = '2017T1/20170312'
    dirname = '2017T1/20170314'
    dirname = '2017T1/20170315'
    obsdate = os.path.basename(dirname)
    d3 = load_from_dir(obsdate, dirname)
    d3.to_json("test.json", orient="records")


if __name__ == "__main__":
    test2()


# if __name__ == "__main__":
#     fn = "/home/jjlee/annex/igrins/20170315/IGRINS_DT_Log_20170315-1_H.txt"
#     d1 = _load_data_pandas(fn)

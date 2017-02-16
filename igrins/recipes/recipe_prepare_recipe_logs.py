import os
import numpy as np

def prepare_recipe_logs(utdate, config_file="recipe.config"):

    from igrins.libs.igrins_config import IGRINSConfig
    config = IGRINSConfig(config_file)

    fn0 = config.get_value('INDATA_PATH', utdate)

    if not os.path.exists(fn0):
        raise RuntimeError("directory {} does not exist.".format(fn0))

    # there could be two log files!
    import glob
    fn_list = glob.glob(os.path.join(fn0, "IGRINS_DT_Log_*-1_H.txt"))
    fn_list.sort()
    print "loading DT log files:", fn_list

    #fn = os.path.join(fn0, "IGRINS_DT_Log_%s-1_H.txt" % (utdate,))

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

    def load_data(fn):
        lines = open(fn).readlines()
        stripped_lines = [s1.strip() for s1 in lines[1].split(",")]
        dtype = [(dtype_replace.get(s1, s1), dtype_map[s1]) for s1 in stripped_lines if s1]

        l = np.genfromtxt(fn,
                          skip_header=2, delimiter=",", dtype=dtype)
        return l

    l_list = [load_data(fn) for fn in fn_list]
    l = np.concatenate(l_list)

    from itertools import groupby

    groupby_keys = ["OBJNAME", "OBJTYPE", "GROUP1", "GROUP2", "EXPTIME"]
    # If the OBJTYPE is flat, we replace its name ot "FLAT ON/OFF" so
    # that it can be assembled in a single recipe even tough their
    # names are different.
    def keyfunc(l1):
        if l1["OBJTYPE"].lower() == "flat":
            objname_tuple = ("FLAT ON/OFF", )
        else:
            objname_tuple = (l1["OBJNAME"], )

        return objname_tuple + tuple(l1[k] for k in groupby_keys[1:])

    # prepare the convertsion map between obsids with different utdate
    # to the current utdate.
    utdate0 = int(utdate)
    obsid_link_list = []

    maxobsid = 0

    for l1 in l:
        fn = l1[0]
        utdate1, obsid = fn.split(".")[0].split("_")[-2:]

        if int(utdate1) != utdate0:
            obsid_link_list.append((utdate1, int(obsid)))
        else:
            maxobsid = max(maxobsid, int(obsid))

    obsid_map = {}
    for i, (utdate1, obsid) in enumerate(obsid_link_list, start=1):
        obsid2 = maxobsid + i
        obsid_map[(utdate1, obsid)] = obsid2


    if obsid_map:
        print "trying to make soft links for files of different utdates"
        from igrins.libs.load_fits import find_fits
        old_new_list = []

        for band in "HK":
            for k, v in obsid_map.items():
                utdate1, obsid = k
                tmpl = "SDC%s_%s_%04d" % (band, utdate1, obsid)
                old_fn = os.path.join(fn0, tmpl+".fits")
                old_fn = find_fits(old_fn)

                newtmpl = "SDC%s_%s_%04d" % (band, utdate, v)
                new_fn = os.path.join(os.path.dirname(old_fn),
                                      os.path.basename(old_fn).replace(tmpl, newtmpl))

                old_new_list.append((old_fn, new_fn))

        for old_fn, new_fn in old_new_list:
            if os.path.exists(new_fn):
                if os.path.islink(new_fn):
                    os.unlink(new_fn)
                else:
                    raise RuntimeError("file already exists: %s" % new_fn)

        for old_fn, new_fn in old_new_list:
            os.symlink(os.path.basename(old_fn), new_fn)

    # now prepare the recipe_logs.
    s_list = []
    for lll in groupby(l, keyfunc):
        grouper = list(lll[1])
        utdate_obsids = [lll1[0].split(".")[0].split("_")[-2:]
                         for lll1 in grouper]

        # convert (utdate, obsid) tuple in the form of 12 or 1:12
        obsids = [obsid_map.get((utdate1, int(obsid)), int(obsid))
                  for (utdate1, obsid) in utdate_obsids]

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

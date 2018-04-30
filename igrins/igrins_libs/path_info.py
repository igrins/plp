
import os
from os.path import join
# import astropy.io.fits as pyfits

import re
from ..utils.file_utils import ensure_dir


groupname_pattern = re.compile(r"(\d+)(\D.*)?")


def get_zeropadded_groupname(groupname):
    if isinstance(groupname, int):
        groupname = "%04d" % groupname
    else:
        m = groupname_pattern.match(groupname)
        if m:
            m1, m2 = m.groups()
            groupname = "%04d%s" % (int(m1), m2 if m2 else "")
        else:
            pass

    return groupname


class IGRINSBasenameHelper(object):

    def __init__(self, obsdate, band):
        self.obsdate = obsdate
        self.band = band

        prefix = "SDC{band}_{obsdate}".format(obsdate=obsdate,
                                              band=band)

        self.basename_pattern = prefix + "_{obsid:04d}"

    def get_section_n_fn(self, obsid, item_desc, postfix=""):
        basename = self.basename_pattern.format(obsid=obsid)
        (section, tmpl) = item_desc
        fn = tmpl.format(basename=basename, postfix=postfix)

        return section, fn


class IGRINSPath(object):
    # IGRINS_CALIB_PATH="calib"
    # IGRINS_REDUCE_DATE=""
    # IGRINS_INDATA_PATH="indata"
    # IGRINS_OUTDATA_PATH="outdata"
    # IGRINS_QA_PATH="outdata/qa"

    sections_names = ["INDATA_PATH",
                      "OUTDATA_PATH",
                      "PRIMARY_CALIB_PATH",
                      "SECONDARY_CALIB_PATH",
                      "QA_PATH",
                      "HTML_PATH"]

    default_paths = dict(QL_PATH="{OUTDATA_PATH}/quicklook")

    sections_names_no_ensuredir = ["INDATA_PATH"]

    def __init__(self, config, obsdate, band, ensure_dir=False):

        self.config = config
        self.obsdate = obsdate
        self.band = band

        self.sections = dict()

        for n in self.sections_names:
            d = self.config.get_value(n, obsdate)
            self.sections[n] = join(self.config.root_dir, d)

        for n in self.default_paths:
            self.sections[n] = self.default_paths[n].format(**self.sections)

        # filename pattern for input files
        self.fn_pattern = join(self.sections["INDATA_PATH"],
                               "SDC%%s_%s_%%s.fits" % (self.obsdate,))

        self.basename_pattern = "SDC%%s_%s_%%s" % (self.obsdate,)

        if ensure_dir:
            self.ensure_dir()

    def ensure_dir(self):
        for k, d in self.sections.items():
            if k not in self.sections_names_no_ensuredir:
                ensure_dir(d)

    def get_section_path(self, section):
        return self.sections[section]

    def get_section_filename_base(self, section, fn, subdir=None):
        if subdir is not None:
            dirpath = join(self.sections[section],
                           subdir)
            ensure_dir(dirpath)
        else:
            dirpath = self.sections[section]

        return join(dirpath,
                    os.path.basename(fn))

    # def get_outdata_filename(self, fn):
    #     return join(self.outdata_path,
    #                 os.path.basename(fn))

    def get_secondary_calib_filename(self, fn, subdir=None):
        p = self.get_section_filename_base("SECONDARY_CALIB_PATH",
                                           fn, subdir)
        return p

        # if subdir is not None:
        #     dirpath = join(self.secondary_calib_path,
        #                    subdir)
        #     ensure_dir(dirpath)
        # else:
        #     dirpath = self.secondary_calib_path
        # return join(dirpath,
        #             os.path.basename(fn))

    # def get_filenames(self, runids):
    #     return [self.get_filename(band, i) for i in runids]

    def get_filename(self, runid):
        groupname = get_zeropadded_groupname(runid)
        return self.fn_pattern % (self.band, groupname)

    def get_basename(self, groupname):
        groupname = get_zeropadded_groupname(groupname)
        basename = self.basename_pattern % (self.band, groupname)
        return basename

    # def get_hdus(self, runids):
    #     fn_list = self.get_filenames(runids)
    #     hdu_list = [pyfits.open(fn)[0] for fn in fn_list]
    #     return hdu_list




# class IGRINSLog(object):
#     def __init__(self, igrins_path, log_dict):
#         self.log = log_dict
#         self.date = igrins_path.utdate
#         self.fn_pattern = join(igrins_path.indata_path,
#                                "SDC%%s_%s_%%04d.fits" % (self.date,))

#     def get_filenames(self, band, runids):
#         return [self.get_filename(band, i) for i in runids]

#     def get_filename(self, band, runid):
#         return self.fn_pattern % (band, runid)

#     def get_cal_hdus(self, band, cal_name):
#         fn_list = [self.get_filename(band, runid) for runid in self.log[cal_name]]
#         hdu_list = [pyfits.open(fn)[0] for fn in fn_list]
#         return hdu_list


class IGRINSFiles:
    pass

if __name__ == "__main__":
    pass

#ip = IGRINSPath("20140525")

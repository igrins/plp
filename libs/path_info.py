
import os
from os.path import join
import libs.fits as pyfits

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

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

    sections_names_no_ensuredir = ["INDATA_PATH"]

    def __init__(self, config, utdate):

        self.config = config
        self.utdate = utdate

        self.sections = dict()

        for n in self.sections_names:
            d = self.config.get_value(n, utdate)
            self.sections[n] = join(self.config.root_dir, d)

        for k, d in self.sections.items():
            if k not in self.sections_names_no_ensuredir:
                ensure_dir(d)


        # filename pattern for input files
        self.fn_pattern = join(self.sections["INDATA_PATH"],
                               "SDC%%s_%s_%%04d.fits" % (self.utdate,))

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


    def get_filenames(self, band, runids):
        return [self.get_filename(band, i) for i in runids]

    def get_filename(self, band, runid):
        return self.fn_pattern % (band, runid)

    def get_hdus(self, band, runids):
        fn_list = self.get_filenames(band, runids)
        hdu_list = [pyfits.open(fn)[0] for fn in fn_list]
        return hdu_list




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

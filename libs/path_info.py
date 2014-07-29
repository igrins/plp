
import os
from os.path import join

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

class IGRINSPath(object):
    IGRINS_CALIB_PATH="calib"
    IGRINS_REDUCE_DATE=""
    IGRINS_INDATA_PATH="indata"
    IGRINS_OUTDATA_PATH="outdata"
    IGRINS_QA_PATH="outdata/qa"

    def __init__(self, utdate):
        self.utdate = utdate

        self.indata_path = join(self.IGRINS_INDATA_PATH,
                                utdate)
        self.primary_calib_path = join(self.IGRINS_CALIB_PATH,
                                       utdate,
                                       "primary")
        self.secondary_calib_path = join(self.IGRINS_CALIB_PATH,
                                         utdate,
                                         "secondary")
        self.qa_path = join(self.IGRINS_QA_PATH,
                             utdate)
        self.outdata_path = join(self.IGRINS_OUTDATA_PATH,
                                 utdate)

        for d in [self.primary_calib_path,
                  self.secondary_calib_path,
                  self.qa_path,
                  self.outdata_path]:
            ensure_dir(d)

    def get_outdata_filename(self, fn):
        return join(self.outdata_path,
                    os.path.basename(fn))

    def get_secondary_calib_filename(self, fn):
        return join(self.secondary_calib_path,
                    os.path.basename(fn))

import astropy.io.fits as pyfits

class IGRINSLog(object):
    def __init__(self, igrins_path, log_dict):
        self.log = log_dict
        self.date = igrins_path.utdate
        self.fn_pattern = join(igrins_path.indata_path,
                               "SDC%%s_%s_%%04d.fits" % (self.date,))

    def get_filename(self, band, runid):
        return self.fn_pattern % (band, runid)

    def get_cal_hdus(self, band, cal_name):
        fn_list = [self.get_filename(band, runid) for runid in self.log[cal_name]]
        hdu_list = [pyfits.open(fn)[0] for fn in fn_list]
        return hdu_list



if __name__ == "__main__":
    ip = IGRINSPath("20140525")


from Tkinter        import *
import ttk

from PL_Frame      import *

from PL_Display import CDisplay

import os.path

class CDisplayTest(CDisplay):

    def StatusInsert(self, *kl, **kwargs):
        pass

    def run_test(self, workdir):
        workdir=os.path.abspath(workdir)
        logname="IGRINS_DT_Log_20140316-1_H.txt"

        if not workdir.endswith("/"):
            workdir = workdir + "/"
        self.reducepath = workdir
        filenamefullpath = os.path.join(workdir, logname)

        filename = self.get_filename(filenamefullpath)
        self.set_loadlogpath(filename, filenamefullpath)

        self.fileprefix = "SDC"+self.band+"_"+cdisplay.obsid+"_" #[To be removed]

        self.datpath = self.reducepath + 'dat/'

        #self.reduce_cal_dark()
        print self.item_list

        item_list = self.item_list["FLAT"]
        ntotal = len(item_list)

        cal_flat = dict(method="med", # or "aver"
                        sigma="2.0",
                        clip=False)


        self.reduce_cal_flat_real(item_list, ntotal, "ON", "OFF",
                                  cal_flat=cal_flat,
                                  do_bp=False)


def get_difference(workdir, fn):
    import astropy.io.fits as pyfits
    f1 = os.path.join(workdir,fn)
    f2 = os.path.join("./test_references", workdir, fn)
    return pyfits.open(f1)[0].data - pyfits.open(f2)[0].data


if __name__ == '__main__':
    root = Tk()
    #cframe = CFrame(master=root, path="")


    cdisplay = CDisplayTest(root, path="")
    workdir = "20140316_H_data"
    cdisplay.run_test(workdir)

    if 1:
        d = get_difference(workdir, "SDCH_20140316-1_FLAT_G1_ON.fits")
        assert np.alltrue(d == 0.)


from Tkinter        import *
import ttk

from PL_Frame      import *

from PL_Display import CDisplay

import os.path

class CDisplayTest(CDisplay):

    def StatusInsert(self, *kl, **kwargs):
        pass

    def init_test(self, workdir):
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


        self.pngpath = self.reducepath+'pngs/'
        if not os.path.exists(self.pngpath): os.mkdir(self.pngpath)

        self.makepng = True

        #self.reduce_cal_dark()
        print self.item_list

    def run_test_flat(self):
        item_list = self.item_list["FLAT"]
        ntotal = len(item_list)

        cal_flat = dict(method="med", # or "aver"
                        sigma="2.0",
                        clip=False)


        self.reduce_cal_flat_real(item_list, ntotal, "ON", "OFF",
                                  cal_flat=cal_flat,
                                  do_bp=False)


    def run_test_arc(self):
        item_list = self.item_list["ARC"]
        ntotal = len(item_list)

        cal_arc_cm = dict(method="med", # or "aver"
                          sigma="2.0",
                          clip=False)

        self.reduce_cal_arc_real(item_list, ntotal,
                                 frametype1="ON",
                                 frametype2="OFF",
                                 cal_arc_bp=False,
                                 cal_arc_cm=cal_arc_cm,
                                 cal_arc_ft=False,
                                 )

    def run_test_std(self):
        item_list = self.item_list["STD"]
        ntotal = len(item_list)

        combine_kwargs = dict(method="med", # or "aver"
                              sigma="2.0",
                              clip=False)

        self.reduce_obj_std_real(item_list, ntotal,
                                 frametype1="A",
                                 frametype2="B",
                                 do_cosmicray=False,
                                 do_badpixel=False,
                                 combine_kwargs=combine_kwargs,
                                 )

    def run_test_tar(self):
        item_list = self.item_list["TAR"]
        ntotal = len(item_list)

        combine_kwargs = dict(method="med", # or "aver"
                              sigma="2.0",
                              clip=False)

        self.reduce_obj_tar_real(item_list, ntotal,
                                 frametype1="A",
                                 frametype2="B",
                                 do_cosmicray=False,
                                 do_badpixel=False,
                                 combine_kwargs=combine_kwargs,
                                 )


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
    cdisplay.init_test(workdir)

    cdisplay.run_test_std()
    cdisplay.run_test_tar()

    if 0:
        d = get_difference(workdir, "SDCH_20140316-1_FLAT_G1_ON.fits")
        assert np.alltrue(d == 0.)

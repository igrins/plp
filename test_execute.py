
from Tkinter        import *
import ttk

from PL_Frame      import *

from PL_Display import CDisplay

class CDisplayTest(CDisplay):

    def StatusInsert(self, *kl, **kwargs):
        pass

    def run_test(self):
        self.reducepath = "/home/jjlee/work/igrins/pipeline_v1.05/20140316_H_data/"
        filenamefullpath = "/home/jjlee/work/igrins/pipeline_v1.05/20140316_H_data/IGRINS_DT_Log_20140316-1_H.txt"

        filename = self.get_filename(filenamefullpath)
        self.set_loadlogpath(filename, filenamefullpath)

        self.fileprefix = "SDC"+self.band+"_"+cdisplay.obsid+"_" #[To be removed]

        self.datpath = self.reducepath + 'dat/'

        #self.reduce_cal_dark()
        print self.item_list

        item_list = self.item_list["FLAT"]
        ntotal = len(item_list)

        cal_flat = dict(method="med", # or "aver"
                        sigma="2",
                        clip=False)


        self.reduce_cal_flat_real(item_list, ntotal, "ON", "OFF",
                                  cal_flat=cal_flat)



if __name__ == '__main__':
    root = Tk()
    #cframe = CFrame(master=root, path="")


    cdisplay = CDisplayTest(root, path="")
    cdisplay.run_test()

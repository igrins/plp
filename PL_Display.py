
# -*- coding: utf-8 -*-
'''
Created on July 24, 2012

Update by Huynh Anh on Feb 04, 2014
update by CKSim 2013-nov-05
Update by Huynh Anh on Oct 14, 2013
update by cksim 2013-Feb-05
'''

#import matplotlib
#matplotlib.use('Agg') # This comment prevents to show the final plots.
import matplotlib.pyplot as plt # Added by Huynh Anh 2014.02.04
from pylab import *  # Added by Huynh Anh 2014.02.04

from Tkinter        import *
import tkFileDialog
import ttk
import tkMessageBox

import sys
sys.path.append("..")
sys.path.append("../..")

from Libs.basic     import *
from Libs.extract     import *
from Libs.Input     import *
from Libs.deskew_loop_order import *
from Libs.manual import *
from Libs.deskew_wave import *
import Libs.ohlines_h as hband
import Libs.ohlines_kv2 as kband

from Libs.H_series import *

from sys import platform as _platform
import time, os

from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec



AP_DEGREE = 3
WL_DEGREE = [3,2]

custompath = './custommap/'


class CDisplay():
    '''
    classdocs
    '''

    def __init__(self, master, path):
        '''
        Constructor
        '''
        self.master = master
        self.path = path

    def string2list(self,input_string):
        input_string = str(input_string)
        if _platform == "linux" or _platform == "linux2" or _platform == "darwin": #'darwin' is iOS
            input_string = input_string.lstrip("('")
            input_string = input_string.rstrip("')")
            input_string = input_string.rstrip("',")
            output = input_string.split("', '")
        elif _platform == "win32": #'win32' is Windows
            output = input_string.split(" ")

        return output


    def loadmappath(self):
        initialdir="./standardmap"

        dir = tkFileDialog.askdirectory(initialdir=initialdir,title="Select a folder that has order extraction map...")
        if dir:
            self.mappathentry.set(dir)
            self.mappath = self.reducepathentry.get() + '/'


    def loadworkpath(self):
        if os.sys.platform.startswith('win'):
            initialdir="D:/_IGRINS/reduce"
        else:
            initialdir="/Volumes/data/IGRINS/reduce"
            if os.path.isdir(initialdir): pass
            #else: initialdir='/Users/chantes/_IGRINS/reduce'

        dir = tkFileDialog.askdirectory(initialdir=initialdir,title="Select a folder to save the reduced files...")
        if dir:
            self.reducepathentry.set(dir)
            self.reducepath = self.reducepathentry.get() + '/'


    def loadlog(self):
        initialdir=StringVar()
        if hasattr(self,'reducepath'): initialdir=self.reducepath
        filenamefullpath = tkFileDialog.askopenfilename(filetypes = [('Text files','*.txt'),('All files','*')], \
                                                        title="Select the observation log file...",initialdir=initialdir)
        if filenamefullpath:
            print filenamefullpath
            if hasattr(self,'reducepath') == False:
                self.reducepathentry.set(filenamefullpath.split('IGRINS_DT_Log')[0])
                self.reducepath = self.reducepathentry.get()
            filename = filenamefullpath.split('/')[-1]
            self.obslog.set(filename)
            self.obsid = filename.split('_')[3]
            self.band = filename.split('.')[0].split('_')[4]

            file = open(filenamefullpath,'r')
            loglines = file.read()
            loglines = loglines.splitlines()
            headerline = loglines[1].split(',')
            ifilename = headerline.index('FILENAME')
            iobjtype = headerline.index('OBJTYPE')
            iframetype = headerline.index('FRAMETYPE')
            igroupID1 = headerline.index('GROUP1')
            igroupID2 = headerline.index('GROUP2')

            #self.cal_dark_l.delete(0,END)
            self.cal_flat_l.delete(0,END)
            self.cal_arc_l.delete(0,END)
            self.obj_std_l.delete(0,END)
            self.obj_tar_l.delete(0,END)

            self.loglines = []
            self.logfilenames = []
            self.logframetypes = []
            self.loggroupID1s = []
            self.loggroupID2s = []
            for logline in loglines[2:]:
                if logline[0] != '#':
                    line = logline.split(',')

                    item = line[ifilename][:-5]+' ['+line[iframetype]+']['+line[igroupID1]+']['+line[igroupID2]+']'
                    #if line[iobjtype] == 'DARK':
                    #    self.cal_dark_l.insert(END,item)
                    if line[iobjtype] == 'FLAT':
                        self.cal_flat_l.insert(END,item)
                    elif line[iobjtype] == 'ARC':
                        self.cal_arc_l.insert(END,item)
                    elif line[iobjtype] == 'SKY':
                        self.cal_arc_l.insert(END,item)
                    elif line[iobjtype] == 'STD':
                        self.obj_std_l.insert(END,item)
                    elif line[iobjtype] == 'TAR':
                        self.obj_tar_l.insert(END,item)

                    self.loglines.append(line)
                    self.logfilenames.append(line[ifilename])
                    self.logframetypes.append(line[iframetype])
                    self.loggroupID1s.append(line[igroupID1])
                    self.loggroupID2s.append(line[igroupID2])

            '''
            for line in self.loglines[2:]:
                item = line[self.ifilename][:-5]+' ['+line[self.iobjtype]+']['+line[self.iframetype]+']'
                if line[self.iobjtype] == 'DARK':
                    darklist.append(item)
                    self.cal_dark_l.insert(END,item)
                elif line[self.iobjtype] == 'FLAT':
                    flatlist.append(item)
                    self.cal_flat_l.insert(END,item)
                elif line[self.iobjtype] == 'ARC':
                    arclist.append(item)
                    self.cal_arc_l.insert(END,item)
                elif line[self.iobjtype] == 'SKY':
                    skylist.append(item)
                    self.cal_arc_l.insert(END,item)
                elif line[self.iobjtype] == 'STD':
                    stdlist.append(item)
                    self.obj_std_l.insert(END,item)
                elif line[self.iobjtype] == 'TAR':
                    tarlist.append(item)
                    self.obj_tar_l.insert(END,item)
            '''

    def addfile(self,list):
        files = tkFileDialog.askopenfilenames(initialdir=self.reducepath,title="Add files to reduce...", \
                                              filetypes = [('FITs files','*.fits'),('All files','*')] )
        if files:
            files = self.string2list(files)
            files.sort()
            for item in files:
                filename = item.split('/')[-1]
                try:
                    i = self.logfilenames.index(filename)
                    line = self.loglines[:][i]
                    newitem = self.logfilenames[i][:-5]+ \
                              ' ['+self.logframetypes[i]+']['+self.loggroupID1s[i]+']['+self.loggroupID2s[i]+']'
                    list.insert(END,newitem)
                    newlist = sorted(list.get(0,END))
                    newlist = sorted(newlist)
                    list.delete(0,END)
                    for x in newlist: list.insert(END,x)
                except:
                    pass

            return list

    def removefile(self,list):
        items = sorted(list.curselection(),reverse=True)
        for i in items:
            list.delete(i)
        return list

    def listbox2filelist(self,listbox,frametype):
        items = listbox.get(0, END)
        ntotal = len(items)
        curselections = listbox.curselection()
        nsel = len(curselections)
        if nsel >= 1:
            curselection_i = [map(int, x) for x in curselections]
            selected = []
            for i in curselection_i:
                selected.append(items[i[0]])
            items = selected
        filenamelist=[]
        groupID1list=[]
        groupID2list=[]
        for item in items:
            if item.split('[')[1].rstrip(']') == frametype :
                filenamelist.append(self.reducepath+item.split(' [')[0]+'.fits')
                groupID1list.append(item.split('[')[2].rstrip(']'))
                groupID2list.append(item.split('[')[3].rstrip(']'))
        n=len(filenamelist)
        groupIDs=list(set(groupID1list+groupID2list))
        groupIDs.sort()

        return filenamelist, groupID1list, groupID2list, groupIDs, n, ntotal

    def Location(self,master):
        ttk.Label(master,text="Working directory").grid(row=1,column=1,sticky=W,pady=1)
        self.reducepathentry = StringVar()
        ttk.Entry(master,width=40,textvariable=self.reducepathentry).grid(row=1,column=2)
        ttk.Button(master, text="Browse...", command=self.loadworkpath).grid(row=1, column=3, sticky=W)

        ttk.Label(master,text="Observation log").grid(row=2,column=1,sticky=W,pady=1)
        self.obslog = StringVar()
        ttk.Entry(master,width=40,textvariable=self.obslog).grid(row=2,column=2)
        ttk.Button(master, text="Browse...", command=self.loadlog).grid(row=2, column=3, sticky=W)

        ttk.Label(master,text="Aperture extraction map").grid(row=3,column=1,sticky=W,pady=1)
        self.oextmappathentry = StringVar()
        ttk.Entry(master,width=40,textvariable=self.oextmappathentry).grid(row=3,column=2)
        self.oextmappathentry.set('STANDARD MAP (Default)')
        ttk.Button(master, text="Browse...", command=self.loadmappath).grid(row=3, column=3, sticky=W)

    '''
    def CalibrationDARK(self,master):
        ttk.Label(master,text="DARK").grid(row=1,column=1, sticky=W)
        self.cal_dark_l = Listbox(master, height=12, width=25, exportselection=0, selectmode=EXTENDED)
        self.cal_dark_l.grid(column=1, row=2,columnspan=2,sticky=(N,W,E,S))
        cal_dark_s = ttk.Scrollbar(master, orient=VERTICAL, command=self.cal_dark_l.yview)
        cal_dark_s.grid(column=3, row=2, sticky=(N,S))
        self.cal_dark_l['yscrollcommand'] = cal_dark_s.set
        ttk.Button(master, text="ADD", command=lambda:self.addfile(self.cal_dark_l)).grid(row=3, column=1, sticky=W)
        ttk.Button(master, text="REMOVE", command=lambda:self.removefile(self.cal_dark_l)).grid(row=3, column=2, sticky=W)

    def CalibrationInfo11(self,master):
        self.cal_dark_bp = IntVar()
        ttk.Checkbutton(master, onvalue=1, offvalue=0, text='Bad pixel correction', \
                        var=self.cal_dark_bp).grid(row=2, column=1,sticky=W)
        ttk.Label(master,text="  ").grid(row=3,column=1, sticky=W)

    def CalibrationCombineMethod1(self,master):
        self.cal_dark_cm = StringVar()
        self.cal_dark_cm.set('med') #default value
        self.cal_dark_cm_med = ttk.Radiobutton(master, text='Median', state='readonly', var=self.cal_dark_cm, value='med')
        self.cal_dark_cm_aver = ttk.Radiobutton(master, text='Average', state='readonly', var=self.cal_dark_cm, value='aver')
        self.cal_dark_cm_med.grid(row=1,column=1)
        self.cal_dark_cm_aver.grid(row=2,column=1)
        self.cal_dark_cm_clip = IntVar() #default
        ttk.Checkbutton(master, onvalue=1, offvalue=0, text='sigma clipping', var=self.cal_dark_cm_clip).grid(row=3, column=1)
        self.cal_dark_cm_sigma = DoubleVar(value=2.0)
        ttk.Entry(master,width=5,textvariable=self.cal_dark_cm_sigma).grid(row=3,column=2)
    '''
    def CalibrationFLAT(self,master):
        ttk.Label(master,text="FLAT").grid(row=1,column=1, sticky=W)
        self.cal_flat_l = Listbox(master, height=12, width=27, exportselection=0, selectmode=EXTENDED)
        self.cal_flat_l.grid(column=1, row=2,columnspan=2,sticky=(N,W,E,S))
        cal_flat_s = ttk.Scrollbar(master, orient=VERTICAL, command=self.cal_flat_l.yview)
        cal_flat_s.grid(column=3, row=2, sticky=(N,S))
        self.cal_flat_l['yscrollcommand'] = cal_flat_s.set
        ttk.Button(master, text="ADD", command=lambda:self.addfile(self.cal_flat_l)).grid(row=3, column=1, sticky=W)
        ttk.Button(master, text="REMOVE", command=lambda:self.removefile(self.cal_flat_l)).grid(row=3, column=2, sticky=W)

    def CalibrationInfo21(self,master):
        self.cal_flat_bp = IntVar()
        ttk.Checkbutton(master, onvalue=1, offvalue=0, text='Bad pixel correction', \
                        var=self.cal_flat_bp).grid(row=2, column=1,sticky=W)
        ttk.Label(master,text="  ").grid(row=3,column=1, sticky=W)

    def CalibrationCombineMethod2(self,master):
        self.cal_flat_cm = StringVar()
        self.cal_flat_cm.set('med') #default value
        self.cal_flat_cm_med = ttk.Radiobutton(master, text='Median', state='readonly', var=self.cal_flat_cm, value='med')
        self.cal_flat_cm_aver = ttk.Radiobutton(master, text='Average', state='readonly', var=self.cal_flat_cm, value='aver')
        self.cal_flat_cm_med.grid(row=1,column=1)
        self.cal_flat_cm_aver.grid(row=2,column=1)
        self.cal_flat_cm_clip = IntVar()
        ttk.Checkbutton(master, onvalue=1, offvalue=0, text='sigma clipping', \
                        var=self.cal_flat_cm_clip).grid(row=3, column=1)
        self.cal_flat_cm_sigma = DoubleVar(value=2.0)
        ttk.Entry(master,width=5,textvariable=self.cal_flat_cm_sigma).grid(row=3,column=2)

    def CalibrationInfo22(self,master):
        self.cal_flat_ft = IntVar()
        ttk.Checkbutton(master, onvalue=1, offvalue=0, text='Review/Modify the\norder extraction map', \
                        var=self.cal_flat_ft).grid(row=1, column=1)
        self.cal_flat_ft.set(0)

    def CalibrationARC(self,master):
        ttk.Label(master,text="ARC").grid(row=1,column=1, sticky=W)
        self.cal_arc_l = Listbox(master, height=12, width=27, exportselection=0, selectmode=EXTENDED)
        self.cal_arc_l.grid(column=1, row=2,columnspan=2,sticky=(N,W,E,S))
        cal_arc_s = ttk.Scrollbar(master, orient=VERTICAL, command=self.cal_arc_l.yview)
        cal_arc_s.grid(column=3, row=2, sticky=(N,S))
        self.cal_arc_l['yscrollcommand'] = cal_arc_s.set
        ttk.Button(master, text="ADD", command=lambda:self.addfile(self.cal_arc_l)).grid(row=3, column=1, sticky=W)
        ttk.Button(master, text="REMOVE", command=lambda:self.removefile(self.cal_arc_l)).grid(row=3, column=2, sticky=W)

    def CalibrationInfo31(self,master):
        self.cal_arc_bp = IntVar()
        ttk.Checkbutton(master, onvalue=1, offvalue=0, text='Bad pixel correction', \
                        var=self.cal_arc_bp).grid(row=2, column=1,sticky=W)
        ttk.Label(master,text="  ").grid(row=3,column=1, sticky=W)

    def CalibrationCombineMethod3(self,master):
        self.cal_arc_cm = StringVar()
        self.cal_arc_cm.set('med') #default value
        self.cal_arc_cm_med = ttk.Radiobutton(master, text='Median', state='readonly', var=self.cal_arc_cm, value='med')
        self.cal_arc_cm_aver = ttk.Radiobutton(master, text='Average', state='readonly', var=self.cal_arc_cm, value='aver')
        self.cal_arc_cm_med.grid(row=1,column=1)
        self.cal_arc_cm_aver.grid(row=2,column=1)
        self.cal_arc_cm_clip = IntVar()
        ttk.Checkbutton(master, onvalue=1, offvalue=0, text='sigma clipping', \
                        var=self.cal_arc_cm_clip).grid(row=3, column=1)
        self.cal_arc_cm_sigma = DoubleVar(value=2.0)
        ttk.Entry(master,width=5,textvariable=self.cal_arc_cm_sigma).grid(row=3,column=2)

    def CalibrationInfo32(self,master):
        self.cal_arc_ft = IntVar()
        ttk.Checkbutton(master, onvalue=1, offvalue=0, text='Review/Modify the\ndistortion correction map', \
                        var=self.cal_arc_ft).grid(row=1, column=1)
        ttk.Label(master,text="  ").grid(row=3,column=1, sticky=W)
        self.cal_arc_ft.set(0)

    def CalibrationExecute(self,master):
        #ttk.Label(master,text="  ").grid(row=1,column=1, sticky=W)
        self.calexebtn=ttk.Button(master, text="Execute", command=self.cal_execute)
        self.calexebtn.grid(row=1, column=1, sticky=W)
        self.calabortbtn=ttk.Button(master, text="Abort")
        self.calabortbtn.config(state=DISABLED)
        self.calabortbtn.grid(row=1, column=2, sticky=W)

    def ObjectStandard(self,master):
        ttk.Label(master,text="STD").grid(row=1,column=1, sticky=W)
        self.obj_std_l = Listbox(master, height=12, width=27, exportselection=0, selectmode=EXTENDED)
        self.obj_std_l.grid(column=1, row=2,columnspan=2,sticky=(N,W,E,S))
        obj_std_s = ttk.Scrollbar(master, orient=VERTICAL, command=self.obj_std_l.yview)
        obj_std_s.grid(column=3, row=2, sticky=(N,S))
        self.obj_std_l['yscrollcommand'] = obj_std_s.set
        ttk.Button(master, text="ADD", command=lambda:self.addfile(self.obj_std_l)).grid(row=3, column=1, sticky=W)
        ttk.Button(master, text="REMOVE", command=lambda:self.removefile(self.obj_std_l)).grid(row=3, column=2, sticky=W)

    def ObjectInfo11(self,master):
        self.obj_std_bp = IntVar()
        self.obj_std_cr = IntVar()
        ttk.Checkbutton(master, onvalue=1, offvalue=0, text='Bad pixel correction', \
                        var=self.obj_std_bp).grid(row=2, column=1,sticky=W)
        ttk.Checkbutton(master, onvalue=1, offvalue=0, text='Cosmic-ray correction', \
                        var=self.obj_std_cr).grid(row=3, column=1,sticky=W)

    def ObjectCombineMethod1(self,master):
        self.obj_std_cm = StringVar()
        self.obj_std_cm.set('aver')
        self.obj_std_cm_med = ttk.Radiobutton(master, text='Median', state='readonly', var=self.obj_std_cm, value='med')
        self.obj_std_cm_aver = ttk.Radiobutton(master, text='Average', state='readonly', var=self.obj_std_cm, value='aver')
        self.obj_std_cm_med.grid(row=1,column=1)
        self.obj_std_cm_aver.grid(row=2,column=1)
        self.obj_std_cm_clip = IntVar()
        ttk.Checkbutton(master, onvalue=1, offvalue=0, text='Sigma clipping', var=self.obj_std_cm_clip).grid(row=3, column=1)
        self.obj_std_cm_sigma = DoubleVar(value=2.0)
        ttk.Entry(master,width=5,textvariable=self.obj_std_cm_sigma).grid(row=3,column=2)


    def ObjectInfo12(self,master):
        self.obj_std_ft = IntVar()
        ttk.Checkbutton(master, onvalue=1, offvalue=0, text='Use custom fine-tuning', \
                        var=self.obj_std_ft).grid(row=4, column=1,sticky=W)
        ttk.Entry(master,width=15).grid(row=5,column=1)

    def ObjectTarget(self,master):
        ttk.Label(master,text="TAR").grid(row=1,column=1, sticky=W)
        self.obj_tar_l = Listbox(master, height=12, width=27, exportselection=0, selectmode=EXTENDED)
        self.obj_tar_l.grid(column=1, row=2,columnspan=2,sticky=(N,W,E,S))
        obj_tar_s = ttk.Scrollbar(master, orient=VERTICAL, command=self.obj_tar_l.yview)
        obj_tar_s.grid(column=3, row=2, sticky=(N,S))
        self.obj_tar_l['yscrollcommand'] = obj_tar_s.set
        ttk.Button(master, text="ADD", command=lambda:self.addfile(self.obj_tar_l)).grid(row=3, column=1, sticky=W)
        ttk.Button(master, text="REMOVE", command=lambda:self.removefile(self.obj_tar_l)).grid(row=3, column=2, sticky=W)

    def ObjectInfo21(self,master):
        self.obj_tar_bp = IntVar()
        self.obj_tar_cr = IntVar()
        ttk.Checkbutton(master, onvalue=1, offvalue=0, text='Bad pixel correction', \
                        var=self.obj_tar_bp).grid(row=2, column=1,sticky=W)
        ttk.Checkbutton(master, onvalue=1, offvalue=0, text='Cosmic-ray correction', \
                        var=self.obj_tar_cr).grid(row=3, column=1,sticky=W)

    def ObjectCombineMethod2(self,master):
        self.obj_tar_cm = StringVar()
        self.obj_tar_cm.set('aver')
        self.obj_tar_cm_med = ttk.Radiobutton(master, text='Median', state='readonly', var=self.obj_tar_cm, value='med')
        self.obj_tar_cm_aver = ttk.Radiobutton(master, text='Average', state='readonly', var=self.obj_tar_cm, value='aver')
        self.obj_tar_cm_med.grid(row=1,column=1)
        self.obj_tar_cm_aver.grid(row=2,column=1)
        self.obj_tar_cm_clip = IntVar()
        ttk.Checkbutton(master, onvalue=1, offvalue=0, text='Sigma clipping', var=self.obj_tar_cm_clip).\
                                grid(row=3, column=1)
        self.obj_tar_cm_sigma = DoubleVar(value=2.0)
        ttk.Entry(master,width=5,textvariable=self.obj_tar_cm_sigma).grid(row=3,column=2)

    def ObjectInfo22(self,master):
        self.obj_tar_ft = IntVar()
        ttk.Checkbutton(master, onvalue=1, offvalue=0, text='Use custom fine-tuning', var=self.obj_tar_ft).\
                                grid(row=4, column=1,sticky=W)
        ttk.Entry(master,width=15).grid(row=5,column=1)

    def ObjectExecute(self,master):
        ttk.Label(master,text="  \n  \n  ").grid(row=1,column=1, sticky=W)
        #ttk.Label(master,text="  ").grid(row=2,column=1, sticky=W)
        self.objexebtn=ttk.Button(master, text="Execute", command=self.obj_execute)
        self.objexebtn.grid(row=2, column=1, sticky=W)
        self.objabortbtn=ttk.Button(master, text="Abort")
        self.objabortbtn.config(state=DISABLED)
        self.objabortbtn.grid(row=2, column=2, sticky=W)

    '''
    def Status(self,master):
        self.status = Text(master, width=50,height=30)
        #self.commoutput.insert(ttk.Tkinter.INSERT, self.connected + ">")
        self.status.grid(column=0, row=1)
        self.statusscroll = Scrollbar(master)
        self.status.pack(side=LEFT, fill=BOTH)
        self.statusscroll.pack(side=RIGHT, fill=Y)
        self.statusscroll.config(command=self.status.yview)
        self.status.config(yscrollcommand=self.statusscroll.set)
    '''

    # input the status text
    def StatusInsert(self, str_text):
        '''
        Input the status in Status Text Widget
        - call the status.insert() function
        - reset the text using status.see() function
        '''

        print str_text
        self.WriteLog.write(str_text+"\n")

        '''
        try:
            data = dataQueue.get(block=False)
        except Queue.Empty:
            pass
        else:
            t.delete('0', END)
            t.insert('0', '%s\n' % str(data))
        t.after(250, lambda: status(t))
        '''


    '''
    master.pack_configure(fill=Tkinter.BOTH, expand=1)
    t = ScrolledText.ScrolledText(master, width=60, height=37)
    t.insert(Tkinter.END, self.log.getText())
    t.configure(state=Tkinter.DISABLED)
    t.see(Tkinter.END)
    t.pack(fill=Tkinter.BOTH)
    '''


    def cal_execute(self):

        pllogfilename = self.obslog.get().replace('_DT_','_PL_')
        self.WriteLog = open(self.reducepath+pllogfilename,'a')
        self.StatusInsert("# Reduction for calibration frames started at "+time.strftime('%Y-%m-%d %H:%M:%S'))

        self.makepng = True
        self.pngpath = self.reducepath+'pngs/'
        if not os.path.exists(self.pngpath): os.mkdir(self.pngpath)

        self.calexebtn.config(state=DISABLED)
        self.calabortbtn.config(state=ACTIVE)
        self.objexebtn.config(state=DISABLED)
        #self.objabortbtn.config(state=DISABLED)

        #self.band = "K"#[To be removed]
        #self.obsid = '20130725'#[To be removed]
        self.fileprefix = "SDC"+self.band+"_"+self.obsid+"_" #[To be removed]

        self.datpath = self.reducepath + 'dat/'

        #self.reduce_cal_dark()
        self.reduce_cal_flat()
        self.reduce_cal_arc()

        self.StatusInsert("-- Done (Calibration Frames) "+time.strftime('%Y-%m-%d %H:%M:%S')+' --\n')
        self.calexebtn.config(state=ACTIVE)
        self.calabortbtn.config(state=DISABLED)
        self.objexebtn.config(state=ACTIVE)

        self.WriteLog.close()

    def obj_execute(self):

        pllogfilename = self.obslog.get().replace('_DT_','_PL_')
        self.WriteLog = open(self.reducepath+pllogfilename,'a')
        self.StatusInsert("Reduction for object frames started at "+time.strftime('%Y-%m-%d %H:%M:%S'))

        self.makepng = True
        self.pngpath = self.reducepath+'pngs/'
        if not os.path.exists(self.pngpath): os.mkdir(self.pngpath)

        self.calexebtn.config(state=DISABLED)
        self.objexebtn.config(state=DISABLED)
        self.objabortbtn.config(state=ACTIVE)

        #self.band = "K"#[To be removed]
        #self.obsid = '20130725'#[To be removed]
        self.fileprefix = "SDC"+self.band+"_"+self.obsid+"_" #[To be removed]

        self.reduce_obj_std()
        self.reduce_obj_tar()

        self.StatusInsert("-- Done (Object Frames) "+time.strftime('%Y-%m-%d %H:%M:%S')+' --\n')

        #from selected files to a number of variables(2D array)
        #path = self.path.get() + '/'
        #print self.obj_std_l.get(0,END)
        self.calexebtn.config(state=ACTIVE)
        #self.calabortbtn.config(state=DISABLED)
        self.objexebtn.config(state=ACTIVE)
        self.objabortbtn.config(state=DISABLED)

        self.WriteLog.close()

    '''
    def reduce_cal_dark(self):
        filelist, groupID1list, groupID2list, groupIDs, n, ntotal = self.listbox2filelist(self.cal_dark_l,'')
        if n >= 1:
            objtype = 'DARK'
            self.StatusInsert("["+objtype+"] "+'-'*(50-len(objtype)-3))
            self.StatusInsert(str(n)+" out of "+str(ntotal)+" files selected.")
            for groupID in groupIDs:
                curlist=[]
                for i in range(0,n):
                    if groupID1list[i]==groupID or groupID2list[i]==groupID:
                        curlist.append(filelist[i])

                self.StausInsert('<Group Number '+str(groupID)+' ('+str(n)+' files)> ')

                imgs, hdrs = self.readechellogram(curlist)
                #correct bad pixel
                if self.cal_dark_bp.get() == 1:
                    self.StatusInsert("Bad pixel correction...")
                    imgs, hdrs = self.badpixcor(fits,headers=hdrs)
                #else:
                #    imgs, hdrs = fits[:], hdrs[:]

                dark, hdr = self.combine(objtype,'', imgs, groupID \
                                       , self.cal_dark_cm.get(), sclip=self.cal_dark_cm_clip\
                                       , sigma=self.cal_dark_cm_sigma, headers=hdrs)


                #del(fits, hdrs, dark,hdr)
     '''

    def reduce_cal_flat(self):
        onfilelist, ongroupID1list, ongroupID2list, ongroupIDs, non, ntotal = self.listbox2filelist(self.cal_flat_l,'ON')
        offfilelist,offgroupID1list,offgroupID2list,offgroupIDs,noff,ntotal = self.listbox2filelist(self.cal_flat_l,'OFF')
        if non == noff and non >= 1 and ongroupIDs==offgroupIDs :
            objtype = 'FLAT'
            self.StatusInsert("["+objtype+"] "+'-'*(50-len(objtype)-3))
            self.StatusInsert(str(non+noff)+" out of "+str(ntotal)+" files selected.")
            for groupID in ongroupIDs:
                curonlist=[]
                curofflist=[]
                for i in range(0,non):
                    if ongroupID1list[i]==groupID or ongroupID2list[i]==groupID:
                        curonlist.append(onfilelist[i])
                    if offgroupID1list[i]==groupID or offgroupID2list[i]==groupID:
                        curofflist.append(offfilelist[i])
                if len(curonlist)==len(curofflist):
                    self.StatusInsert('<Group Number '+str(groupID)+' ('+str(non+noff)+' files)> ')

                    imgsA, hdrsA = self.readechellogram(curonlist)
                    imgsB, hdrsB = self.readechellogram(curofflist)

                    print 'list A', imgsA, len(imgsA)
                    print 'list B', imgsB, len(imgsB)

                    #correct bad pixel
                    if self.cal_flat_bp.get() == 1:
                        self.StatusInsert("Bad pixel correction...")
                        imgsA, hdrsA = self.badpixcor(imgsA,headers=hdrsA)
                        imgsB, hdrsB = self.badpixcor(imgsB,headers=hdrsB)

                    on, hdr_on = self.combine(objtype, 'ON', imgsA, groupID, self.cal_flat_cm.get() \
                                              , sclip=self.cal_flat_cm_clip.get(), sigma=self.cal_flat_cm_sigma.get() \
                                              , headers=hdrsA)

                    off, hdr_off = self.combine(objtype, 'OFF', imgsB, groupID, self.cal_flat_cm.get() \
                                              , sclip=self.cal_flat_cm_clip.get(), sigma=self.cal_flat_cm_sigma.get() \
                                              , headers=hdrsB)

                    flat, hdr = self.subtract(objtype, 'ON-OFF', on, off, groupID, headers=hdr_on)
                    flat = ip.cosmicrays(flat, threshold=2000, flux_ratio=0.2, wbox=3)

                    flat, hdr = self.normalizeto1(flat, header=hdr_on)

                    self.save1file(self.fileprefix+objtype+'_G'+str(groupID),flat,header=hdr)

                #--aperture extraction--
                stripfiles = self.deskew(objtype,flat, groupID, header=hdr,outname='.ONOFF')
                #check_deskew = deskew_plot(self.band, self.reducepath, name=self.fileprefix+objtype+ \
                #                                       '_G'+str(groupID)+'.ONOFF', start=0, end=23)
                if self.cal_flat_ft.get() == 1: #fine-tuning mode'
                    print 'Please wait for the residual display (Standard extraction map)'
                    check_deskew = deskew_plot(self.band, self.reducepath, name=self.fileprefix+objtype+ \
                                                       '_G'+str(groupID)+'.ONOFF', start=0, end=23)
                    finetune_flag = tkMessageBox.askyesno("Order extraction", "Make new order extraction map?")
                    #finetune_flag = True

                    if finetune_flag == True:

                        if self.band == "H":

                            ap_tracing_strip(self.band, filename=self.reducepath+self.fileprefix+objtype+'_G'+str(groupID)+'.fits', \
                                         npoints=40, spix=10, dpix=20, thres=400, \
                                         start_col= 1300, ap_thres=None, degree=None, devide=8.0, target_path=custompath)

                        if self.band == "K":

                            ap_tracing_strip(self.band, filename=self.reducepath+self.fileprefix+objtype+'_G'+str(groupID)+'.fits', \
                                         npoints=40, spix=10, dpix=20, thres=400, \
                                         start_col= 1024, ap_thres=None, degree=None, devide=20.0, target_path=custompath)

                        stripfiles = self.deskew(objtype,flat, groupID, header=hdr, deskew_path=custompath, \
                                                 outname='_custom.ONOFF') #deskew using update mapping data
                        print 'Please wait for the new display for the residuals (customized mapping)'
                        check_deskew = deskew_plot(self.band, self.reducepath, name=self.fileprefix+objtype+ \
                                                        '_G'+str(groupID)+'_custom.ONOFF', start=0, end=23)
                        #finetune_flag = tkMessageBox.askyesno("Order extraction", "Make new order extraction map?")
                #--end of aperture extraction--


    def reduce_cal_arc(self):
        onfilelist, ongroupID1list, ongroupID2list, ongroupIDs, non, ntotal = self.listbox2filelist(self.cal_arc_l,'ON')
        offfilelist,offgroupID1list,offgroupID2list,offgroupIDs,noff,ntotal = self.listbox2filelist(self.cal_arc_l,'OFF')
        if non == noff and non >= 1 and ongroupIDs==offgroupIDs :
            objtype = 'ARC'
            self.StatusInsert("["+objtype+"] "+'-'*(50-len(objtype)-3))
            self.StatusInsert(str(non+noff)+" out of "+str(ntotal)+" files selected.")
            for groupID in ongroupIDs:
                curonlist=[]
                curofflist=[]
                for i in range(0,non):
                    if ongroupID1list[i]==groupID or ongroupID2list[i]==groupID:
                        curonlist.append(onfilelist[i])
                    if offgroupID1list[i]==groupID or offgroupID2list[i]==groupID:
                        curofflist.append(offfilelist[i])
                if len(curonlist)==len(curofflist):
                    self.StatusInsert('<Group Number '+str(groupID)+' ('+str(non+noff)+' files)> ')

                    imgsA, hdrsA = self.readechellogram(curonlist)
                    imgsB, hdrsB = self.readechellogram(curofflist)
                    if self.cal_arc_bp.get() == 1:
                        self.StatusInsert("Bad pixel correction...")
                        imgsA, hdrsA = self.badpixcor(imgsA,headers=hdrsA)
                        imgsB, hdrsB = self.badpixcor(imgsB,headers=hdrsB)

                    on, hdr_on = self.combine(objtype, 'ON', imgsA, groupID, self.cal_arc_cm.get(), sclip=self.cal_arc_cm_clip.get(), sigma=self.cal_arc_cm_sigma.get(), headers=hdrsA)
                    off, hdr_off = self.combine(objtype, 'OFF', imgsB, groupID, self.cal_arc_cm.get(), sclip=self.cal_arc_cm_clip.get(), sigma=self.cal_arc_cm_sigma.get(), headers=hdrsB)

                    #arc, hdr = self.subtract(objtype, 'ON-OFF', on, off, groupID, headers=hdr_on)
                    arc, hdr = on, hdr_on # test with no subtract dark
                    arc = ip.cosmicrays(arc, threshold=1000, flux_ratio=0.2, wbox=3)   # Add by Huynh Anh 2014.04.08

                    self.save1file(self.fileprefix+objtype+'_G'+ \
                                   str(groupID),arc,header=hdr)  #self.combine(objtype, '', arc, self.cal_arc_cm.get(), headers=hdr_arc)

                #--aperture extraction--
                stripfiles = self.deskew(objtype,arc, groupID, header=hdr, deskew_path=custompath, outname='.ONOFF')

                # --wavelength calibration using OH line list - Added by Huynh Anh 2014.02.04
                if self.cal_arc_ft.get() == 1: # fine-tuning mode'
                    #print 'Wavelength calibration using referent OH line list'
                    #print '>>>>>>>>> Running >>>>>>>>>>'
                    self.StatusInsert('Wavelength Calibration using reference OH line list')
                    self.wavelength_oh(objtype, groupID)

                else:

                    #id_stripfiles = self.identify(objtype,stripfiles,makepng=self.makepng)
                    pass



    def reduce_obj_std(self):
        Afilelist, AgroupID1list, AgroupID2list, AgroupIDs, nA, ntotal = self.listbox2filelist(self.obj_std_l,'A')
        Bfilelist, BgroupID1list, BgroupID2list, BgroupIDs, nB, ntotal = self.listbox2filelist(self.obj_std_l,'B')
        if nA == nB and nA >= 1 and AgroupIDs==BgroupIDs:
            objtype = 'STD' #'STANDARD'
            self.StatusInsert("["+objtype+"] "+'-'*(50-len(objtype)-3))
            self.StatusInsert(str(nA+nB)+" out of "+str(ntotal)+" files selected.")
            for groupID in AgroupIDs:
                curAlist=[]
                curBlist=[]
                for i in range(0,nA):
                    if AgroupID1list[i]==groupID or AgroupID2list[i]==groupID:
                        curAlist.append(Afilelist[i])
                    if BgroupID1list[i]==groupID or BgroupID2list[i]==groupID:
                        curBlist.append(Bfilelist[i])
                if len(curAlist)==len(curBlist):
                    self.StatusInsert('<Group Number '+str(groupID)+' ('+str(nA+nB)+' files)> ')

                    imgsA, hdrsA = self.readechellogram(curAlist)
                    imgsB, hdrsB = self.readechellogram(curBlist)
                    #correct cosmic rays
                    if self.obj_std_cr.get()==1:
                        self.StatusInsert("Cosmic ray correction... It might take long computing time.")
                        imgsA, hdrsA = self.cosmicraycor(imgsA,headers=hdrsA)
                        imgsB, hdrsB = self.cosmicraycor(imgsB,headers=hdrsB)
                    else:
                        imgsA, hdrsA = imgsA[:], hdrsA[:]
                        imgsB, hdrsB = imgsB[:], hdrsB[:]

                    #correct bad pixel
                    if self.obj_std_bp.get() == 1:
                        self.StatusInsert("Bad pixel correction...")
                        imgsA, hdrsA = self.badpixcor(imgsA,headers=hdrsA)
                        imgsB, hdrsB = self.badpixcor(imgsB,headers=hdrsB)
                    else:
                        imgsA, hdrsA = imgsA[:], hdrsA[:]
                        imgsB, hdrsB = imgsB[:], hdrsB[:]

                    imgs, hdrs = self.subtract(objtype,'A-B',imgsA,imgsB, groupID, headers=hdrsA)

                    imgs, hdrs = self.nflat(objtype,imgs,groupID,headers=hdrs)

                    img, hdr = self.combine(objtype,'AB',imgs,groupID, self.obj_std_cm.get(), sclip=self.obj_std_cm_clip.get(), sigma=self.obj_std_cm_sigma.get(), headers=hdrs)

                    stripfiles = self.deskew(objtype, img, groupID, header=hdr, deskew_path=custompath, outname='.AB') #deskew

                    #stripfiles = self.identify(objtype,stripfiles)  # transform using zemax mapping

                    stripfiles = self.transform_aperture(objtype, groupID)

                    specfiles = self.trace_n_average_AB(stripfiles,pngpath=self.makepng)

                    #specfiles = ['SDCH_20130827-1_STANDARD_G%s.AB.tr.%03i.tpn' %(groupID,i) for i in range(1,24)]
                    specfiles = self.H_n_cont_removal(specfiles,threslow=1.5, threshigh=4., \
                                                     pngpath=self.pngpath)

                    #std_spectra.append(spectrum)
                    #del(id_stripfiles,spectra)

                    try: self.std[groupID]=specfiles
                    except: self.std={groupID:specfiles}

                    '''
                    #2013-12-24 CKSim below pickle-do's are for the development stage.
                    file=open(self.reducepath+'std_spectra.txt','wb')
                    pickle.dump(std_spectra, file)
                    file.close()
                    file=open(self.reducepath+'wave.txt','wb')
                    pickle.dump(waves, file)
                    file.close()
                    '''



    def reduce_obj_tar(self):
        Afilelist, AgroupID1list, AgroupID2list, AgroupIDs, nA, ntotal = self.listbox2filelist(self.obj_tar_l,'A')
        Bfilelist, BgroupID1list, BgroupID2list, BgroupIDs, nB, ntotal = self.listbox2filelist(self.obj_tar_l,'B')
        if nA == nB and nA >= 1 and AgroupIDs==BgroupIDs:
            objtype = 'TAR' #'TARGET'
            self.StatusInsert("["+objtype+"] "+'-'*(50-len(objtype)-3))
            self.StatusInsert(str(nA+nB)+" out of "+str(ntotal)+" files selected.")
            for groupID in AgroupIDs:
                curAlist=[]
                curBlist=[]
                for i in range(0,nA):
                    if AgroupID1list[i]==groupID or AgroupID2list[i]==groupID:
                        curAlist.append(Afilelist[i])
                    if BgroupID1list[i]==groupID or BgroupID2list[i]==groupID:
                        curBlist.append(Bfilelist[i])
                if len(curAlist)==len(curBlist):
                    self.StatusInsert('<Group Number '+str(groupID)+' ('+str(nA+nB)+' files)> ')

                    imgsA, hdrsA = self.readechellogram(curAlist)
                    imgsB, hdrsB = self.readechellogram(curBlist)
                    #correct cosmic rays
                    if self.obj_std_cr.get()==1:
                        self.StatusInsert("Cosmic ray correction...")
                        imgsA, hdrsA = self.cosmicraycor(imgsA,headers=hdrsA)
                        imgsB, hdrsB = self.cosmicraycor(imgsB,headers=hdrsB)
                    else:
                        imgsA, hdrsA = imgsA[:], hdrsA[:]
                        imgsB, hdrsB = imgsB[:], hdrsB[:]

                    #correct bad pixel
                    if self.obj_std_bp.get() == 1:
                        self.StatusInsert("Bad pixel correction...")
                        imgsA, hdrsA = self.badpixcor(imgsA,headers=hdrsA)
                        imgsB, hdrsB = self.badpixcor(imgsB,headers=hdrsB)
                    else:
                        imgsA, hdrsA = imgsA[:], hdrsA[:]
                        imgsB, hdrsB = imgsB[:], hdrsB[:]

                    imgs, hdrs = self.subtract(objtype,'A-B',imgsA,imgsB, groupID, headers=hdrsA)

                    imgs, hdrs = self.nflat(objtype,imgs,groupID,headers=hdrs)

                    img, hdr = self.combine(objtype,'AB',imgs,groupID, self.obj_std_cm.get(), sclip=self.obj_tar_cm_clip.get(), sigma=self.obj_tar_cm_sigma.get(), headers=hdrs)

                    stripfiles = self.deskew(objtype,img, groupID, header=hdr, deskew_path=custompath, outname='.AB') #deskew

                    #stripfiles = self.identify(objtype,stripfiles)  # transform using zemax mapping

                    stripfiles = self.transform_aperture(objtype, groupID)

                    specfiles = self.trace_n_average_AB(stripfiles,pngpath=self.makepng)

                    specfiles = self.telcor(specfiles,pngpath=self.makepng)

    '''
    def readechellogram(self, list):

        imgs = []
        hdrs = []
        for item in list:
            img, hdr = readfits(item)

            #2014-03-10 cksim
            #H-band: rotate 90 CW band flip horizontally
            #K-band: 90
            if self.band == 'H':
                img = np.rot90(img)
                img = np.fliplr(img)
            if self.band == 'K':
                img = np.rot90(img,3)

            img_min = array(img).min()

            if img_min <= 0:
                img -= img_min

            imgs.append(img)
            hdrs.append(hdr)

        return imgs,hdrs
    '''

    def readechellogram(self, list):

        imgs = []
        hdrs = []
        for item in list:
            img, hdr = readfits(item)
            #img_min = array(img).min()
            #print 'item', item
            #if img_min <= 0:
            #    img -= img_min

            imgs.append(img)
            hdrs.append(hdr)
        print 'imgs', imgs
        return imgs,hdrs

    def cosmicraycor(self, images, headers=None):


        imgs = []
        hdrs = []
        n=len(images)
        for i in xrange(n):
            crcor = cosmicrays(images[i], threshold=1000, flux_ratio=0.2, wbox=3)
            imgs.append(crcor)

            if headers == None: hdr=newheader()
            else: hdr = headers[i].copy()
            hdr.update('CRAYCOR', time.strftime('%Y-%m-%d %H:%M:%S'))
            hdrs.append(hdr)

        del(images,headers)
        return imgs,hdrs


    def badpixcor(self, images, headers=None):

        if self.band == 'H':
            badpixelfile = "./badpixel_h.fits"
        else:
            badpixelfile = "./badpixel_k.fits"

        badpixelmask, bphdr = readfits(badpixelfile)

        imgs = []
        hdrs = []
        n=len(images)
        for i in xrange(n):
            img = fixpix(images[i], badpixelmask, 'cinterp') #'cinterp' or 'linterp'
            imgs.append(img)

            if headers == None: hdr = newheader()
            hdr=headers[i].copy()
            #hdr.update('BADPCOR', time.strftime('%Y-%m-%d %H:%M:%S'))
            #hdr.update('BADPCOR', badpixelfile)
            hdrs.append(hdr)

        del(images,headers)
        return imgs,hdrs

    def _old_read_n_badpixcor(self, objtype, list, badpixelflag):
        if badpixelflag == 1:
            self.StatusInsert("Bad pixel correction...")
            badpixelfile = "../../../System/FITS/PL/badpixel.fits"
            badpixelmask, hdr = readfits(badpixelfile)

        fits = []
        hdrs = []
        for item in list:
            data, hdr = readfits(item)
            if badpixelflag == 1:
                data = fixpix(data, badpixelmask, 'cinterp') #'cinterp' or 'linterp'
                hdr = hdr #[Update header here]
            fits.append(data)
            hdrs.append(hdr)

        return fits,hdrs

    def _old_cosmicraycor(self, objtype, imgs, headers=''):
         self.StatusInsert("Cosmic ray correction...")

         n=len(imgs)
         if n==2048:
             imgs = [imgs]
             headers = [headers]
             n=1

         data = []
         hdrs = []
         for i in range(0,n):
             img = imgs[i]
             crcor = cosmicrays(img, threshold=1000, flux_ratio=0.2, wbox=3)
             data.append(crcor)

             if headers[0] != '':
                 hdr = headers[i] #[Update header here]
             else:
                 hdr = headers
             hdrs.append(hdr)

         return data,hdrs

    def combine(self, objtype, frametype, imgs, groupID, method, sclip=0, sigma=0., headers=None):
        n = len(imgs)
        #print 'imgs', imgs
        print 'n combine', n
        if n>= 1 :
            if frametype != '':
                file = self.fileprefix+objtype+"_G"+groupID+"_"+frametype
            else:
                file = self.fileprefix+objtype+"_G"+groupID
            if n==2048:
                comb = imgs
                if headers == None: hdr = newheader()
                else: hdr = headers

            elif n==1:
                comb = imgs[0]
                if headers == None: hdr = newheader()
                else: hdr = headers[0]


            else:
                self.StatusInsert("Combining ("+method+")... ["+objtype+"]["+groupID+"]["+frametype+"] files")
                if headers == None:
                    hdr = newheader() #Do not weighing when headers are none
                else:
                    hdr = headers[0]
                if sclip == 1:
                    self.StatusInsert("Sigma clipping (sigma * %3.1f)" %sigma)
                    imgs = sigma_clip(imgs, axis=0, sig=sigma)#.data
                    imgs.data[imgs.mask] = np.nan
                    imgs=imgs.data
                    hdr.update('HISTORY','sigma clipping sigma * %3.1f' %sigma)

                #exptimes = [headers[item]['EXPTIME'] for item in range(n)]
                #weights = exptimes / sum(exptimes)
                #for img, weight in zip(imgs,weights): img *= weight

                if method == 'med':
                    comb = imcombine(imgs, mode='median')
                elif method == 'aver':
                    comb = imcombine(imgs, mode='average')
                hdr.update('HISTORY','combine %d files' %n)

                self.save1file(file,comb,header=hdr)

        del(objtype, frametype, imgs, groupID, method, headers, n)
        return comb, hdr

    def subtract(self, objtype, frametype, on_imgs, off_imgs, groupID, headers=None):
        n1, n2 = len(on_imgs), len(off_imgs)
        print 'n1, n2', n1, n2

        if n1 != n2:
           return
        elif n1==n2:
            on_file = self.fileprefix+objtype+'_G'+str(groupID)+'_'+frametype.split('-')[0]
            off_file = self.fileprefix+objtype+'_G'+str(groupID)+'_'+frametype.split('-')[1]
            self.StatusInsert('Subtracting... "' + on_file +'" - "' + off_file +'"')

            if n1 == 1:
                on_img = on_imgs[0]
                off_img = off_imgs[0]
                if headers == None: hdr = newheader()
                else: hdr = headers[0]

                ABfile = self.fileprefix+objtype+'_G'+groupID+'_'+frametype+'001'
                tmp = imarith(on_img, off_img, '-')
                hdr.update('IMARITH',on_file+' - '+off_file )
                self.save1file(ABfile,tmp,header=hdr)
            if n1 == 2048 :
                ABfile = self.fileprefix+objtype+'_G'+groupID+'_'+frametype+'001'
                tmp = imarith(on_imgs, off_imgs, '-')
                if headers == None: hdr = newheader()
                else: hdr = headers
                hdr.update('HISTORY',on_file+' - '+off_file )
                self.save1file(ABfile,tmp,header=hdr)
                return tmp, hdr

            AB=[]
            hdrs=[]
            for i in range(0,n1):
                tmp = imarith(on_imgs[i], off_imgs[i], '-')
                AB.append(tmp)
                ABfile = self.fileprefix+objtype+'_G'+groupID+'_'+frametype+'%03d' %(i+1)

                if headers == None: hdr = newheader()
                else: hdr = headers[i]
                hdr.update('HISTORY', on_file+' - '+off_file)
                hdrs.append(hdr)
                self.save1file(ABfile,tmp,header=hdr)

        return AB, hdrs

    def normalizeto1(self,img,header=None):
        #min= float(np.nanmin(img))
        #max= float(np.nanmax(img)) # np.nanmax(img.ravel())
        #scale = max - min
        #print 'scale', min, max, scale

        #img -= min
        h_scale = 3200 # Constant number for normalization
        k_scale = 1700

        if self.band == 'H':
            img = np.divide(img, h_scale)
        if self.band == 'K':
            img = np.divide(img, k_scale)

        if header == None: header = newheader()
        header.update('HISTORY', 'Normalized maximum value to unity')

        self.StatusInsert('Normalized maximum value to unity.')

        return img, header


    def nflat(self, objtype, imgs, groupID, headers=None):
        try:
            flat, fhdr = readfits(self.masterflatfile)
        except:
            try:
                self.masterflatfile = self.reducepath+self.fileprefix+"FLAT_G"+str(groupID)+".fits"
                flat, fhdr = readfits(self.masterflatfile)
            except:
                self.masterflatfile = tkFileDialog.askopenfilename(filetypes = [('FITs files','*.fits'),('All files','*')] ,title="Select the master FLAT file for the flat fielding...",initialdir=self.reducepath)
                flat, fhdr = readfits(self.masterflatfile)

        n = len(imgs)
        if n == 2048:
            imgs = [imgs]
            if headers == None: headers = [newheader()]
            else: headers = [headers]
            n=1

        imgs_nf = []
        hdrs = []
        for i in range(0,n):
            tmp = imarith(imgs[i], flat, '/')
            imgs_nf.append(tmp)

            if headers == None: hdr = newheader()
            else: hdr = headers[i]
            hdr.update('HISTORY', '/ NFLAT'+self.masterflatfile.split('/')[-1]) #[Update header here]

            hdrs.append(hdr)


        try: self.StatusInsert("Divide by " + self.masterflatfile.split('/')[-1])
        except: self.StatusInsert("Divide by [FLAT]")
        return imgs_nf, hdrs


    def deskew(self, objtype, img, groupID, header='', deskew_path='./standardmap/', outname=''):

        '''
        if finetune:
            flatfile = self.fileprefix+"FLAT.fits"
            flat, header = readfits(flatfile) #for test only

            self.StatusInsert("Extracting orders...using "+flatfile.rstrip(".fits"))
        '''

        if deskew_path == './standardmap/':
            self.StatusInsert("Extracting orders...using predefined functions")
        else:
            self.StatusInsert("Extracting orders...using custom-made functions")

        ny, nx = img.shape#2048,2048

        ap_width = 60
        apfiles = \
          glob.glob(deskew_path+'apmap_%s_%02d.*.dat' % \
                       (self.band, AP_DEGREE))
        apfiles.sort()

        if len(apfiles) == 0:
            self.StatusInsert("No data files of the transformation function")
        else:
            if deskew_path == './standardmap/':
                wlfiles = \
                  glob.glob(deskew_path+'apwav_%s_%02d_%02d.*.dat' % \
                            (self.band, WL_DEGREE[0], WL_DEGREE[1]))
                wlfiles.sort()
                ostrips, owaves, ohdrs = \
                  extract_ap(img, apfiles, wlfiles=wlfiles, \
                                 ap_width=ap_width, header=header)
            else:
                ostrips, ohdrs = \
                  extract_ap_custom(img, apfiles, wlfiles=None, \
                                 ap_width=ap_width, header=header)

            nap = len(ostrips)
            stripfiles = []
            for i in range(0,nap):
                stripfile = self.fileprefix+objtype+'_G'+str(groupID)+outname+'.'+'%03d' % (i+1)
                #self.StatusInsert("Saving... "+stripfile)
                #savefits(self.reducepath+stripfile+'.fits',ostrips[i],ohdrs[i])
                self.save1file(stripfile,ostrips[i],header=ohdrs[i])
                stripfiles.append(stripfile+'.fits')



        '''
        rimg = np.zeros([ny, nx])
        rwave = np.zeros([ny, nx])
        rap_y = []
        y0 = 0
        apnum = 1
        for strip, wave in zip(ostrips, owaves):
            rimg[y0:(y0+ap_width),:] = strip
            rwave[y0:(y0+ap_width),:] = wave
            rap_y.append(y0)
            y0 = y0 + ap_width
            stripimg.append(rimg)
            stripfile = self.fileprefix+objtype+'.'+'ap%02d' % apnum
            savefits(self.reducepath+stripfile+'.fits',rimg)
            apnum = apnum+1
        '''

        return stripfiles


    def wavelength_oh(self, objtype, groupID): #by Huynh Anh

        if not os.path.exists(self.datpath): os.mkdir(self.datpath) #2014-02-18 CKSim inserted this line.

        if self.band == 'H':
            number = range(1,24)
            order =  range(1,24)
            start = 0
        if self.band == 'K':
            number = range(1,20)
            order =  range(1,20)
            start = 2

        for i in range(start, max(number)):

            print '>>>>>>> Working with aperture', i+1

            if i <=8:
                order[i] = '00' + str(number[i])
            else:
                order[i] = '0' + str(number[i])
            stripfile = self.reducepath + self.fileprefix + objtype + '_G' + str(groupID) +'.ONOFF.' + str(order[i]) + '.fits'

            # Referent lines
            #lxx = pixel[i]
            #lwave = wavelength[i]
            #print 'lxx', lxx, lwave
            #t_image, delta_shift, coeff_transform, linear_par, linear_wave, lxx_tr = calibration(order[i], stripfile, lxx, lwave, \
            #                                                                                     datpath=self.datpath, datapath=self.reducepath, \
            #                                                                                     outputname='t' + self.fileprefix + objtype + '_G'+ str(groupID) + '.ONOFF.')

            imgv1, ahdr = ip.readfits(stripfile)
            col, line = imgv1.shape
            img = imgv1[5:(line-5), :]
            ny, nx = img.shape
            #print 'len img', ny, nx
            wavescale, mhdr = ip.readfits('./standardmap/' + 'wavemap_' + self.band + '_02_ohlines_' + str(i) + '.fits')
            #print 'len wavescale', len(wavescale)
            linear_wave = mhdr.get('CRVAL1') + mhdr.get('CDELT1') * np.arange(nx)
            delta_lamda = mhdr.get('CDELT1')

            if self.band =='H':
                lxx_tr = hband.pixel_t[i]
                lwave = hband.wavelength[i]
            else:
                lxx_tr = kband.pixel_t[i]
                lwave = kband.wavelength[i]

            linear_par = [mhdr.get('LCOEFF1'), mhdr.get('LCOEFF2')]

            tstrip = transform_p(img, wavescale)

            thdr = mhdr.copy()

            outputfile = self.fileprefix + objtype + '_G' + str(groupID) +'.ONOFF.tr.' + str(order[i]) + '.fits'
            ip.savefits(self.reducepath + outputfile, tstrip, header=thdr)
            np.savetxt(self.reducepath + self.fileprefix + objtype + '_G' + str(groupID) +'.ONOFF.tr.' + str(order[i]) + '.wave', linear_wave)

            check(stripfile, order[i], tstrip, lxx_tr, lwave, linear_par, linear_wave, pngpath= self.pngpath, datpath=self.datpath,\
                    outputname= self.band + '_result_t' + self.fileprefix + objtype + '_G'+ str(groupID) + '.ONOFF.')

            # Plot results
            final = plt.figure(1, figsize=(16,6))
            ax = final.add_subplot(111)
            a = np.genfromtxt(self.datpath + self.band + '_result_t' + self.fileprefix + objtype + '_G'+ str(groupID) + '.ONOFF.' + str(order[i]) + '.dat')
            ax.plot(a[:,0], a[:,3], 'k.')
            plt.xlabel('Wavelength [um]')
            plt.ylabel('Delta X [pixels]')
            plt.title('Distortion correction and wavelength calibration')

            x_majorLocator = MultipleLocator(0.02)
            x_majorFormatter = FormatStrFormatter('%0.3f')
            x_minorLocator = MultipleLocator(0.004)
            #y_majorLocator = MultipleLocator(max(a[:,3] - min(a[:,3])/ 10))
            #y_majorFormatter = FormatStrFormatter('%0.2f')
            #y_minorLocator = MultipleLocator(max(a[:,3] - min(a[:,3])/ 50))

            ax.xaxis.set_major_locator(x_majorLocator)
            ax.xaxis.set_major_formatter(x_majorFormatter)
            ax.xaxis.set_minor_locator(x_minorLocator)
            #ax.yaxis.set_major_locator(y_majorLocator)
            #ax.yaxis.set_major_formatter(y_majorFormatter)
            #ax.yaxis.set_minor_locator(y_minorLocator)

        plt.show()


    def transform_aperture(self, objtype, groupID):

        if self.band == 'H':
            number = range(1,24)
            order =  range(1,24)
            start = 0
        if self.band == 'K':
            number = range(1,20)
            order =  range(1,20)
            start = 2

        tr_stripfiles=[]


        for i in range(start, max(number)):

            print '>>>>>>> Working with aperture', i+1

            if i <=8:
                order[i] = '00' + str(number[i])
            else:
                order[i] = '0' + str(number[i])

            stripfile = self.fileprefix + objtype + '_G' + str(groupID) +'.AB.' + str(order[i]) + '.fits'
            imgv1, ahdr = ip.readfits(self.reducepath + stripfile)
            col, line = imgv1.shape
            img = imgv1[5:(line-5), :]
            ny, nx = img.shape
            #print 'len, img', nx, ny
            wavescale, mhdr = ip.readfits('./standardmap/' + 'wavemap_' + self.band + '_02_ohlines_' + str(i) + '.fits')
            #print 'len wavescale', len(wavescale)
            linear_wave = mhdr.get('CRVAL1') + mhdr.get('CDELT1') * np.arange(nx)
            delta_lamda = mhdr.get('CDELT1')

            tstrip = transform_p(img, wavescale)

            thdr = mhdr.copy()

            outputfile = self.fileprefix + objtype + '_G' + str(groupID) +'.AB.tr.' + str(order[i]) + '.fits'
            ip.savefits(self.reducepath + outputfile, tstrip, header=thdr)
            np.savetxt(self.reducepath + self.fileprefix + objtype + '_G' + str(groupID) +'.AB.tr.' + str(order[i]) + '.wave', linear_wave)

            tr_stripfiles.append(outputfile)

        return tr_stripfiles

    def identify(self,objtype,stripfiles,makepng=False):
        self.StatusInsert('Transforming stripes.....using predefined functions')
        tr_stripfiles=[]

        for item in stripfiles:
            output=item[:-9]+'.tr'+item[-9:-5]
            #output=item[:-9]+item[-9:-5]

            transform_ap_file(self.reducepath+item,outputfile=self.reducepath+output+'.fits')

            self.StatusInsert('Saved... '+output+'.fits')
            tr_stripfiles.append(output+'.fits')

            if makepng == True:
                draw_strips_file(self.reducepath+output+'.fits', self.reducepath+output+'.wave', linefile='ohlines.dat', \
                                 desc=output, target_path=self.pngpath)

        return tr_stripfiles


    def trace_n_average_AB(self,filenames,pngpath=False):
        '''
        #2013-11-22 cksim
        trace A and B and do the subtraction A-B using simple rectangular (distortion correction should have done good)
        input: filenames (fits file names after distortion correction and wavelength calibration)
        output: (A-B)s fits files, (A-B)s (python numpy 1-d array), wavelength information array ([start wavelength, step])
        requrement: trace_star()
        '''
        self.StatusInsert('Tracing and extracting the A and B stellar signatures...')

        outfiles=[]

        for filename in filenames:

            data, hdr = readfits(self.reducepath+filename)
            col, line = data.shape

            if self.band =='H':
                startpart = 300  # Start from pixel >=100 apply in case of H-band
                endpart = 1900
                data = data[:, startpart:endpart]
            else:
                startpart = 100  # Start from pixel >=100 apply in case of H-band
                endpart = 2000
                data = data[:, startpart:endpart]

            #data = data[15:50,:]
            if hdr == None: hdr = newheader()
            Astrip, Astartline, Aendline, Ahdr = self.trace_star(data, header=hdr)
            Bstrip, Bstartline, Bendline, Bhdr = self.trace_star(data, header=hdr, absorption=True)

            AB = (Astrip.sum(axis=0) - Bstrip.sum(axis=0))/2 #average or the summations along the spatial direction
            AB = np.asarray(AB)

            ABhdr = Ahdr.copy()
            ABhdr.update('IMARITH', 'average A and B beams, (A-(-B))/2')

            # Update wavelength
            startwave = ABhdr['CRVAL1'] + startpart*ABhdr['CDELT1']
            ABhdr.update('CRVAL1', startwave)

            outfile = filename.split('.fits')[0]+'.tpn'
            self.save1file(outfile,AB,header=ABhdr)
            outfiles.append(outfile)


            if pngpath == True:
                wave = [ABhdr['CRVAL1'], ABhdr['CDELT1']]
                f1 = plt.figure(1, figsize=(9,8),dpi=200)
                f1.suptitle(outfile)
                gs = gridspec.GridSpec(3,1,height_ratios=[1,3,3])

                a1 = plt.subplot(gs[0], title='Extracted and wavelength calibrated strip' \
                                 , xlabel='Pixel Number in Spectral Direction' \
                                 , ylabel='Spatial\nDiretion' \
                                 , aspect=0.7)
                z1, z2 = ip.zscale(data)
                a1.imshow(data,cmap='hot', vmin=z1, vmax=z2, origin='lower', aspect='auto')
                for item in ([a1.title, a1.xaxis.label, a1.yaxis.label] + \
                             a1.get_xticklabels() + a1.get_yticklabels()):
                    item.set_fontsize(15)
                plt.setp( a1.get_yticklabels(), visible=False)

                b1 = plt.subplot(gs[1], title='Sum along the spectral direction' \
                                 , xlabel='Pixel Number in Spatial Direction' \
                                 , ylabel='Data Number')
                b1.plot(data.sum(axis=1),'k')
                b1.plot([Astartline,Aendline], [data.sum(axis=1).max()/2,data.sum(axis=1).max()/2], 'r',label='FWHM"')
                b1.plot([Bstartline,Bendline], [data.sum(axis=1).min()/2,data.sum(axis=1).min()/2], 'r')
                b1.plot(max(AB),)
                b1.set_xlim(0,data.shape[0])
                for item in ([b1.title, b1.xaxis.label, b1.yaxis.label] + \
                             b1.get_xticklabels() + b1.get_yticklabels()):
                    item.set_fontsize(15)
                plt.setp( b1.get_yticklabels(), visible=False)

                c1 = plt.subplot(gs[2], title='Average of A and (-B)' \
                                 , xlabel=r'Wavelength ($\mu$m)' \
                                 , ylabel='Intensity')
                nwave = len(Astrip.sum(axis=0))
                wv2 = wave[0] + (nwave*wave[1])
                wavelist = arange(wave[0],wv2, wave[1],dtype=np.float64)
                wavelist = wavelist[:nwave]
                #wave = np.loadtxt(self.reducepath+filename.split('.fits')[0]+'.wave' )
                c1.plot(wavelist, AB, 'k', label='Average')
                #c1.plot(wave, AB, 'k', label='Average')  # edit by Huynh Anh
                c1.set_xlim(wavelist[0],wavelist[-1])
                for item in ([c1.title, c1.xaxis.label, c1.yaxis.label] + \
                             c1.get_xticklabels() + c1.get_yticklabels()):
                    item.set_fontsize(15)
                plt.setp( c1.get_yticklabels(), visible=False)

                f1.tight_layout()
                plt.subplots_adjust(top=0.9)
                plt.savefig(self.pngpath+outfile+'.png')
                #plt.close('all')
                matplotlib.pyplot.close()

        return outfiles


    def trace_star(self, data, header=None, absorption=False):
         #2013-12-05 cksim - 50% of FWHM (following prof. Pak's comments)
         #2013-12-26 cksim - 100% of FWHM (following prof. Pak's comments)

        if absorption == True: sign = (-1)
        else: sign = 1
        ny, nx = data.shape
        y = data[5:(ny-5), :].sum(axis=1) * sign #summation along spectral direction to see the cross=cut of A-B stellar profile
        #plt.plot(y)
        #plt.show()
        thres = max(y) #* 0.5  #FWHM 2013-12-05 cksim

        peakx=(peak_find(y,thres=thres,mode=''))[0]  #'gauss'
        #print 'peak', peakx

        if thres > 0:
            contidion = y > 0.5*thres #y>thres
        else:
            contidion = y > 2*thres

        index=contidion.nonzero()

        print 'index', index[0], len(index[0])

        valid = list(np.where((index[0] - min(index[0])) > 20)[0])

        print 'valid', valid
        index = np.delete(index[0], valid)

        print 'index out', index

        if len(index) > 19:
            startline, endline = peakx-10, peakx+10
            print 'start, end', startline, endline
        else:
            startline,endline = index[0], index[-1]
            print 'start, end', startline, endline
        #2013-12-05 cksim,
        #2013-12-05 cksim: FWHM refers to 5 or 6 pixel in the simulated echellogram. threshold=0.05*max case refers to 11 or 12

        startline = startline +5  # Edit by Huynh Anh check start and end lines
        endline = endline +5
        strip = data[startline:endline+1]
        if header == None: header = newheader()
        header.update('HISTORY', 'Stellar signature tracing, lines:%d-%d' % (startline,endline))

        return strip, startline, endline, header


    def save1file(self,filename,data,header=None):
        if len(data)==1: data = data[0]
        self.StatusInsert("Saving... "+filename)
        if header == None: header=newheader()

        if os.path.isfile(self.reducepath+filename+'.fits'):
            os.remove(self.reducepath+filename+'.fits')
        savefits(self.reducepath+filename+'.fits',data,header=header)



    def H_n_cont_removal(self,filenames,threslow=1.5,threshigh=4.,pngpath=False):
        import scipy.stats as mstats

        self.StatusInsert("Removing H lines from the standard...")

        outfiles=[]
        for filename in filenames:

            data, hdr = readfits(self.reducepath+filename+'.fits')
            spec = np.transpose(data).ravel()

            waveinfo = [hdr['CRVAL1'], hdr['CDELT1']]

            wave = waveinfo[0] + np.arange(len(spec))*waveinfo[1]

            #--H-lines(gaussian absorption)
            FWHM=0.004 # as default
            sigma = FWHM / (2.*math.sqrt(2*math.log(2)))  # Added by Huynh Anh 20140319
            H_lines = [item for item in H_series('Br') if ((item >= wave[0]-FWHM/2.) & (item <= wave[-1]+FWHM/2.))]

            p0=()
            for H_line in H_lines:
                p0 += 0.3,H_line,1,sigma#(FWHM)



            if len(p0) == 0 :
                gfit = np.zeros_like(wave)
                pc = np.polyfit(wave,spec,3)
                cfit = np.polyval(pc,wave)

            elif len(p0) > 0 :
                '''
                Step 1: Gaussian (using p0) + fit a polynomial
                '''
                gfit = gauss_mixture(wave, p0)

                def contfit(p):
                    '''
                    continuum fit (non-Gaussian points + Gaussian centers)
                    '''
                    vc = np.where(gfit < max(gfit)*0.01)[0]
                    if 'vv' in locals(): vc = list( set(vc) & set(vv))
                    specc = spec[vc]
                    for i in range(len(p)/3):
                        peak = p[i*3]
                        center = p[i*3+1]
                        ispeak = np.argmin(abs(wave-center))
                        vc = np.append(vc,ispeak)
                        specc = np.append(specc,spec[ispeak]*(1+peak))
                    issort=np.argsort(vc)
                    vc = vc[issort]
                    specc = specc[issort]

                    pc = np.polyfit(wave[vc],specc,3)
                    cfit = np.polyval(pc,wave)

                    fit = cfit * (1 - gfit)
                    '''
                    continuum fit update (differences fit + sigma-clip)
                    '''
                    _, vv = A0Vsclip(spec,fit, low=threslow, high=threshigh) #vv is list of valid points after sigma-clip
                    specd = spec - (-cfit*gfit)
                    pc = np.polyfit(wave[vv],specd[vv],3)
                    cfit = np.polyval(pc,wave)

                    return cfit, vv

                cfit, vv = contfit(p0)
                fit = cfit * (1 - gfit)

                '''
                Step 2: find new Gaussian (popt using derived continuum fit) and then fit a new polynomial
                '''

                specg = 1 - (spec / cfit)

                try: popt, pcov = curve_fit(gauss_mixture,wave[vv],specg[vv],p0)
                except: popt = p0

                gfit = gauss_mixture(wave,popt)

                cfit, vv = contfit(popt)



            fit = cfit * (1 - gfit)

            spec_hremoved = spec  - (-cfit*gfit)
            spec_cremoved = spec / fit


            for H_line in H_lines:
                hdr.update('HISTORY', 'H line(s) removal (%f7.5 micron' %H_line)

            houtfile = filename+'.Br'
            self.save1file(houtfile,np.transpose(spec_hremoved),header=hdr)

            hdr.update('HISTORY', 'Continuum removal')
            coutfile = filename+'.Br.co'
            self.save1file(coutfile,np.transpose(spec_cremoved),header=hdr)
            outfiles.append(coutfile)


            if not pngpath == False:
                f1 = plt.figure(1, figsize=(9,7), dpi=200)
                f1.suptitle(coutfile)

                p1 = f1.add_subplot(211)#, title='Before')#,xlabel=r'Wavelength ($\mu$m)')
                p1.plot(wave,spec,'k')
                p1.plot(wave,cfit,'g',label='Continuum fit')
                p1.plot(wave,fit,'r',label='Continuum + H lines fit')
                p1.set_xlim(wave[0],wave[-1])
                p1.legend(loc='upper left',bbox_to_anchor=(1,1), fontsize=11)

                p2 = f1.add_subplot(212, title='After Removing..',xlabel=r'Wavelength ($\mu$m)')
                p2.plot(wave,spec_hremoved,'k',label='H lines')
                p2.plot(wave,cfit,'r')#,label='fit')
                p2.plot(wave,spec_cremoved,'b',label='H-lines & Continuum')
                p2.plot(wave,cfit/cfit,'r')#,label='fit')
                p2.set_xlim(wave[0],wave[-1])
                p2.legend(loc='upper left',bbox_to_anchor=(1,1), fontsize=11)

                f1.tight_layout()
                plt.subplots_adjust(top=0.9,right=0.7)
                plt.savefig(pngpath+coutfile+'.png')
                #plt.close('all')
                matplotlib.pyplot.close()

        return outfiles

    def telcor(self,filenames,pngpath=False,threslow=1.,threshigh=4.):

        outfiles=[]
        for filename in filenames:
            stdfilename = filename.replace('TAR','STD')+'.Br.co'
            data, hdr = readfits(self.reducepath+stdfilename+'.fits')
            nstd = np.transpose(data).ravel()

            data, hdr = readfits(self.reducepath+filename+'.fits')
            tar = np.transpose(data).ravel()

            waveinfo = [hdr['CRVAL1'], hdr['CDELT1']]
            w = waveinfo[0] + np.arange(len(tar))*waveinfo[1]

            coeff = np.polyfit(w,tar,3)
            cfit = np.polyval(coeff,w)
            _, vv = A0Vsclip(tar,cfit, low=threslow, high=threshigh) #vv is list of valid points after sigma-clip
            coeff = np.polyfit(w[vv],tar[vv],3)
            cfit = np.polyval(coeff,w)
            ntar = tar / cfit

            tarcor = cfit * (ntar / nstd)

            hdr.update('HISTORY', '/'+stdfilename)
            outfile = filename+'.te'
            self.save1file(outfile,np.transpose(tarcor),header=hdr)
            outfiles.append(outfile)


            if self.makepng == True:
                #2013-11-28 cksim, below plot for the development stage
                f2 = plt.figure(2,figsize=(9,7), dpi=200)
                f2.suptitle(outfile)

                a1 = f2.add_subplot(211, title='Before')#, xlabel=r'Wavelength ($\mu$m)')
                a1.plot(w,tar,'k', label='Target')
                a1.plot(w,cfit,'r', label='Continuum fit')
                a1.plot(w,nstd,'b', label='Normalized Standard')
                a1.legend(loc='upper left',bbox_to_anchor=(1,1), fontsize=11)
                #a1.set_ylim([-1,5])
                a1.set_xlim(w[0],w[-1])

                a2 = f2.add_subplot(212, title='After Telluric Correction', xlabel=r'Wavelength ($\mu$m)')
                a2.plot(w,tarcor, 'k')#, label = 'Result')
                #a2.set_ylim([-1,3])
                a2.set_xlim(w[0],w[-1])\

                f2.tight_layout()
                plt.subplots_adjust(top=0.9,right=0.7)
                plt.savefig(self.pngpath+outfile+'.png')
                #plt.close('all')
                matplotlib.pyplot.close()



        return outfiles

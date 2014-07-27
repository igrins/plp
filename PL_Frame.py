'''
Created on July 24, 2012

@author: bykwon
'''
from Tkinter        import *
import ttk

import threading

from PL_Display    import *
    
class CFrame(ttk.Frame,threading.Thread):
    '''
    classdocs
    '''


    def __init__(self, master, path):
        '''
        Constructor
        '''   
        ttk.Frame.__init__(self, master)
        self.master=master
        self.master.title("Pipeline Package v1.07")
        self.master.iconname("Pipeline Package v1.07")
        
        self.draw = CDisplay(self.master, path)
        
        self.Main()    
        self.Draw()
                   
    def Main(self):
        mainframe = ttk.Panedwindow(self.master, orient=HORIZONTAL)
        self.frame1 = ttk.Frame(mainframe, padding="10 10 5 10")
        self.frame2 = ttk.Frame(mainframe, padding="5 10 10 10")
        
        mainframe.add(self.frame1)
        mainframe.add(self.frame2)
        
        mainframe.grid()
        
        self.Frame1()
        self.Frame2()
    
    def Frame1(self):
        #divided Frame1 by Vertical
        F1 = ttk.Panedwindow(self.frame1, orient=VERTICAL)
        self.f1topf=ttk.Frame(F1)
        space=ttk.Frame(F1)
        f1bottomf=ttk.Frame(F1)        
        
        F1.add(self.f1topf)
        F1.add(space)
        F1.add(f1bottomf)
        F1.grid()
        
        F3=ttk.PanedWindow(f1bottomf, orient=HORIZONTAL)
        F3calf=ttk.Labelframe(F3,text='Calibration Frame')
        F3objf=ttk.Labelframe(F3,text='Object Frame')
        
        F3.add(F3calf)
        F3.add(F3objf)
        F3.grid()
        
       # f1bottomf.add(F3cal)
        #f1bottomf.add(F3obj)
        #f1bottomf.grid()
        
        F3caltop=ttk.Panedwindow(F3calf, orient=VERTICAL)
        F3caltopf=ttk.Frame(F3caltop)
        F3caltop.add(F3caltopf)
        F3caltop.grid()
        
        F3calbottom=ttk.Panedwindow(F3calf, orient=VERTICAL)
        self.Cexecutef = ttk.Frame(F3calbottom) 
        spacec=ttk.Frame(F3calbottom)
        F3calbottom.add(spacec)
        F3calbottom.add(self.Cexecutef)
        F3calbottom.grid()
        
        F3objtop=ttk.Panedwindow(F3objf, orient=VERTICAL)
        F3objtopf=ttk.Frame(F3objtop)        
        F3objtop.add(F3objtopf)
        F3objtop.grid()
        
        F3objbottom=ttk.Panedwindow(F3objf, orient=VERTICAL)
        self.Oexecutef = ttk.Frame(F3objbottom)
        spaceo=ttk.Frame(F3objbottom) 
        F3objbottom.add(spaceo)
        F3objbottom.add(self.Oexecutef)
        F3objbottom.grid()
        
        F3c=ttk.Panedwindow(F3caltopf, orient=HORIZONTAL)
#        calfram1=ttk.Labelframe(F3c,text="Dark")
        calfram1=ttk.Frame(F3c)#)
        space1=ttk.Frame(F3c)
        calfram2=ttk.Frame(F3c)#,text="Flat")
        space2=ttk.Frame(F3c)
        calfram3=ttk.Frame(F3c)#,text="Arc")        
        space3=ttk.Frame(F3c)
        
        F3o=ttk.Panedwindow(F3objtopf, orient=HORIZONTAL)
        objfram1=ttk.Frame(F3o)#,text="Standard")
        space4=ttk.Frame(F3o)
        objfram2=ttk.Frame(F3o)#,text="Target")
        
        F3c.add(calfram1)
        F3c.add(space1)
        F3c.add(calfram2)
        F3c.add(space2)
        F3c.add(calfram3)
        F3c.add(space3)
        F3c.grid()
                
        F3o.add(objfram1)
        F3o.add(space4)
        F3o.add(objfram2)
        F3o.grid()
        
        #calibration frame
        #dark
        F41 = ttk.Panedwindow(calfram1, orient=VERTICAL)
        calibnote1=ttk.Frame(F41)
        self.calib11 = ttk.Frame(F41)
        space12=ttk.Frame(F41)
        self.ccm1 = ttk.Labelframe(F41, text="Combine method")
        space13 = ttk.Frame(F41)
        self.calib12 = ttk.Frame(F41)
        
        F41.add(calibnote1)
        F41.add(self.calib11)
        F41.add(space12)
        F41.add(self.ccm1)
        F41.add(space13)
        F41.add(self.calib12)
        F41.grid()
        
           #flat
        F42 = ttk.Panedwindow(calfram2, orient=VERTICAL)
        calibnote2=ttk.Frame(F42)
        self.calib21 = ttk.Frame(F42)
        space22=ttk.Frame(F42)
        self.ccm2 = ttk.Labelframe(F42, text="Combine method")
        space23 = ttk.Frame(F42)
        self.calib22 = ttk.Frame(F42)
        
        F42.add(calibnote2)
        F42.add(self.calib21)
        F42.add(space22)
        F42.add(self.ccm2)
        F42.add(space23)
        F42.add(self.calib22)
        F42.grid()

        #arc
        F43 = ttk.Panedwindow(calfram3, orient=VERTICAL)
        calibnote3=ttk.Frame(F43)
        self.calib31 = ttk.Frame(F43)
        space32=ttk.Frame(F43)
        self.ccm3 = ttk.Labelframe(F43, text="Combine method")
        space33 = ttk.Frame(F43)
        self.calib32 = ttk.Frame(F43)
        
        F43.add(calibnote3)
        F43.add(self.calib31)
        F43.add(space32)
        F43.add(self.ccm3)
        F43.add(space33)
        F43.add(self.calib32)
        F43.grid()

        
        #calibration notebook
#        calibn = ttk.Notebook(calibnote)
#        calibl1 = Listbox(calibnote1)
#        self.Cflatf = ttk.Frame(calibn) 
#        self.Carcf = ttk.Frame(calibn)
        self.Cdarkf = ttk.Frame(calibnote1) 
        self.Cdarkf.grid()
#        calibl1.grid()
#        calibl2 = Listbox(calibnote2)
        self.Cflatf = ttk.Frame(calibnote2)
        self.Cflatf.grid()
#        calibl2.grid()
#        calibl3 = Listbox(calibnote3)
        self.Carcf = ttk.Frame(calibnote3) 
        self.Carcf.grid()
#        calibl3.grid()
#        calibn.add(self.Cflatf, text='FLAT')
#        calibn.add(self.Carcf, text='ARC')
#        calibn.grid()
        
        #Object frame std
        F51 = ttk.Panedwindow(objfram1, orient=VERTICAL)
        objnote1=ttk.Frame(F51)
        self.obj11 = ttk.Frame(F51)
        space14=ttk.Frame(F51)
        self.ocm1 = ttk.Labelframe(F51, text="Combine method")
        space15 = ttk.Frame(F51)
        self.obj12 = ttk.Frame(F51)
        
        F51.add(objnote1)
        F51.add(self.obj11)
        F51.add(space14)
        F51.add(self.ocm1)
        F51.add(space15)
        F51.add(self.obj12)
        F51.grid()
        
          #Object frame tar
        F52 = ttk.Panedwindow(objfram2, orient=VERTICAL)
        objnote2=ttk.Frame(F52)
        self.obj21 = ttk.Frame(F52)
        space24=ttk.Frame(F52)
        self.ocm2 = ttk.Labelframe(F52, text="Combine method")
        space25 = ttk.Frame(F52)
        self.obj22 = ttk.Frame(F52)
        
        F52.add(objnote2)
        F52.add(self.obj21)
        F52.add(space24)
        F52.add(self.ocm2)
        F52.add(space25)
        F52.add(self.obj22)
        F52.grid()
      
        #object notebook
#        objn = ttk.Notebook(objnote)
#        self.Ostandardf = ttk.Frame(objn) 
#        self.Otargetf = ttk.Frame(objn) 
#        objn.add(self.Ostandardf, text='Standard')
#        objn.add(self.Otargetf, text='Target')
#        objn.grid()
        self.Ostandardf = ttk.Frame(objnote1) 
        self.Ostandardf.grid()
        self.Otargetf = ttk.Frame(objnote2) 
        self.Otargetf.grid()
        
    def Frame2(self):
        F2 = ttk.Panedwindow(self.frame2, orient=VERTICAL)
        self.status=ttk.Frame(F2)
        F2.add(self.status)
#        F2.config(width=1024,height=500)
        F2.grid() 
        
    
    def Draw(self):
        
        #Top Frame in Frame1
        self.draw.Location(self.f1topf)
        
        #Calibration Frame
        #self.draw.CalibrationDARK(self.Cdarkf)  # Keep Dark frame for future work (now do not show)
        #self.draw.CalibrationInfo11(self.calib11)
        #self.draw.CalibrationCombineMethod1(self.ccm1)
 #       self.draw.CalibrationInfo12(self.calib12)
        
        self.draw.CalibrationFLAT(self.Cflatf)
        self.draw.CalibrationInfo21(self.calib21)
        self.draw.CalibrationCombineMethod2(self.ccm2)
        self.draw.CalibrationInfo22(self.calib22)
        
        self.draw.CalibrationARC(self.Carcf)
        self.draw.CalibrationInfo31(self.calib31)
        self.draw.CalibrationCombineMethod3(self.ccm3)
        self.draw.CalibrationInfo32(self.calib32)
        
        self.draw.CalibrationExecute(self.Cexecutef)
        self.draw.ObjectExecute(self.Oexecutef)
        
        #Object Frame
        self.draw.ObjectStandard(self.Ostandardf)
        self.draw.ObjectInfo11(self.obj11)
        self.draw.ObjectCombineMethod1(self.ocm1)
        #self.draw.ObjectInfo12(self.obj12)
        
        self.draw.ObjectTarget(self.Otargetf)
        self.draw.ObjectInfo21(self.obj21)
        self.draw.ObjectCombineMethod2(self.ocm2)
        #self.draw.ObjectInfo22(self.obj22)

        
        #Frame2   
#        self.draw.Status(self.status)
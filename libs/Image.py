'''
Created on Jun 06, 2011

@author: jmlee
'''
from Tkinter        import *
import ttk

class CImage(object):
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
        
    def ImageMode(self, master):
        readoutvar = StringVar()
        exptimevar = StringVar()
        ttk.Label(master, width=18, text="Readout Mode").grid(row=1, column=1, padx=5)
        readout = ttk.Combobox(master, width=12, textvar=readoutvar)
        readout.grid(row=1, column=2, padx=15, pady=5)
        ttk.Label(master, width=18, text="Exposure Time").grid(row=2, column=1, padx=5)
        exptime = ttk.Entry(master, width=12, textvariable=exptimevar)
        exptime.grid(row=2, column=2, padx=15, pady=5)
        
        return readout, readoutvar, exptime, exptimevar
        
    def ImageStorage(self, master):
        autosave_flag = IntVar()
        ttk.Checkbutton(master, 
                        text="Automatically Save ", 
                        onvalue=1, offvalue=0, 
                        variable=autosave_flag
                        ).grid(row=1, column=1, 
                               columnspan=2, 
                               padx=5, sticky=W)
        usedefault_flag = IntVar()
        ttk.Checkbutton(master, 
                        text="Use Default Path", 
                        onvalue=1, offvalue=0, 
                        variable=usedefault_flag
                        ).grid(row=2, column=1, 
                               columnspan=2, 
                               padx=5, sticky=W)
        location_path = StringVar() 
        ttk.Entry(master, width=30, 
                  textvariable=location_path
                  ).grid(row=3, column=1, pady=5, sticky=E)
        savebtn = ttk.Button(master, text="SAVE")
        savebtn.grid(row=3,column=2,sticky=E,padx=5,pady=2)
        
        return autosave_flag, usedefault_flag, location_path, savebtn 
                          
         
    def ImageStatus(self, master):
        etime = (StringVar(), StringVar(), StringVar())
              
        ttk.Label(master, width=15, text="Start Time : ").grid(row=1, column=1, padx=5, pady=2, sticky=W)
        ttk.Label(master, width=15, text="End Time : ").grid(row=2, column=1, padx=5, pady=2, sticky=W)
        ttk.Label(master, width=15, text="Elapsed Time : ").grid(row=3, column=1, padx=5, pady=2, sticky=W)
        ttk.Label(master, foreground="red", textvariable=etime[0]).grid(row=1,column=2,padx=5,pady=2, sticky=W)
        ttk.Label(master, foreground="red", textvariable=etime[1]).grid(row=2,column=2,padx=5,pady=2, sticky=W)
        ttk.Label(master, foreground="red", textvariable=etime[2]).grid(row=3,column=2,padx=5,pady=2, sticky=W)
        
        percent = IntVar() 
        ttk.Progressbar(master, orient=HORIZONTAL, 
                        length=250, mode='determinate', 
                        variable=percent
                        ).grid(row=4,column=1,columnspan=2,padx=5,pady=2)
        percent.set(0)
        
        return etime, percent 
from Tkinter import * 
import datetime, time 
import ConfigParser as cp 

import sys
sys.path.append("..")
sys.path.append("../..")

class CSetConfig(Frame):
    
    def __init__(self, master, mod=""):
        
        self.path = '../../System/Config/IGRINS.ini'
        
        cfg = LoadConfig(self.path)
        self.cfg = cfg
        
        Frame.__init__(self, master)
        master.title("IGRINS Configuration Setting - "+mod)
        
        if mod == "" : mod_names = self.cfg.sections()
        else: mod_names = [mod] 
        frames = [] 
        
        ent_idx = 0
        entries = []  
        for i, mod_name in enumerate(mod_names): 
            frames.append(LabelFrame(master, text=mod_name, padx=5, pady=5))
            ent_names = self.cfg.options(mod_name)
            
            for ent_name in ent_names:
                # print out only WANTED list 
                
                
                Label(frames[i], text=ent_name, width=15).grid(row=ent_idx, column=1, sticky=W)
                entries.append((mod_name, ent_name, StringVar()))
                Entry(frames[i], textvariable=entries[ent_idx][2], width=40).grid(row=ent_idx, column=2)
                entries[ent_idx][2].set(self.cfg.get(mod_name, ent_name))
                ent_idx += 1 
            frames[i].grid(row=1, column=i, sticky="N E W S")
        
        frames.append(Frame(master, padx=5, pady=5))
        Button(frames[i+1], text="Load", padx=5, pady=5, command=self.LoadParams).grid(row=i+1, column=0)
        Button(frames[i+1], text="Save", padx=5, pady=5, command=self.SaveParams).grid(row=i+1, column=0)
        frames[i+1].grid(row=i+1, column=1)
        self.entries = entries
        self.mod_names = mod_names
        self.ent_names = ent_names  
        
    def SaveParams(self):
        for mod_name, ent_name, ent_val in self.entries:
            self.cfg.set(mod_name, ent_name, ent_val.get())
            
        SaveConfig(cfg=self.cfg, path=self.path)
    
    def LoadParams(self):
        for mod_name, ent_name, ent_val in self.entries:
            ent_val.set(self.cfg.get(mod_name, ent_name))
         
        
                       
def LoadConfig(path=None):
    
    if path == None: return 
    # load config object 
    cfg = cp.ConfigParser()
    
    # read the configure file 
    cfg.read(path)
    
    # read the device list string and list and time-out interval  
    
    # save the parameters 
    return cfg 
            
def SaveConfig(cfg=None, path=None):
    
    if cfg == None: return 
        
    with open(path, 'wb') as cfgfile:
        cfg.write(cfgfile)        
        
        
if __name__ == "__main__":
    root = Tk()
    CSetConfig(master=root).mainloop() 

else :    
    def Run(mod=None):
        root = Toplevel()
        CSetConfig(master=root, mod=mod).winfo_toplevel()    
           
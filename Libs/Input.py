'''
Created on Jun 06, 2011

@author: jmlee
'''
from Tkinter        import *
import ttk

from InputCommon    import *

class CEntry(object):
    '''
    classdocs
    '''
        
    def __init__(self, master, rownum, colnum):
        '''
        Constructor
        '''
        self.master = master
        self.rownum = rownum
        self.colnum = colnum        
        
    def ShowEntry(self):
        name = ttk.Label(self.master, text=self.name)
        name.grid(row=self.rownum, column=self.colnum, sticky=W, padx=5, pady=5)
                           
    
class CEntryMaxMin(CEntry):
    '''
    classdocs
    '''
        
    def __init__(self, master, entryname, rownum, colnum):
        '''
        Constructor
        '''      
        CEntry.__init__(self, master, rownum, colnum)
        self.name = entryname
        
    def ShowEntry(self):
        name = ttk.Label(self.master, text=self.name)
        name.grid(row=self.rownum, column=self.colnum, sticky=W, padx=5, pady=5)
        self.max = ttk.Entry(self.master, width=5, justify=RIGHT)
        self.max.grid(row=self.rownum, column=self.colnum + 1, padx=5)
        self.min = ttk.Entry(self.master, width=5, justify=RIGHT)
        self.min.grid(row=self.rownum, column=self.colnum + 2, padx=5)
               
    def ShowValue(self, v1, v2):
        if v1[len(v1)-1] == '\n':
            v1 = v1[:len(v1)-1]
        self.max.delete(0, END)
        self.max.insert(0, v1)
        
        if v2[len(v2)-1] == '\n':
            v2 = v2[:len(v2)-1]      
        self.min.delete(0, END)
        self.min.insert(0, v2)
    
    def SaveEntry(self, s, file):
        str = s + self.max.get().replace("\n", "") + ',' + self.min.get().replace("\n", "") 
        file.writelines(str + '\n')
        
            
class CEntryIp(CEntry):
    '''
    classdocs
    '''
        
    def __init__(self, master, entryname, rownum, colnum):
        '''
        Constructor
        '''
        CEntry.__init__(self, master, rownum, colnum)
        self.name = entryname
        self.TYPES = (self.ip1, self.ip2,
                      self.ip3, self.ip4) = (ttk.Entry(self.master, width=3, justify=RIGHT),
                                             ttk.Entry(self.master, width=3, justify=RIGHT),
                                             ttk.Entry(self.master, width=3, justify=RIGHT),
                                             ttk.Entry(self.master, width=3, justify=RIGHT))   
    def ShowEntry(self):
        name = ttk.Label(self.master, text=self.name)
        name.grid(row=self.rownum, column=self.colnum, sticky=W, padx=5, pady=5)
        ttk.Frame(self.master, width=15).grid(row=self.rownum, column=self.colnum + 1)
        
        for i, e in enumerate(self.TYPES):
            e.grid(row=self.rownum, column=self.colnum + i * 2 + 2)
            if i < 3:
                ttk.Label(self.master, text=".").grid(row=self.rownum, column=self.colnum + i * 2 + 3)
    
    def ShowValue(self, *v):
        for i, e in enumerate(self.TYPES):
            e.delete(0, END)
            s = v[i]
            if s[len(s)-1] == '\n':
                s = s[:len(s)-1]
            e.insert(0, s)
            
    def SaveEntry(self, s, file):
        str = s + self.ip1.get().replace("\n", "") + '.' + self.ip2.get().replace("\n", "") + '.' + self.ip3.get().replace("\n", "") + '.' + self.ip4.get().replace("\n", "")  
        file.writelines(str + '\n')
    
class CEntryPort(CEntry):
    '''
    classdocs
    '''
        
    def __init__(self, master, entryname, rownum, colnum):
        '''
        Constructor
        '''
        CEntry.__init__(self, master, rownum, colnum)
        self.name = entryname
        
    def ShowEntry(self):
        name = ttk.Label(self.master, text=self.name)
        name.grid(row=self.rownum, column=self.colnum, sticky=W, padx=5, pady=5)
        ttk.Frame(self.master, width=15).grid(row=self.rownum, column=self.colnum + 1)
        self.e = ttk.Entry(self.master, width=15, justify=RIGHT)
        self.e.grid(row=self.rownum, column=self.colnum + 2, padx=5)
    
    def ShowValue(self, v1):
        self.e.delete(0, END)
        if v1[len(v1)-1] == '\n':
            v1 = v1[:len(v1)-1]
        self.e.insert(0, v1)
                   
    def SaveEntry(self, s, file):
        str = s + self.e.get().replace("\n", "")
        file.writelines(str + '\n')

class CEntryCurTarHea(CEntry):
    '''
    classdocs
    '''
        
    def __init__(self, master, entryname, rownum, colnum):
        '''
        Constructor
        '''      
        CEntry.__init__(self, master, rownum, colnum)
        self.name = entryname
        
    def ShowEntry(self):
        name = ttk.Label(self.master, text=self.name, width=10)
        name.grid(row=self.rownum, column=self.colnum, sticky=W, padx=5, pady=5)
        
        self.cur = ttk.Entry(self.master, width=7, takefocus=0, justify=RIGHT)
        self.cur.grid(row=self.rownum, column=self.colnum + 1, padx=5)
        
        self.tar = ttk.Entry(self.master, width=7, takefocus=1, justify=RIGHT)
        self.tar.grid(row=self.rownum, column=self.colnum + 2, padx=5)
        
        self.pow = ttk.Entry(self.master, width=7, takefocus=1, justify=RIGHT)
        self.pow.grid(row=self.rownum, column=self.colnum + 3, padx=5)
        
    def ShowValue(self, data,state):
        self.cur.config(state="active")
        self.cur.delete(0, END)
        self.cur.insert(0, data)
        if state == "warm":
            self.cur.config(foreground='red')
        else:
            self.cur.config(foreground='black')
        
        self.cur.config(state="readonly")
        
    def ShowValuePow(self, data):
        self.pow.config(state="active")
        self.pow.delete(0, END)
        self.pow.insert(0, data)
        self.pow.config(state="readonly")
    def ShowValueTar(self, data):
        self.tar.config(state="active")
        self.tar.delete(0, END)
        self.tar.insert(0, data)
        
    
    def GetSetting(self):
        return self.tar.get()
        
    def SaveValue(self):
        return ' %8.3f' % float(self.cur.get().replace("\n", ""))

class CEntryCurTarHeaTem(CEntry):
    '''
    classdocs
    '''
        
    def __init__(self, master, entryname, rownum, colnum):
        '''
        Constructor
        '''      
        CEntry.__init__(self, master, rownum, colnum)
        self.name = entryname
        
    def ShowEntry(self):
        name = ttk.Label(self.master, text=self.name, width=10)
        name.grid(row=self.rownum, column=self.colnum, sticky=W, padx=5, pady=5)
        
        self.cur = ttk.Entry(self.master, width=7, takefocus=0, justify=RIGHT)
        self.cur.grid(row=self.rownum, column=self.colnum + 1, padx=5)
        
        self.tar = ttk.Entry(self.master, width=7, takefocus=1, justify=RIGHT)
        self.tar.grid(row=self.rownum, column=self.colnum + 2, padx=5)
        self.tar.config(state='disable')
        self.pow = ttk.Entry(self.master, width=7, takefocus=1, justify=RIGHT)
        self.pow.grid(row=self.rownum, column=self.colnum + 3, padx=5)
        self.pow.config(state='disable')
    def ShowValue(self, data,state):
        self.cur.config(state="active")
        self.cur.delete(0, END)
        self.cur.insert(0, data)
        if state == "warm":
            self.cur.config(foreground='red')
        else:
            self.cur.config(foreground='black')
        
        self.cur.config(state="readonly")
        
    def ShowValuePow(self, data):
        self.pow.config(state="active")
        self.pow.delete(0, END)
        self.pow.insert(0, data)
        self.pow.config(state="readonly")
    def ShowValueTar(self, data):
        self.tar.config(state="active")
        self.tar.delete(0, END)
        self.tar.insert(0, data)
        
    
    def GetSetting(self):
        return self.tar.get()
        
    def SaveValue(self):
        return ' %8.3f' % float(self.cur.get().replace("\n", ""))


class CEntryCurTar(CEntry):
    '''
    classdocs
    '''
        
    def __init__(self, master, entryname, rownum, colnum):
        '''
        Constructor
        '''      
        CEntry.__init__(self, master, rownum, colnum)
        self.name = entryname
        
    def ShowEntry(self):
        name = ttk.Label(self.master, text=self.name, width=10)
        name.grid(row=self.rownum, column=self.colnum, sticky=W, padx=5, pady=5)
        
        self.cur = ttk.Entry(self.master, width=7, takefocus=0, justify=RIGHT)
        self.cur.grid(row=self.rownum, column=self.colnum + 1, padx=5)
        
        self.tar = ttk.Entry(self.master, width=7, takefocus=1, justify=RIGHT)
        self.tar.grid(row=self.rownum, column=self.colnum + 2, padx=5)
        '''
        self.pow = ttk.Entry(self.master, width=7, takefocus=1, justify=RIGHT)
        self.pow.grid(row=self.rownum, column=self.colnum + 3, padx=5)
        '''
    def ShowValue(self, data):
        self.cur.config(state="active")
        self.cur.delete(0, END)
        self.cur.insert(0, data)
        self.cur.config(state="readonly")
    '''    
    def ShowValuePow(self, data):
        self.pow.config(state="active")
        self.pow.delete(0, END)
        self.pow.insert(0, data)
        self.pow.config(state="readonly")
    '''
    def GetSetting(self):
        return self.tar.get()
        
    def SaveValue(self):
        return ' %8.3f' % float(self.cur.get().replace("\n", ""))
   
class CEntryCur(CEntry):
    '''
    classdocs
    '''
        
    def __init__(self, master, entryname, rownum, colnum):
        '''
        Constructor
        '''      
        CEntry.__init__(self, master, rownum, colnum)
        self.name = entryname
        
    def ShowEntry(self):
        name = ttk.Label(self.master, text=self.name, width=12)
        name.grid(row=self.rownum, column=self.colnum, sticky=W, padx=5, pady=5)
        
        self.cur = ttk.Entry(self.master, width=7, takefocus=0, justify=RIGHT)
        self.cur.grid(row=self.rownum, column=self.colnum + 1, padx=5)
        
    def ShowValue(self, data,state):
        self.cur.config(state="active")
        self.cur.delete(0, END)
        self.cur.insert(0, data)
        if state == "warm":
            self.cur.config(foreground='red')
        else:
            self.cur.config(foreground='black')
        
        self.cur.config(state="readonly")
        
        
    def GetSetting(self):
        return self.tar.get()
        
    def SaveValue(self):
    	try:
    		float(self.cur.get())
        	return ' %8.3f' % float(self.cur.get().replace("\n", ""))
        except ValueError:
        	return "    "+self.cur.get().replace("\n", "")

class CEntryCurVol(CEntry):
    '''
    classdocs
    '''
        
    def __init__(self, master, entryname, rownum, colnum):
        '''
        Constructor
        '''      
        CEntry.__init__(self, master, rownum, colnum)
        self.name = entryname
        
    def ShowEntry(self):
        name = ttk.Label(self.master, text=self.name, width=12)
        name.grid(row=self.rownum, column=self.colnum, sticky=W, padx=5, pady=5)
        
        self.vol = ttk.Entry(self.master, width=7, takefocus=0, justify=RIGHT)
        self.vol.grid(row=self.rownum, column=self.colnum + 1, padx=5)
        
        
        self.cur = ttk.Entry(self.master, width=7, takefocus=0, justify=RIGHT)
        self.cur.grid(row=self.rownum, column=self.colnum + 2, padx=5)
        
        
        
    def ShowCurValue(self, data):
        self.cur.config(state="active")
        self.cur.delete(0, END)
        self.cur.insert(0, data)
        self.cur.config(state="readonly")
    
    def ShowVolValue(self, data):
        self.vol.config(state="active")
        self.vol.delete(0, END)
        self.vol.insert(0, data)
        self.vol.config(state="readonly")
        
class CEntryHea(CEntry):
    '''
    classdocs
    '''
        
    def __init__(self, master, entryname, rownum, colnum):
        '''
        Constructor
        '''      
        CEntry.__init__(self, master, rownum, colnum)
        self.name = entryname
        
    def ShowEntry(self):
        name = ttk.Label(self.master, text=self.name, width=10)
        name.grid(row=self.rownum, column=self.colnum, sticky=W, padx=5, pady=5)
        
        self.hea = ttk.Entry(self.master, width=7, takefocus=0, justify=RIGHT)
        self.hea.grid(row=self.rownum, column=self.colnum + 1, padx=5)
        
    def ShowValue(self, data):
        self.hea.config(state="active")
        self.hea.delete(0, END)
        self.hea.insert(0, data)
        self.hea.config(state="readonly")
        
    def GetSetting(self):
        return self.tar.get()
        
    def SaveValue(self):
        return ' %8.3f' % float(self.cur.get().replace("\n", ""))
        
class CEntryRADec(CEntry):
    '''
    classdocs
    '''
        
    def __init__(self, master, entryname, rownum, colnum):
        '''
        Constructor
        '''      
        CEntry.__init__(self, master, rownum, colnum)
        self.MODES = [("1", "RA", "h", "m", "s"), ("2", "Dec", "", "", "")]
        
    def ShowEntry(self, master):
        for i, m, e1, e2, e3 in self.MODES:
            ttk.Label(master, text=m).grid(row=i, column=1, sticky=W, pady=5)
    
            ttk.Entry(master, width=5, justify=RIGHT).grid(row=i, column=2)
            ttk.Label(master, text=e1, padding="0 0 5 0").grid(row=i, column=3)
    
            ttk.Entry(master, width=5, justify=RIGHT).grid(row=i, column=4)
            ttk.Label(master, text=e2, padding="0 0 5 0").grid(row=i, column=5)
    
            ttk.Entry(master, width=5, justify=RIGHT).grid(row=i, column=6)
            ttk.Label(master, text=e3, padding="0 0 5 0").grid(row=i, column=7)

class CBand():
    '''
    classdocs
    '''
    

    def __init__(self, master, rownum, colnum):
        '''
        Constructor
        '''
        self.master = master
        self.rownum = rownum
        self.colnum = colnum
             
        self.BANDS = [self.e_h, self.e_k] = [ttk.Entry(self.master, width=7, justify=RIGHT),
                                             ttk.Entry(self.master, width=7, justify=RIGHT)]
        
        self.VALUE = [e1, e2] = [StringVar(), StringVar()]
              
    def ShowEntry(self, mode):
        for i, m in enumerate(self.BANDS):
            if mode == "active" :                
                m.config(state="normal")
        
            elif mode == "disable" : 
                m.config(state="disable")
            
            elif mode == "readonly" : 
                m.config(state="readonly")
            
            m.grid(row=self.rownum+i, column=self.colnum, padx=1, pady=3)
    
    def GetEntry(self):
        for i, m in enumerate(self.BANDS):
            self.VALUE[i] = m.get()
        return float(self.VALUE[0]), float(self.VALUE[1])
    
    def ShowValue(self, v):
        for i, m in enumerate(self.BANDS):
            m.delete(0, END)
            m.insert(0, v[i])
            
    def GetActiveEntryValue(self, type):
        if type == "h" : 
            value = self.e_h.get()
            
        elif type == "k" : 
            value = self.e_k.get()
            
        return value
    
    def ActiveEntry(self, type):
        for m in self.BANDS:
            m.config(state="disable")
        
        if type == "h" : 
            self.e_h.config(state="normal")
            
        elif type == "k" : 
            self.e_k.config(state="normal")
            
        
class CBandLabel(CBand):
    '''
    classdocs
    '''
    

    def __init__(self, master, rownum, colnum):
        '''
        Constructor
        '''
        CBand.__init__(self, master, rownum, colnum)
        
        self.BANDSLABEL = [self.t_h, self.t_k] = [ttk.Label(self.master, text="H", width=4, anchor="center"),
                                                  ttk.Label(self.master, text="K", width=4, anchor="center")]
              
    def ShowEntry(self, mode):
        CBand.ShowEntry(self, mode)
        
        for i, n in enumerate(self.BANDSLABEL):
            if mode == "active" :                
                n.config(state="active")
        
            elif mode == "disable" : 
                n.config(state="disable")
            
            else :
                pass
                
            n.grid(row=self.rownum+i, column=self.colnum-1, padx=1)
    
    def GetEntry(self):
        v1, v2 = CBand.GetEntry(self)
        return v1, v2
    
    def ShowValue(self, v):
        CBand.ShowValue(self, v)
        return
    
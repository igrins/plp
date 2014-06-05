'''
Created on July 24, 2012

@author: bykwon
'''
from Tkinter        import *
import ttk

from PL_Frame      import *

if __name__ == '__main__':
    root = Tk()
    CFrame(master=root, path="").mainloop()
    
#else :    
#    def Run():    
#        root = Toplevel()
#        CFrame(master=root, path="../Packages/PL/").winfo_toplevel()
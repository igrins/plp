#2013-12-20 Chae Kyung SIM (cksim@khu.ac.kr)
def H_series(seriesname=None,levels=20,inf=False,verbose=False):

    R = 1.0973731568539e1 #Rydberg const. [1/micron], 1.097373e7[1/m]
    seriesfullnames = {1:'Lyman',2:'Balmer',3:'Paschen',4:'Brackett',5:'Pfund',6:'Humphreys'}
    seriesabbrnames = {'Ly':1,'Ba':2,'Pa':3,'Br':4,'Pf':5,'Hu':6}
    
    Brwaves = []
    transnames = []
    if not seriesname:
        n_lows = seriesfullnames.keys()
    else:  
        n_lows = [seriesabbrnames[seriesname[:2]]]
#         if seriesname.startswith('Ly'): n_lows=[1]
#         if seriesname.startswith('Ba'): n_lows=[2]
#         if seriesname.startswith('Pa'): n_lows=[3]
#         if seriesname.startswith('Br'): n_lows=[4]
#         if seriesname.startswith('Pf'): n_lows=[5]
#         if seriesname.startswith('Hu'): n_lows=[6]
#         # if seriesname.startswith('Fu'): n_low=7
    
    for n_low in n_lows:
        if verbose : print seriesfullnames[n_low]
        for n_up in range(n_low+1,levels+1):
            Brwave = 1./(R*(1./(n_low**2.)-1./(n_up**2.)))        
            if verbose == True: print n_up, Brwave
            Brwaves.append(Brwave)        
       
        if inf == True: #when n_up = inf., 1/inf = 0.
            Brwave = 1./(R*(1./(n_low**2.)-0.))
            Brwaves.append(Brwave)     
            if verbose == True: print 'inf.', Brwave    
            
    return Brwaves


'''     
if __name__ == '__main__':
    import math
    import numpy as np
    import basic as ip
    import matplotlib.pyplot as plt 
    
    H_series('Paschen',4,verbose=True)
    H_series('Brackett',inf=True,verbose=True)    
    
    waveum = np.arange(1.475440,1.820510,0.0001) #H-band
    #waveum = np.arange(1.944730,2.489240,0.0001) #K-band
    
    peak = 0.3
    FWHM = 0.004
    sigma = FWHM / (2.*math.sqrt(2*math.log(2)))
    H_lines = [item for item in H_series(verbose=True) if ((item >= waveum[0]-FWHM*2) & (item <= waveum[-1]+FWHM*2))]

    for H_line in H_lines:
        if H_line >= min(waveum) and H_line <= max(waveum):
            try: Br += ip.gauss(waveum,peak,H_line,sigma)
            except: Br = ip.gauss(waveum,peak,H_line,sigma)
        
    Br = 1 - Br
    
    plt.plot(waveum,Br)
    plt.show()    
        
    pass
    
'''  
    
    
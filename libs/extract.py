'''
 wskang 
 2012-07-05
 [description]
  Aperture extracting routines   
  1. Aperture extraction 
  2. Transformation by wavelength data   
  
 [modules]
  - numpy, scipy 
  - matplotlib
  - imageprocess.py
  - mapping.py  
'''

import time, os
import itertools 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata 

import basic as ip 

# polynomial fitting order
PDEGREE = [5,3]

# define the temporary directory 
if os.sys.platform.startswith('linux'):
    TDIR = '/home/igrins/Temp/'
elif os.sys.platform.startswith('win'): 
    TDIR = 'D:/Dropbox/'
else:
    TDIR = '/Volumes/data/Gdrive/IGRINS/wskang/'
#print TDIR

# define the pathes
FITTING_PATH = 'fitdata/'
ONESTEP_PATH = '1step/'
TWOSTEP_PATH = '2step/'
PNG_PATH = './pngs/'
IMAGE_PATH = 'images/'
MANUAL_PATH = 'manual/'

def extract_strips(filename, band, apnum=[], pdeg=PDEGREE, \
                   PA=0, offset=[1023.5,1023.5], pscale=0.018, \
                   slit_len=[-1,1], slit_step=0.025, wave_step=0.00001, \
                   fitting_path=FITTING_PATH, \
                   target_path=ONESTEP_PATH):
    '''
    Extract the strips directly based on ZEMAX analysis fitting data 
     (using mapping parameters like position angle, pixel scale, ... 
     - input :  for each band 
        1. FITTING DATA (fitting_path) 
        2. MAPPING DATA (PA,offset,pscale) 
        
    '''
    fpath, fname = ip.split_path(filename)
    if ip.exist_path(target_path) == False: target_path = fpath
    # extract the file name only without extension
    name = '.'.join(fname.split('.')[:-1])
    
    img, hdr = ip.readfits(filename)
    # read order information from file 
    onum, odesc, owv1, owv2 = ip.read_orderinfo(band)
    
    if len(apnum) == 0:
        apnum = range(len(onum))
    
    # read image size 
    ny, nx = img.shape 
    
    #==============================================================================
    # Extract strips based on ZEMAX fitting data 
    #==============================================================================     
    descs = []     
    strips = [] 
    wavelengths = [] 
    
    for k in apnum:
        desc, wv1, wv2 = (odesc[k], owv1[k], owv2[k])
        print "order # = %s, wrr = [%f, %f]" % (desc, wv1, wv2)
        # read the echellogram fitting data 
        mx = np.loadtxt(FITTING_PATH+'mx_%s_%02d_%02d.dat' % (desc, pdeg[0], pdeg[1]))
        my = np.loadtxt(FITTING_PATH+'my_%s_%02d_%02d.dat' % (desc, pdeg[0], pdeg[1]))
        
        # make X dimension array (for wavelength) 
        twave = np.arange(wv1, wv2, wave_step, dtype=np.float64)
        n_wave = len(twave)
        # make Y dimension array (for slit)
        tslit = np.arange(slit_len[0],slit_len[1]+slit_step, slit_step, dtype=np.float64)
        n_slit = len(tslit)
        # make 2D array for wavelength, slit-positions 
        swave = np.zeros([n_slit,n_wave], dtype=np.float64)
        sslit = np.zeros([n_slit,n_wave], dtype=np.float64)
        for i in range(n_wave):
            sslit[:,i] = tslit
        for i in range(n_slit):
            swave[i,:] = twave
        # find X, Y positions for each wavelength and slit-position
        sx = ip.polyval2d(swave, sslit, mx, deg=pdeg)
        sy = ip.polyval2d(swave, sslit, my, deg=pdeg)
        # transform into pixel units 
        px, py = ip.xy2pix(sx, sy, PA=PA, offset=offset, pscale=pscale)
        # check image range 0 < x < 2048
        xmin, xmax = (0,n_wave) 
        for i in range(n_slit):
            vv = np.where((px[i,:] >= 0) & (px[i,:] < nx))[0]
            if np.min(vv) > xmin: xmin = np.min(vv)
            if np.max(vv) < xmax: xmax = np.max(vv)
        
        # extract the aperture by using interpolation from image
        tstrip = ip.imextract(img, px[:,xmin:xmax], py[:,xmin:xmax])
        twave = twave[xmin:xmax]
        print ' + Wavelength valid range = [%f, %f]' % (twave[0], twave[-1])
        
        descs.append(desc)
        wavelengths.append(twave)
        strips.append(tstrip)

    #==============================================================================
    # Save the strips in FITS format
    #==============================================================================    
    for d, w, s in zip(descs, wavelengths, strips):
        shdr = header.copy()
        
        shdr.update('GEN-TIME', time.strftime('%Y-%m-%d %H:%M:%S'))
        shdr.update('LNAME', lname)
        shdr.update('ECH-ORD', d)
        # WCS header ========================================
        shdr.update('WAT0_001', 'system=world')
        shdr.update('WAT1_001', 'wtype=linear label=Wavelength units=microns units_display=microns')
        shdr.update('WAT2_001', 'wtype=linear')
        shdr.update('WCSDIM', 2)
        shdr.update('DISPAXIS', 1)
        shdr.update('DC-FLAG', 0)
        
        # wavelength axis header ============================= 
        shdr.update('CTYPE1', 'LINEAR  ')
        shdr.update('LTV1',   1)
        shdr.update('LTM1_1', 1.0)
        shdr.update('CRPIX1', 1.0)
          #header.update('CDELT1', w[1]-w[0])
        shdr.update('CRVAL1', w[0])
        shdr.update('CD1_1',  w[1]-w[0])
        
        # slit-position axis header ==========================
        shdr.update('CTYPE2', 'LINEAR  ')
        shdr.update('LTV2',   1)
        shdr.update('LTM2_2', 1.0)
        shdr.update('CRPIX2', 1.0)
          #header.update('CDELT1', w[1]-w[0])
        shdr.update('CRVAL2', -1.0)
        shdr.update('CD2_2',  slit_step)
        
        # save FITS with header 
        ip.savefits(EXTRACT_PATH+'IGRINS_%s_%s.fits' % (lname,d), s, header=shdr)
        np.savetxt(EXTRACT_PATH+'IGRINS_%s_%s.wave' % (lname,d), w)
        

def extract_ap_file(imgfile, apfiles, wlfiles=None, ap_width=60, \
                    target_path=None):
    '''
    Extract the apertures using ap_tracing solution and wavelength data 
    - inputs:
     1. imgfile (input FITS) 
     2. apfiles (a list of input ap_tracing files) 
     3. wlfiles (a list of wavelength fitting coefficient files)
     4. ap_width (to be extracted with this pixel width) 
     5. target_path (output directory)  
    
    '''
    if len(apfiles) != len(wlfiles): wlfiles = None 
    
    img, header = ip.readfits(imgfile)
    fpath, fname = ip.split_path(imgfile)
    if ip.exist_path(target_path) == False: target_path = fpath
    # extract the file name only without extension
    name = '.'.join(fname.split('.')[:-1])
    
    ostrips, owaves, ohdrs = \
       extract_ap(img, apfiles, wlfiles=wlfiles, \
                  header=header, ap_width=ap_width, target_path=target_path )
       
    for strip, wave, hdr in zip(ostrips, owaves, ohdrs):
        ap_num = hdr.get('AP-NUM')
        ip.savefits(target_path+name+'.%03d.fits' % (ap_num,), strip, header=hdr)
        
    
def extract_ap(img, apfiles, wlfiles=None, header=None, ap_width=60):
    '''
    Extract the apertures using ap_tracing solution and wavelength data files
     - INPUTS :  
      1. img (input 2D image data) 
      2. apfiles (a list of input ap_tracing files) 
      3. wlfiles (a list of wavelength fitting coefficient files)
      4. ap_width (to be extracted with this pixel width) 
     - OUTPUTS :
      1. strips (output 2D strip image data)
      2. waves (output 2D wavelength data for the strip) 
      3. hdrs (output FITS header for 2D strip image)   
    '''
    # read image size 
    ny, nx = img.shape 
    
    #ap_width=60 #just for test on 2013-08-30
    
    ostrips, owaves, ohdrs = [], [], [] 
    for k, apfile in enumerate(apfiles): #
        #===========================================================
        # extract the strip with a regular width 
        #===========================================================
        ap_coeff = np.loadtxt(apfile)
        # extract the aperture number from file name 
        ap_num = int(apfile.split('.')[-2])
        # read the wavelength fitting coefficients 
        
        if wlfiles != None:
            wl_coeff = np.loadtxt(wlfiles[k])
            wtmp = wlfiles[k].split('.')[-3].split('_')
            #print wtmp
            # find the dimension of 2d fitting coefficients
            xdim, ydim = int(wtmp[2])+1, int(wtmp[3])+1
            # find the ap_tracing mode 
            ap_mode = wtmp[0]
        else:
            ap_mode = '' 
            wl_coeff = None
        
        # calculate the x, y positions of aperture bottom curve
        xpos = np.arange(nx)
        
        
        sparepixels = 5
        
        #if ap_num >= 0 and ap_num <=8 : sparepixels = 20
        #elif ap_num >= 9 and ap_num <=18 : sparepixels = 7
        #else : sparepixels = 5

        #sparepixels = 0 #just for test (to fix)
        # bottom curve 
        #ypos = np.polynomial.polyval(xpos, ap_coeff[0,:])  #for numpy version lower than 1.7.0
        ypos = np.polynomial.polynomial.polyval(xpos, ap_coeff[0,:]) -sparepixels #for numpy version higher than or equal to 1.7.0
        #bottomcurve, = plt.plot(range(0,nx),ypos)
        # top curve 
        #ypos_top = np.polynomial.polyval(xpos, ap_coeff[1,:])#for numpy version lower than 1.7.0
        ypos_top = np.polynomial.polynomial.polyval(xpos, ap_coeff[1,:]) + sparepixels#for numpy version higher than or equal to 1.7.0        
        # make 2D strip array 
       
        #ap_width = np.average(ypos_top - ypos)
        if k>=0 and k<=9:
            ap_width = 70
        elif k>9 and k<=17:
            ap_width = 65
        else:
            ap_width = 63
               
        strip = np.zeros([ap_width,nx], dtype=np.double)
        # generate the coordinate array for strip  
        yy, xx = np.indices(strip.shape)
        # add the y-direction aperture curve to the y-coordinates of strip 
        ap_yy = np.array(yy, dtype=np.double) + ypos
        ap_xx = np.array(xx, dtype=np.double)
        # convert positions into integer indices of aperture positions
        #iyy = np.array(np.floor(ap_yy), dtype=np.int) #we use "round" 2013-09-03 meeting with SPak & Huynh Anh
        iyy = np.array(np.round(ap_yy), dtype=np.int)
        # calculate the fraction subtracted by integer pixel index 
        fyy = ap_yy - iyy 
        # find the valid points
        vv = np.where((iyy >= 0) & (iyy <= ny-2))
        # input the value into the strip
        #we use "round" 2013-09-03 meeting with SPak & Huynh Anh
        #and we don't scale or interpolate pixel intensities 2013-09-03 meeting with SPak & Huynh Anh
        #strip[yy[vv],xx[vv]] = \
        #   img[iyy[vv],xx[vv]] * (1.0 - fyy[yy[vv],xx[vv]]) + \
        #   img[(iyy[vv]+1),xx[vv]] * fyy[yy[vv],xx[vv]]
        strip[yy[vv],xx[vv]] = img[iyy[vv],xx[vv]]
        
        #===========================================================
        # update the FITS header
        #===========================================================
        if header != None:
            hdr = header.copy()
            #hdr.update('AP-MODE',ap_mode)
            hdr.update('AP-NUM', ap_num)
            c1list = []
            for c in ap_coeff[0,:]:
                c1list.append('%.8E' % (c,))
            hdr.update('AP-COEF1', ','.join(c1list))
            c2list = []
            for c in ap_coeff[1,:]:
                c2list.append('%.8E' % (c,))
            hdr.update('AP-COEF2', ','.join(c2list))
            hdr.update('AP-WIDTH', ap_width)
            
            # input the wavelength solution for a strip 
            
            if wl_coeff != None:
                hdr.update('WV-DIM', '%d,%d' % (xdim, ydim))
                for i in range(xdim):
                    wllist = []
                    for c in wl_coeff[(i*(ydim)):(i*ydim+ydim)]:
                        wllist.append('%.8E' % (c,))
                    hdr.update('WV-X%03d' % (i,), ','.join(wllist))
                wave = ip.polyval2d(xx, yy, wl_coeff, deg=[(xdim-1), (ydim-1)])
        else:
            wave = None
            hdr = None 
            
        ostrips.append(strip)
        owaves.append(wave)
        ohdrs.append(hdr)
        
    return ostrips, owaves, ohdrs  

def extract_ap_custom(img, apfiles, wlfiles=None, header=None, ap_width=60):
    '''
    Extract the apertures using ap_tracing solution and wavelength data files
     - INPUTS :  
      1. img (input 2D image data) 
      2. apfiles (a list of input ap_tracing files) 
      3. wlfiles (a list of wavelength fitting coefficient files)
      4. ap_width (to be extracted with this pixel width) 
     - OUTPUTS :
      1. strips (output 2D strip image data)
      2. waves (output 2D wavelength data for the strip) 
      3. hdrs (output FITS header for 2D strip image)   
    '''
    # read image size 
    ny, nx = img.shape 
    #ap_width=60 #just for text on 2013-08-30
    
    ostrips, owaves, ohdrs = [], [], [] 
    for k, apfile in enumerate(apfiles):
        #===========================================================
        # extract the strip with a regular width 
        #===========================================================
        ap_coeff = np.loadtxt(apfile)
        # extract the aperture number from file name 
        ap_num = int(apfile.split('.')[-2])
        
        '''
        # read the wavelength fitting coefficients 
        if wlfiles != None:
            wl_coeff = np.loadtxt(wlfiles[k])
            wtmp = wlfiles[k].split('.')[-3].split('_')
            #print wtmp
            # find the dimension of 2d fitting coefficients
            xdim, ydim = int(wtmp[2])+1, int(wtmp[3])+1
            # find the ap_tracing mode 
            ap_mode = wtmp[0]
        else:
            ap_mode = '' 
            wl_coeff = None
        '''
        # calculate the x, y positions of aperture bottom curve
        xpos = np.arange(nx)
        
        sparepixels = 5
        
        #if ap_num >= 0 and ap_num <=8 : sparepixels = 20
        #elif ap_num >= 9 and ap_num <=18 : sparepixels = 7
        #else : sparepixels = 5

        #sparepixels = 0 #just for test (to fix), should be used this sparepixels = 0 due to the wavelength calibration process of checking distortion correction 
        # bottom curve 
        #ypos = np.polynomial.polyval(xpos, ap_coeff[0,:]) #for numpy version lower than 1.7.0
        ypos = np.polynomial.polynomial.polyval(xpos, ap_coeff[0,:]) - sparepixels \
               #for numpy version higher than or equal to 1.7.0
        #bottomcurve, = plt.plot(range(0,nx),ypos)
        # top curve 
        #ypos_top = np.polynomial.polyval(xpos, ap_coeff[1,:])#for numpy version lower than 1.7.0
        ypos_top = np.polynomial.polynomial.polyval(xpos, ap_coeff[1,:]) + sparepixels \
                   #for numpy version higher than or equal to 1.7.0        
        # make 2D strip array 
        #ap_width = np.average(ypos_top - ypos)  # aperture with should be as a constant value of ap_width = 60.
        
        if k>=0 and k<=9:
            ap_width = 70
        elif k>9 and k<=17:
            ap_width = 65
        else:
            ap_width = 63
        
        strip = np.zeros([ap_width,nx], dtype=np.double)
        # generate the coordinate array for strip  
        yy, xx = np.indices(strip.shape)
        # add the y-direction aperture curve to the y-coordinates of strip 
        ap_yy = np.array(yy, dtype=np.double) + ypos
        ap_xx = np.array(xx, dtype=np.double)
        # convert positions into integer indices of aperture positions
        #iyy = np.array(np.floor(ap_yy), dtype=np.int) #we use "round" 2013-09-03 meeting with SPak & Huynh Anh
        iyy = np.array(np.round(ap_yy), dtype=np.int)
        # calculate the fraction subtracted by integer pixel index 
        fyy = ap_yy - iyy 
        # find the valid points
        vv = np.where((iyy >= 0) & (iyy <= ny-2))
        # input the value into the strip
        #we use "round" 2013-09-03 meeting with SPak & Huynh Anh
        #and we don't scale or interpolate pixel intensities 2013-09-03 meeting with SPak & Huynh Anh
        #strip[yy[vv],xx[vv]] = \
        #   img[iyy[vv],xx[vv]] * (1.0 - fyy[yy[vv],xx[vv]]) + \
        #   img[(iyy[vv]+1),xx[vv]] * fyy[yy[vv],xx[vv]]
        strip[yy[vv],xx[vv]] = img[iyy[vv],xx[vv]]
        
        #===========================================================
        # update the FITS header
        #===========================================================
        
        if header != None:
            hdr = header.copy()
            #hdr.update('AP-MODE',ap_mode)
            hdr.update('AP-NUM', ap_num)
            c1list = []
            for c in ap_coeff[0,:]:
                c1list.append('%.8E' % (c,))
            hdr.update('AP-COEF1', ','.join(c1list))
            c2list = []
            for c in ap_coeff[1,:]:
                c2list.append('%.8E' % (c,))
            hdr.update('AP-COEF2', ','.join(c2list))
            hdr.update('AP-WIDTH', ap_width)
            '''
            # input the wavelength solution for a strip 
            if wl_coeff != None:
                hdr.update('WV-DIM', '%d,%d' % (xdim, ydim))
                for i in range(xdim):
                    wllist = []
                    for c in wl_coeff[(i*(ydim)):(i*ydim+ydim)]:
                        wllist.append('%.8E' % (c,))
                    hdr.update('WV-X%03d' % (i,), ','.join(wllist))
                wave = ip.polyval2d(xx, yy, wl_coeff, deg=[(xdim-1), (ydim-1)])
        else:
            wave = None
            hdr = None 
             '''
        ostrips.append(strip)
        #owaves.append(wave)
        ohdrs.append(hdr)
        
    return ostrips, ohdrs


def line_identify(stripfile, linedata=[], outputfile=None, \
                  npoints=30, spix=5, dpix=5, thres=15000):
    '''
    Identify the lines in the strip based on the line database
    - inputs : 
     1. stripfile (with wavelength data or not) 
     2. linefile (for line database) 
    '''
    
    def key_press(event): 
        
        ax = event.inaxes 
        if ax == None: return 
        ax_title = ax.get_title()
        
        if event.key == 'q': plt.close('all')

        if event.key == 'm':
            click_x, click_y = event.xdata, event.ydata
            rr = np.arange((x2-2*dpix),(x2+2*dpix+1), dtype=np.int)
            #p = ip.gauss_fit(rr, row[rr], p0=[1, x2, 1])
            #a2.plot(rr, ip.gauss(rr, *p), 'r--')
            #mx = p[1]
            mx, mval = ip.find_features(rr, row[rr], gpix=dpix)
            mx, mval = mx[0], mval[0]
            a2.plot(mx, mval, 'ro')
            fig.canvas.draw()
            fwv = cwv[np.argmin(np.abs(cwv-wav[mx]))]
            strtmp = raw_input('input wavelength at (%d, %d) = %.7f:' % (mx, my, fwv))
            if strtmp == '': strtmp = fwv
            try:
                mwv = np.float(strtmp)
                lxx.append(mx)
                lyy.append(my)
                lwv.append(mwv)
                a1.plot(mx, my, 'ro')
                fig.canvas.draw()
                print '%.7f at (%d, %d)' % (mwv, mx, my)
            except:
                print 'No input, again...'
                                   
    
    if len(linedata) == 2: 
        cwv, cflx  = linedata
    else:
        print 'No input data for line information'
        cwv, cflx = None, None 
        
    strip, hdr = ip.readfits(stripfile)
    fpath, fname = ip.split_path(stripfile)
    if outputfile == None: outputfile = fpath+'c'+fname
    # extract the file name only without extension
    name = '.'.join(fname.split('.')[:-1])

    ny, nx = strip.shape
    yy, xx = np.indices(strip.shape)
    
    if 'WV-DIM' in hdr.keys(): 
        xdim, ydim = np.array(hdr.get('WV-DIM').split(','), dtype=np.int)
        wl_coeff = np.zeros([xdim*ydim]) 
        for i in range(xdim):
            tmp = hdr.get('WV-X%03d' % (i,))
            wl_coeff[(i*ydim):(i*ydim+ydim)] = np.array(tmp.split(','), dtype=np.double)
        awave = ip.polyval2d(xx, yy, wl_coeff, deg=[xdim-1, ydim-1])
    else:
        print 'No wavelength data in FITS header'
        awave = None 
        
    
    # define figure object 
    fig = plt.figure(figsize=(14,7))
    a1 = fig.add_subplot(211, title='strip')
    a2 = fig.add_subplot(212, title='line')
    
    # draw the strip image
    z1, z2 = ip.zscale(strip)         
    a1.imshow(strip, cmap='gray',vmin=z1, vmax=z2, aspect='auto')
    a1.set_xlim(0,nx)
    a1.set_ylim(-10,ny+10)
            
    lspec = np.sum(strip, axis=0)
    lxpos = np.arange(nx)
    a2.plot(lxpos, lspec)
    a2.set_xlim(0,nx)
    # draw the lines in the database  
    if (awave != None) & (cwv != None):
        twave = awave[int(ny/2),:]
        print twave
        wv1, wv2 = np.min(twave), np.max(twave)
        for wv0 in cwv[(cwv > wv1) & (cwv < wv2)]:
            x0 = np.argmin(np.abs(twave-wv0))
            a2.plot([x0, x0], [0, ny], 'r--', linewidth=1, alpha=0.4)
  
    # make the array for emission feature points 
    lxx, lyy, lwave = ([], [], [])
    fig.canvas.mpl_connect('key_press_event', key_press)
    print '[m] mark the line and input the wavelength'
    print '[q] quit from this aperture'
    
    plt.show()
                
    return 
    

#def transform_ap_file(stripfile, wave_step=0.00001, outputfile=None): #2014-01-13 cksim
def transform_ap_file(stripfile, wave_step=False, outputfile=None): #2014-01-13 cksim
    '''
    Apply linear interpolation to the strip with a regular wavelength
    - INPUTS :
     1. stripfile (FITS)
     2. wave_step (to be extracted with this wavelength step for a pixel)
     3. outputfile (FITS filename) 
    - OUTPUTS: 
     1. tranformed 2D strip data with header   
    '''
    
    astrip, ahdr = ip.readfits(stripfile)
    fpath, fname = ip.split_path(stripfile)
    if outputfile == None: outputfile = fpath+fname+'.tr'
    # extract the file name only without extension
    name = '.'.join(fname.split('.')[:-1])
    ny, nx = astrip.shape
    yy, xx = np.indices(astrip.shape)
    
    if 'WV-DIM' in ahdr.keys(): 
        xdim, ydim = np.array(ahdr.get('WV-DIM').split(','), dtype=np.int)
        wl_coeff = np.zeros([xdim*ydim]) 
        for i in range(xdim):
            tmp = ahdr.get('WV-X%03d' % (i,))
            wl_coeff[(i*ydim):(i*ydim+ydim)] = np.array(tmp.split(','), dtype=np.double)
        awave = ip.polyval2d(xx, yy, wl_coeff, deg=[xdim-1, ydim-1])
        
    else:
        print 'No wavelength data in FITS header'
        return None, None 
    
    
    if wave_step == False: wave_step = ( (np.max(awave) - np.min(awave)) ) / (nx-1) #2013-01-14 cksim
    #--new version--       
    tstrip, xwave = transform_ap(astrip, awave, wave_step=wave_step)

    wv1, wv2 = np.min(xwave), np.max(xwave)
   
    '''old version
    wv1, wv2 = (np.min(awave), np.max(awave))
    xwave = np.arange(wv1, wv2, wave_step)
    nwave = len(xwave)
    #print nx, ny, nwave, np.min(awave), np.max(awave)
    tstrip = np.zeros([ny,nwave])
    
    for i in range(ny):
        row = astrip[i,:]
        wv = awave[i,:]
        xrow = np.interp(xwave, wv, row)
        tstrip[i,:] = xrow
    '''

    thdr = ahdr.copy() 
    thdr.update('TRN-TIME', time.strftime('%Y-%m-%d %H:%M:%S'))
    # WCS header ========================================
    thdr.update('WAT0_001', 'system=world')
    thdr.update('WAT1_001', 'wtype=linear label=Wavelength units=microns units_display=microns')
    thdr.update('WAT2_001', 'wtype=linear')
    thdr.update('WCSDIM', 2)
    thdr.update('DISPAXIS', 1)
    thdr.update('DC-FLAG', 0)
    
    # wavelength axis header ============================= 
    thdr.update('CTYPE1', 'LINEAR  ')
    thdr.update('LTV1',   1)
    thdr.update('LTM1_1', 1.0)
    thdr.update('CRPIX1', 1.0)
    thdr.update('CDELT1', wave_step)
    thdr.update('CRVAL1', wv1)
    thdr.update('CD1_1',  wave_step)
    
    # slit-position axis header ==========================
    thdr.update('CTYPE2', 'LINEAR  ')
    thdr.update('LTV2',   1)
    thdr.update('LTM2_2', 1.0)
    thdr.update('CRPIX2', 1)
    thdr.update('CRVAL2', 1)
    thdr.update('CD2_2',  1)

    ip.savefits(outputfile, tstrip, header=thdr)
    np.savetxt('.'.join(outputfile.split('.')[:-1])+'.wave', xwave)
    
    '''plt.subplot(211)
    plt.imshow(astrip, aspect=2)
    plt.xlim(0,nx)
    plt.subplot(212)
    plt.imshow(tstrip, aspect=2) 
    plt.xlim(0,nwave)
    plt.show() 
    '''
    ##2013-11-21 cksim inserted below draw_strips_file()
    #draw_strips_file('.'.join(outputfile.split('.')[:-1])+'.fits', '.'.join(outputfile.split('.')[:-1])+'.wave', linefile='ohlines.dat', \
    #    target_path=outputfile.split('SDC')[0], desc='SDC'+outputfile.split('SDC')[1].split('.fits')[0])
    
    return tstrip, thdr 

def transform_ap(astrip, awave, wave_step=0.00001): 
    '''
    Apply linear interpolation to the strip with a regular wavelength
    - INPUTS :
     1. strip data (2D array)
     2. wavelength data (2D array)  
     3. wave_step (to be extracted with this wavelength step for a pixel)
    - OUTPUTS : 
     1. transformed strip data (2D array)
     2. transformed wavelength data (2D array)    
    '''
    
    ny, nx = astrip.shape
        
    wv1, wv2 = (np.min(awave), np.max(awave))
    xwave = np.arange(wv1, wv2, wave_step)
    nwave = len(xwave)
    #print nx, ny, nwave, np.min(awave), np.max(awave)
    tstrip = np.zeros([ny,nwave])
    
    for i in range(ny):
        row = astrip[i,:]
        wv = awave[i,:]
        xrow = np.interp(xwave, wv, row)
        tstrip[i,:] = xrow
        
    return tstrip, xwave  

def draw_strips_file(stripfile, wavefile, linefile='ohlines.dat', \
                     desc='', target_path=PNG_PATH):
    '''

    Draw the strips with wavelength 
     - INPUTS:
      1. strip file name (.fits) 
      2. wavelength file name (.wave)
      3. line data file  
     - OUTPUTS:
      PNG files      
    '''
    strip, hdr = ip.readfits(stripfile)
    wave = np.loadtxt(wavefile)
    cwv, cflx = ip.read_lines(linefile)
    linedata = (cwv, cflx)
    
    draw_strips(strip, wave, linedata=linedata, \
                desc=desc, lampname=linefile[:-9], target_path=target_path)   

    
def draw_strips(strip, wave, desc='', linedata=[], lampname='', grayplot=False, \
                target_path=PNG_PATH):
    '''
    Draw the strips with wavelength 
     - INPUTS:
      1. strip data 
      2. wavelength data
      3. line data 
     - OUTPUTS:
      PNG files      
    '''
    #print 'draw_strips', lampname
    if len(linedata) > 0: 
        cwv, cflx = linedata
    wmin, wmax = (np.min(wave), np.max(wave))
    
    z1, z2 = ip.zscale(strip)
    
    f2 = plt.figure(2,figsize=(12,5),dpi=200)
    a1 = f2.add_subplot(211, aspect='auto')
    a2 = f2.add_subplot(212, aspect='auto')
    
    ny, nx = strip.shape
    cuty = ny/2-8
    a1.imshow(strip, cmap='hot', aspect=nx/1000.0, vmin=z1, vmax=z2)
    a1.plot([0,nx],[cuty,cuty], 'g--', alpha=0.8, linewidth=3)
    a1.set_xlim((0,nx))
    a1.set_ylim((0,ny))
    a1.set_title('%s' % (desc,))
    
    a2.plot(wave, strip[cuty,:], 'b-', linewidth=1)
    ww = np.where((cwv < wmax) & (cwv > wmin))[0]
    fmax = np.zeros([len(ww)]) + np.max(strip[cuty,:])*1.05
    a2.plot(cwv[ww], fmax, 'r|', markersize=10, markeredgewidth=2, alpha=0.7 )
    if lampname != '': lampname = '('+lampname+')'
    a2.set_title('%s %s' % (desc,lampname))
    a2.set_xlabel('Wavelength [um]')
    a2.set_xlim((wmin, wmax))
    a2.set_ylim((0, fmax[0]*1.1))
#    f2.savefig(target_path+'%s_prof.png' % (desc,)) #2013-11-21 cksim commented this
    f2.savefig(target_path+'%s.png' % (desc,)) #2013-11-21 cksim
    
    if grayplot == True:
        f3 = plt.figure(3, figsize=(12,5),dpi=200)
        a3 = f3.add_subplot(111)
        
        a3.imshow(strip, cmap='gray', aspect=nx/200.0, vmin=z1, vmax=z2)
        x, y = (np.arange(nx), np.sum(strip, axis=0))
        a3.plot(x, (y-np.min(y))/(np.max(y)-np.min(y))*ny*0.9,\
                'b-', linewidth=3, alpha=0.7)
        x, y = ( (cwv[ww]-wmin)/(wmax-wmin)*(nx-1), np.zeros(len(ww))+ny*1.05) 
        a3.plot(x, y, 'r|', markersize=10, markeredgewidth=2, alpha=0.7)
        a3.set_xlim((0,nx))
        a3.set_ylim((0,ny*1.1))
        cticks = wticks[np.where((wticks > wmin) & (wticks < wmax))]
        xticks = (cticks-wmin)/(wmax-wmin)*(nx-1)
        #print xticks
        a3.set_yticks([])
        a3.set_xticks(xticks)
        a3.set_xticklabels(['%.3f' % x for x in cticks])
        a3.set_xlabel('Wavelength [um]')
        f3.savefig(target_path+'%s_gray.png' % (desc,))
    
    plt.close('all')    
    
   
def test1(band, lname='ohlines'):
    '''
    Extract the strip spectra and save them
    '''
    slit_step = 0.03
    slit_len = 2.0 
    timg, theader = ip.readfits(IMAGE_PATH+'IGRINS_%s_%s.fits' % (band,lname))
    extract_strips(timg, theader, band, lname, \
                slit_len=slit_len, slit_step=slit_step)

def test2(band, lname='ohlines'): 
          
    '''
    Draw the original image   
    '''
    timg, thdr = ip.readfits(IMAGE_PATH+'IGRINS_%s_%s.fits' % (band,lname))
    z1, z2 = ip.zscale(timg, contrast=0.2)
    
    f1 = plt.figure(1,figsize=(12,12),dpi=200)
    a1 = f1.add_subplot(111, aspect='equal')
    a1.imshow(timg, cmap='hot', vmin=z1, vmax=z2)
    a1.set_xlim((0,2048))
    a1.set_ylim((0,2048))
    a1.set_title('IGRINS %s band - %s (simulated)' % (band,lname))
    a1.set_xlabel('X [pixel]')
    a1.set_ylabel('Y [pixel]')
    f1.savefig(PNG_PATH+'IGRINS_%s_%s.png' % (band,lname))
    plt.close('all')

def test3(band, lname='ohlines'):
    '''
    Draw the extracted strip images with wavelength  
    '''    
    onum, odesc, owv1, owv2 = ip.read_orderinfo(band)
    cwv, cflx = ip.read_lines(lname+'.dat')
    
    wticks = np.arange(1.5, 2.5, 0.005)
    
    for desc in odesc:
        strdesc = 'IGRINS_%s_%s_%s' % (band, lname, desc)
        
        strip, shdr = ip.readfits(ONESTEP_PATH+'IGRINS_%s_%s.fits' % (desc,lname))
        wave = np.loadtxt(ONESTEP_PATH+'IGRINS_%s_%s.wave' % (desc,lname))
        
        draw_strips(strip, wave, desc=strdesc, linedata=[cwv, cflx])
        
def test4(band, lname='ohlines'):
    
    if band == 'H':
        ap_num = 23
    else: 
        ap_num = 20
    
    cwv, cflx = ip.read_lines(lname+'.dat')
         
    for k in range(ap_num):
        
        strdesc = 'IGRINS_%s_%s.%03d' % (band, lname, k)
        strip, hdr = ip.readfits(MANUAL_PATH+strdesc+'.fits')
        z1, z2 = ip.zscale(strip)   
        f2 = plt.figure(2,figsize=(12,5),dpi=200)
        a1 = f2.add_subplot(211, aspect='auto')
        
        ny, nx = strip.shape
        a1.imshow(strip, cmap='hot', aspect=nx/1000.0, vmin=z1, vmax=z2)
        a1.set_xlim((0,nx))
        a1.set_ylim((0,ny))
        a1.set_title('%s' % (strdesc,))
        
        f2.savefig(PNG_PATH+strdesc+'.png')
        
        strdesc = 'tIGRINS_%s_%s.%03d' % (band, lname, k)
        strip, hdr = ip.readfits(MANUAL_PATH+strdesc+'.fits')
        wave = np.loadtxt(MANUAL_PATH+strdesc+'.wave')
        draw_strips(strip, wave, desc=strdesc, linedata=[cwv, cflx])
        
def test5():
    stripfile=MANUAL_PATH+'IGRINS_H_ohlines.000.fits'
    cwv, cflx = ip.read_lines('ohlines.dat')
    line_identify(stripfile, linedata=(cwv, cflx), thres=12000)
    
        
def test8(band):
    '''
    Test the aperture extraction with a regular width 
    '''
    if band == 'H':
        ap_num = 23
    else:
        ap_num = 20 
    
    t_start = time.time()
    
    img, hdr = ip.readfits(IMAGE_PATH+'IGRINS_%s_ohlines.fits' % (band,))
    
    aplist, wllist, stlist = [], [], [] 
    for i in range(ap_num):
        aplist.append(MANUAL_PATH+'apmap_%s_07.%03d.dat' % (band, i))
        wllist.append(MANUAL_PATH+'apwav_%s_03_02.%03d.dat' % (band, i))
        stlist.append(MANUAL_PATH+'IGRINS_%s_ohlines.%03d.fits' % (band, i))
    
    extract_ap_file(IMAGE_PATH+'IGRINS_%s_ohlines.fits' % (band,), \
               aplist, wllist, target_path=MANUAL_PATH)
    for stripfile in stlist:
        transform_ap_file(stripfile)
    
    #tstrip, thdr = \
    #  trasnform_ap(adescs[0], astrips[0], awaves[0], header=ahdrs[0])
    
    
    t_end = time.time()     
               
    print t_end-t_start
                
if __name__ == '__main__':
    #test5()
    #test3('H')
    #test4('K')
    line_identify('../test_ck/flat.H.002.fits')
    #test8('H')
    #test8('K')
    
    
    pass

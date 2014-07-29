'''
 wskang 
 2012-07-05
 [description]
  Manual aperture extraction & wavelength calibration   
    1. manual aperture tracing by 
      + stellar spectrum (peak) 
      + flat spectrum (strip)
      + generate the "aperture tracing data" of strip 
    2. line identification by 
      + comparison lamp or sky emission spectrum 
      + generate the "wavelength data" of strip 
  
 [modules]
  - numpy, scipy 
  - matplotlib
  - pyfits 
   
  - (polyfit2d.py) 
  - imageprocess.py 
'''
import time, os
import numpy as np 

import basic as ip 
#import mapping 

import matplotlib.pyplot as plt 


#band = 'H'
#filename = '140227_H_Flat_no_pinhole_1s_rotate.fits'

strdate = time.strftime('%Y%m%d')
'''
if os.sys.platform.startswith('linux'):
    TDIR = '/home/wskang/Temp/'
elif os.sys.platform.startswith('win'): 
    TDIR = 'D:/Temp/'
'''
'''
TDIR = './'


# define the global parameters and pathes 
FITTING_PATH = TDIR + 'IGRINS/fittingdata/'
IMAGE_PATH = TDIR + 'IGRINS/images/'
'''
MANUAL_PATH = './custom/'

#PDEGREE = [5,3]

def peak_find(icol, thres=15000, dpix=5, mode=''):
    '''
    Find the positions of apertures for each column 
     - INPUTS: 
      1. 1D column data array 
      2. threshold level 
      3. dpix (pixel width to distinguish different feature)  
    '''
    #icol = np.mean(img[:,(xpos-width):(xpos+width)],axis=1)
    ix = np.arange(len(icol))
    caps = ip.find_signif_peaks(icol, thres=thres)
    rcaps = [] 
    if caps != None: 
        # check the nearby positions
        bcol = np.zeros([len(icol)], dtype=np.int)
        bcol[caps] = 1 
        ileft, iright = ip.find_edges_true_regions((bcol == 1), dpix=dpix)
        # combine the nearby positions
        for xleft, xright in zip(ileft, iright):
            # find the maximum position of "icol" within (xleft,xright) 
            x_argmax = xleft + np.argmax(icol[xleft:(xright+1)])
            x1 = np.max([x_argmax - dpix, 0])
            x2 = np.min([x_argmax + dpix, len(icol)-1])+1   
            # calculate the intensity-averaged center
            if mode == 'gauss':
                #plt.ioff()
                #plt.plot(np.arange(x1,x2),icol[x1:x2])
                #plt.show()
                gcoeff = ip.gauss_fit(np.arange(x1,x2), icol[x1:x2], p0=[thres,x_argmax,dpix])
                x_peak = gcoeff[1]
            else:
                x_peak = np.sum(icol[x1:x2]*np.arange(x1,x2))/np.sum(icol[x1:x2])
            rcaps.append(x_peak)
    return rcaps 

def strip_find(band, icol, dpix=50, margin=10, devide=8.0):
    '''
    Find the start and the end position of each strip 
      by using numerical derivatives 
      - input 
        1. column data (1d-array) 
        2. dpix (pixel width to distinguish different aperture)
        3. margin (margin for eliminating the edge of numerical derivatives)
      - output 
        1. a list of starting positions of the apertures
        2. a list of ending positions of the apertures
    '''
    
    if band == "H":
        devide = 5.5
    else:
        devide = 20.0
    
    ny = len(icol)
    print 'len icol', ny
    yy = np.arange(ny)
    dval = ip.nderiv(yy, icol)
    
    # maximun and minimum of numerical derivatives
    vmin, vmax = np.nanmin(dval[margin:(ny-margin)]), np.nanmax(dval[margin:(ny-margin)])
    print 'vmin, vmax', vmin, vmax
    # define the threshold for detecting local maxima
    vthres = (np.abs(vmax) +  np.abs(vmin)) / devide #8.0  # Edit by Huynh Anh
    # find the local maxima and minima 
    pll, pval = ip.find_features(yy, dval, gpix=dpix, thres=vthres)
    nll, nval = ip.find_features(yy, -dval, gpix=dpix, thres=vthres)
    # exclude the edge positions 
    pvv, = np.where((pll > margin) & (pll < ny-margin))
    nvv, = np.where((nll > margin) & (nll < ny-margin))
    p_num, n_num = len(pll), len(nll)
    
    if p_num * n_num == 0:
        print 'Found no strips ...' 
        p_pos, n_pos = None, None
    else:
        p_pos, n_pos = pll[pvv], nll[nvv]
    
    #p_pos, n_pos = pll, nll

    '''
    plt.subplot(211)
    plt.plot(yy,icol)
    plt.plot(p_pos, icol[p_pos], 'go')
    plt.plot(n_pos, icol[n_pos], 'ro')
    plt.subplot(212)
    plt.plot(yy,dval)
    plt.plot(p_pos, dval[p_pos], 'go')
    plt.plot(n_pos, dval[n_pos], 'ro')
    plt.show()
    '''
    return p_pos, n_pos 

def ap_tracing_peak(img, npoints=40, spix=10, dpix=20, thres=1400, \
                start_col=None, ap_thres=None, degree=None):
    ''' 
    Trace the apertures starting at "start_col" position (for stellar spectrum) 
     - inputs
        1. npoints : slicing number along the column direction
        2. spix : summing pixel size along column for finding peaks
        3. dpix : detecting pixel range of feature to be regarded as the same  
        4. thres : detecting intensity level for emission features 
        5. start_col : starting column number for tracing 
        6. ap_thres : threshold of pixel range to identify nearby aperture features
        7. degree : degree of polynomial fitting      
     - outputs
        coefficients of polynomial fittings for each aperture   
    '''
    # image size 
    ny, nx = img.shape  
    
    # define the threshold pixel range to be matched with nearby aperture positions 
    if ap_thres == None: ap_thres= nx/npoints/2
    # define the degree of polynomials for fitting 
    if degree == None: degree = 3 #5
    
    # find the apertures in the middle column
    if start_col == None: start_col = nx/2
    icol = np.mean(img[:,(start_col-spix/2):(start_col+spix/2)],axis=1)
    mcaps = peak_find(icol, thres=thres, dpix=dpix)
    n_ap = len(mcaps)

    # make set of (x, y) positions for each aperture
    # add the (x, y) positions at default column 
    apx = []
    apy = []
    for mcap in mcaps:
        apx.append([start_col]) 
        apy.append([mcap])
    
    # define the tracing direction (left, right) 
    xsteps = np.linspace(spix,nx-spix,npoints)
    xlefts = xsteps[(xsteps < start_col)]
    xrights = xsteps[(xsteps > start_col)]
    
    # the left side columns 
    pcaps = np.copy(mcaps) 
    for xpos in reversed(xlefts):
        icol = np.mean(img[:,(xpos-spix/2):(xpos+spix/2)],axis=1)
        caps = np.array(peak_find(icol, dpix=dpix, thres=thres))
        print xpos, len(caps), n_ap
        # check the aperture number based on mcap positions 
        for i, pcap in enumerate(pcaps): 
            tcap = caps[(caps < pcap+ap_thres) & ( caps > pcap-ap_thres)]
            if len(tcap) == 1: 
                # save this position to the (x, y) position list
                apx[i].append(xpos)
                apy[i].append(tcap[0])
                # save this position for the next tracing loop 
                pcaps[i] = tcap[0] 
            elif len(tcap) == 0: 
                print 'ap[%d](x=%d) : The matching aperture position was not found' \
                   % (i, xpos)
            else: 
                print 'ap[%d](x=%d) : The matching aperture position was too many(%d)' \
                   % (i, xpos, len(tcap))
    
    # the right side columns 
    pcaps = np.copy(mcaps)                      
    for xpos in xrights:
        icol = np.mean(img[:,(xpos-spix/2):(xpos+spix/2)],axis=1)
        caps = np.array(peak_find(icol, dpix=dpix, thres=thres))
        print xpos, len(caps), n_ap
        # check the aperture number based on mcap positions 
        for i, pcap in enumerate(pcaps): 
            tcap = caps[(caps < pcap+ap_thres) & ( caps > pcap-ap_thres)]
            if len(tcap) == 1: 
                # save this position to the (x, y) position list
                apx[i].append(xpos)
                apy[i].append(tcap[0])
                # save this position for the next tracing loop
                pcaps[i] = tcap[0]
            elif len(tcap) == 0: 
                print 'ap[%d](x=%d) : The matching aperture position was not found' \
                   % (i, xpos)
            else: 
                print 'ap[%d](x=%d) : The matching aperture position was too many(%d)' \
                   % (i, xpos, len(tcap))
                         
    # sorting the (x, y) positions along x-direction for each aperture  
    # polynomial fitting with degree 
    ap_coeffs = []
    z1, z2 = ip.zscale(img)
    plt.imshow(img, cmap='hot', vmin=z1, vmax=z2)
    for i in range(n_ap-1): 
        tapx = np.array(apx[i],dtype=np.float)
        tapy = np.array(apy[i],dtype=np.float)
        tsort = np.argsort(tapx)
        coeff = np.polynomial.polynomial.polyfit(tapx[tsort],tapy[tsort],degree)
        ap_coeffs.append(coeff)
        yfit = np.polynomial.polynomial.polyval(tapx[tsort],coeff)
        plt.plot(tapx[tsort],tapy[tsort], 'go')
        plt.plot(tapx[tsort],yfit,'b-', linewidth=10, alpha=0.3)
        print 'ap[%d]: delta_y = %12.8f %12.8f ' \
          % (i, np.mean(tapy[tsort]-yfit), np.std(tapy[tsort]-yfit))
        
    plt.xlim(0,2048)
    plt.ylim(0,2048)
    #plt.show()
    return ap_coeffs

def ap_tracing_strip(band, filename, npoints=40, spix=10, dpix=20, thres=14000, \
                start_col=None, ap_thres=None, degree=None, \
                devide=8.0, target_path=MANUAL_PATH):
    ''' 
    Trace the apertures starting at "start_col" position (for FLAT spectrum)
     - inputs
        0. band 
        1. npoints : slicing number along the column direction
        2. spix : summing pixel size along vertical column for finding strips
        3. dpix : detecting pixel range of feature to be regarded as the same  
        4. thres : detecting intensity level for emission features 
        5. start_col : starting column number for tracing 
        6. ap_thres : threshold of pixel range to identify nearby aperture features
        7. degree : degree of polynomial fitting      
     - outputs
        1. coefficients of polynomial fittings 
           for start position of each aperture on y-axis   
        2. coefficients of polynomial fittings 
           for end position of each aperture on y-axis
    '''
    img, hdr = ip.readfits(filename)
    img, hdr = ip.readfits(filename)
    fpath, fname = ip.split_path(filename)
    if ip.exist_path(target_path) == False: target_path = fpath
    name = '.'.join(fname.split('.')[:-1])
    # image size 
    ny, nx = img.shape  
    
    # define the threshold pixel range to be matched with nearby aperture positions 
    if ap_thres == None: ap_thres= nx/npoints/4
    # define the degree of polynomials for fitting 
    if degree == None: degree = 3 #7
    
    # find the apertures in the middle column
    #if start_col == None: start_col = start_col #1500 #nx/2
    icol = np.mean(img[:,(start_col-spix/2):(start_col+spix/2)], axis=1)
    #print 'icol', icol, len(icol)
    mcaps1, mcaps2 = strip_find(band, icol)
    n_ap = len(mcaps1)

    # make set of (x, y) positions for each aperture
    # add the (x, y) positions at default column 
    apx1, apx2 = [], []
    apy1, apy2 = [], []
    for mcap1, mcap2 in zip(mcaps1, mcaps2):
        apx1.append([start_col]) 
        apx2.append([start_col])
        apy1.append([mcap1])
        apy2.append([mcap2])
    
    # define the tracing direction (left, right) 
    xsteps = np.linspace(spix,nx-spix,npoints)
    xlefts = xsteps[(xsteps < start_col)]
    xrights = xsteps[(xsteps > start_col)]
    
    # the left side columns 
    pcaps1, pcaps2 = np.copy(mcaps1), np.copy(mcaps2)
    for xpos in reversed(xlefts):
        icol = np.mean(img[:,(xpos-spix/2):(xpos+spix/2)],axis=1)
        caps1, caps2 = np.array(strip_find(band, icol, dpix=dpix, margin=10))
        #print xpos, n_ap, len(caps1), len(caps2)
        # check the aperture number based on mcap positions 
        for i, pcap1, pcap2 in zip(range(n_ap), pcaps1, pcaps2): 
            tcap1 = caps1[(caps1 < pcap1+ap_thres) & ( caps1 > pcap1-ap_thres)]
            if len(tcap1) == 1: 
                # save this position to the (x, y) position list
                apx1[i].append(xpos)
                apy1[i].append(tcap1[0])
                # save this position for the next tracing loop 
                pcaps1[i] = tcap1[0] 
            elif len(tcap1) == 0: 
                print 'ap[%d](x=%d) : The matching aperture position was not found' \
                   % (i, xpos)
            else: 
                print 'ap[%d](x=%d) : The matching aperture position was too many(%d)' \
                   % (i, xpos, len(tcap1))

            tcap2 = caps2[(caps2 < pcap2+ap_thres) & ( caps2 > pcap2-ap_thres)]
            if len(tcap2) == 1: 
                # save this position to the (x, y) position list
                apx2[i].append(xpos)
                apy2[i].append(tcap2[0])
                # save this position for the next tracing loop 
                pcaps2[i] = tcap2[0] 
            elif len(tcap2) == 0: 
                print 'ap[%d](x=%d) : The matching aperture position was not found' \
                   % (i, xpos)
            else: 
                print 'ap[%d](x=%d) : The matching aperture position was too many(%d)' \
                   % (i, xpos, len(tcap2))

    
    # the right side columns 
    pcaps1, pcaps2 = np.copy(mcaps1), np.copy(mcaps2)                      
    for xpos in xrights:
        icol = np.mean(img[:,(xpos-spix/2):(xpos+spix/2)],axis=1)
        caps1, caps2 = np.array(strip_find(band, icol, dpix=dpix))
        print xpos, n_ap, len(caps1), len(caps2)
        # check the aperture number based on mcap positions 
        for i, pcap1, pcap2 in zip(range(n_ap), pcaps1, pcaps2): 
            tcap1 = caps1[(caps1 < pcap1+ap_thres) & ( caps1 > pcap1-ap_thres)]
            if len(tcap1) == 1: 
                # save this position to the (x, y) position list
                apx1[i].append(xpos)
                apy1[i].append(tcap1[0])
                # save this position for the next tracing loop 
                pcaps1[i] = tcap1[0] 
            elif len(tcap1) == 0: 
                print 'ap[%d](x=%d) : The matching aperture position was not found' \
                   % (i, xpos)
            else: 
                print 'ap[%d](x=%d) : The matching aperture position was too many(%d)' \
                   % (i, xpos, len(tcap1))

            tcap2 = caps2[(caps2 < pcap2+ap_thres) & ( caps2 > pcap2-ap_thres)]
            if len(tcap2) == 1: 
                # save this position to the (x, y) position list
                apx2[i].append(xpos)
                apy2[i].append(tcap2[0])
                # save this position for the next tracing loop 
                pcaps2[i] = tcap2[0] 
            elif len(tcap2) == 0: 
                print 'ap[%d](x=%d) : The matching aperture position was not found' \
                   % (i, xpos)
            else: 
                print 'ap[%d](x=%d) : The matching aperture position was too many(%d)' \
                   % (i, xpos, len(tcap2))

    
    # sorting the (x, y) positions along x-direction for each aperture  
    # polynomial fitting with degree 
    ap_coeffs1, ap_coeffs2 = [], []
    z1, z2 = ip.zscale(img)
    plt.figure(figsize=(15,15))
    plt.imshow(img, cmap='gray', vmin=z1, vmax=z2)
    for k in range(len(apx1)): #n_ap): 
        tapx1 = np.array(apx1[k],dtype=np.float)
        tapx2 = np.array(apx2[k],dtype=np.float)
        tapy1 = np.array(apy1[k],dtype=np.float)
        tapy2 = np.array(apy2[k],dtype=np.float)
        
        tsort1 = np.argsort(tapx1)
        tsort2 = np.argsort(tapx2)
        coeff1 = np.polynomial.polynomial.polyfit(tapx1[tsort1],tapy1[tsort1],degree)
        coeff2 = np.polynomial.polynomial.polyfit(tapx2[tsort2],tapy2[tsort2],degree)
        
        # save the fitting coefficients 
        ap_coeff = np.zeros([2,degree+1])
        ap_coeff[0,:] = coeff1
        ap_coeff[1,:] = coeff2
        np.savetxt(target_path+'apmap_%s_%02d.%03d.dat' % (band, degree, k), ap_coeff)
        
        ap_coeffs1.append(coeff1)
        ap_coeffs2.append(coeff2)
        
        yfit1 = np.polynomial.polynomial.polyval(tapx1[tsort1],coeff1)
        yfit2 = np.polynomial.polynomial.polyval(tapx2[tsort2],coeff2)
        
        plt.plot(tapx1[tsort1],tapy1[tsort1], 'go')
        plt.plot(tapx2[tsort2],tapy2[tsort2], 'ro')
        
        print 'ap[%d]: delta_y1 = %12.8f %12.8f ' \
          % (k, np.mean(tapy1[tsort1]-yfit1), np.std(tapy1[tsort1]-yfit1))
        print 'ap[%d]: delta_y2 = %12.8f %12.8f ' \
          % (k, np.mean(tapy2[tsort2]-yfit2), np.std(tapy2[tsort2]-yfit2))
        
        xx = np.arange(nx)
        yfit1 = np.polynomial.polynomial.polyval(xx,coeff1)
        yfit2 = np.polynomial.polynomial.polyval(xx,coeff2)
        
        plt.plot(xx,yfit1,'g-', linewidth=3, alpha=0.6)
        plt.plot(xx,yfit2,'r-', linewidth=3, alpha=0.6)
        
    plt.xlim(0,nx)
    plt.ylim(0,ny)
    plt.show()
    plt.savefig(target_path+name+'_aptracing2.png')
    plt.close('all')
    
    return ap_coeffs1, ap_coeffs2


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
            rr = np.arange((click_x-2*dpix),(click_x+2*dpix+1), dtype=np.int)
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
    
def _ap_extract1(img, ap_coeffs, width=[-30,30], hdr=None):
    '''
    Extract the strips based on the stellar spectrum with a regular width
     - inputs 
      1. image (2d-array)
      2. coefficients (a list of 1d-array) 
      3. width (+/- positions) 
      4. header 
     - outputs
      1. strips (a list of 2d-array) 
      2. headers (a list of header object)  
    '''
    ny, nx = img.shape 
    
    # generate the x position array 
    xx = np.arange(nx)
    # define the y-direction width for extraction 
    ywidth = width[1]-width[0]+1
    # generate the list of strip image arrays 
    strips = []
    hdrs = [] 
    for i, coeff in enumerate(ap_coeffs): 
        tstrip = np.zeros([ywidth,nx], dtype=np.float64)
        # calculate the y-position of each aperture for x coordinate 
        yy = np.polynomial.polynomial.polyval(xx,coeff)
        # extract the each segment (y-direction) for each x-position 
        for xpos, ypos in zip(xx,yy):
            iy = np.int(ypos)
            # calculate the fraction ratio from the discrete pixel coordinate 
            frac = ypos - iy 
            # define the min/max y-position and the array index to the strip array 
            #print xpos, iy+width[0], iy+width[1]
            if (iy+width[1] < 0) | (iy+width[0] > ny-1):
                print 'ap[%d], x=%d, [%d,%d] :Y-range is not valid' % (i,xpos,iy+width[0],iy+width[1])
                continue
            if (iy+width[0] < 0):
                y1, y2 = (0, iy+width[1]+1)
                p1, p2 = (ywidth-(y2-y1), ywidth)
            elif (iy+width[1] > ny-1):
                y1, y2 = (iy+width[0], ny-1)
                p1, p2 = (0, ywidth-(y2-y1))
            else:
                y1, y2 = (iy+width[0], iy+width[1]+1)
                p1, p2 = (0, ywidth)
            # make the column segment array 
            
            tcol = img[y1:y2,xpos]*(1.0-frac) + img[(y1+1):(y2+1),xpos]*frac
            # input the column segment to the strip array 
            tstrip[p1:p2,xpos] = tcol
            
        print 'ap[%d] : extracted ' % (i,)
        strips.append(tstrip)
        
        if hdr == None:
            thdr = None 
        else: 
            thdr = hdr.copy() 
            thdr.update('AP-TIME', time.strftime('%Y-%m-%d %H:%M:%S'))
            thdr.update('AP_NUM', i)
            strcoeff = '' 
            for c in coeff:
                strcoeff = strcoeff + '%12.8f' % (c,)
            thdr.update('AP-COEF', strcoeff)
            thdr.update('AP-YRAN', '[%d,%d]' % (width[0], width[1]))
        hdrs.append(thdr)    
        
    return strips, hdrs

def _ap_extract2(band, filename, ap_coeffs1 ,ap_coeffs2, \
                width=60, ap_num=[], target_path=MANUAL_PATH, 
                savepng=False):
    '''
    Extract the strips based on the FLAT image 
     - inputs 
      0. band 
      1. image file name FITS 
      2. coefficients (1) (a list of 1d-array) - start position of y-axis
      3. *coefficients (2) (a list of 1d-array) - end position of y-axis
      4. width of output strip 2D-array  
      5. ap_num (a list of aperture number, which will be extracted)
     - outputs
      0. FITS files 
      1. strips (a list of 2d-array) 
      2. headers (a list of header object)  
    '''
    img, hdr = ip.readfits(filename)
    fpath, fname = ip.split_path(filename)
    if ip.exist_path(target_path) == False: target_path = fpath
    # extract the file name only without extension
    name = '.'.join(fname.split('.')[:-1])
    
    ny, nx = img.shape
    if len(ap_num) == 0:
        ap_num = range(len(ap_coeffs1))
    
    n_ap = len(ap_num)
        
    strips = [] 
    hdrs = []
    f1 = plt.figure(1, figsize=(15,15))
    f2 = plt.figure(2, figsize=(15,2))
    a1 = f1.add_subplot(111)
    a2 = f2.add_subplot(111)
    
    a1.imshow(img, cmap='gray')
    for i in ap_num:
        # generate the output 2d-strips 
        tstrip = np.zeros([width, nx], dtype=np.float64)
        
        '''
        # (case 1) if ap_coeffs2 is exist, for arbitrary aperture width 
        if ap_coeffs2 != None:
            c1 = ap_coeff[i]
            c2 = ap_coeffs2[i]
            # define the x-positions and y-positions 
            xpos = np.arange(nx)
            ypos1 = np.polynomial.polynomial.polyval(xpos, c1)
            ypos2 = np.polynomial.polynomial.polyval(xpos, c2)
            # use interpolation 
            for x0, y1, y2 in zip(xpos, ypos1, ypos2):
                ypos = np.linspace(y1, y2, width)
                yinp = np.interp(ypos, np.arange(ny), img[:,x0])
                tstrip[:,x0] = yinp
         
        '''
        # (case 2) using just aperture width for extracting from the ap_coeffs1
        c1 = ap_coeffs1[i]
        c2 = ap_coeffs2[i]
        # define the x-positions and y-positions 
        xpos = np.arange(nx)
        ypos1 = np.polynomial.polynomial.polyval(xpos, c1)
        ypos2 = ypos1 + width 
        # use the cropping method 
        yy, xx = np.indices(tstrip.shape)
        # add the y-direction aperture curve to the y-coordinates
        ayy = yy + ypos1
        iyy = np.array(np.floor(ayy), dtype=np.int)
        fyy = ayy - iyy 
        # find the valid points
        vv = np.where((iyy >= 0) & (iyy <= ny-2))
        tstrip[yy[vv],xx[vv]] = \
           img[iyy[vv],xx[vv]] * (1.0 - fyy[yy[vv],xx[vv]]) + \
           img[(iyy[vv]+1),xx[vv]] * fyy[yy[vv],xx[vv]]
    
        # draw the extracting position
        a1.plot(xpos, ypos1, 'g-', linewidth=3, alpha=0.5) 
        a1.plot(xpos, ypos2, 'r-', linewidth=3, alpha=0.5)
                                
        if savepng == True : 
            a2.imshow(tstrip, cmap='gray')
            f2.savefig(target_path+name+'.%03d.png' % (i,))
            a2.cla()
        
        print 'ap[%d] : extracted ' % (i,)
        strips.append(tstrip)
        
        if hdr == None:
            thdr = None 
        else: 
            thdr = hdr.copy() 
            thdr.update('AP-TIME', time.strftime('%Y-%m-%d %H:%M:%S'))
            thdr.update('AP-MODE', 'manual')
            thdr.update('AP-NUM', i)
            c1list = [] 
            for c in c1:
                c1list.append('%.8E' % (c,))
            thdr.update('AP-COEF1', ','.join(c1list))
            c2list = []  
            for c in c2:
                c2list.append('%.8E' % (c,))
            thdr.update('AP-COEF2', ','.join(c2list))
            thdr.update('AP-WIDTH', '%d' % (width,))
        
        hdrs.append(thdr) 
        ip.savefits(target_path+name+'.%03d.fits' % (i,), tstrip, header=thdr)
         
    a1.set_xlim(0,nx)
    a1.set_ylim(0,ny)
    f1.savefig(target_path+name+'.all.png')            
    plt.close('all')
    
    return strips, hdrs


def _line_identify_old(stripfile, linefile=None, \
                  npoints=30, spix=4, dpix=5, thres=13000, fdeg=5, desc=''):

    def key_press(event): 
        
        if event.key == 'q': plt.close('all')
        
        if event.key == 'a':
            line_idx = np.argmin( (event.xdata - ysignal)**2 ) 
            line_x = ysignal[line_idx]
        
            print line_idx, line_x
            # mark the points for fitting 
            ll, = np.where((xx < line_x+dpix) & (xx > line_x-dpix))
            print len(ll)
            l1 = ax.plot([line_x, line_x], [0,ny], 'r-', linewidth=5, alpha=0.5)
            l2 = ax.plot(xx[ll], yy[ll], 'ro')
            coeff = np.polynomial.polynomial.polyfit(xx[ll],yy[ll],fdeg)
            fig.canvas.draw() 
        
            print coeff 
            ycoeff[line_idx,:] = np.array(coeff)
            # redraw for the marked points 
        
            # read the wavelength for this line 
            input = raw_input('input wavelength:')
            try:
                line_wv = np.float(input)
            except: 
                line_wv = 0.0 
                
            print line_idx, line_x, line_wv
            ywave[line_idx] = line_wv
            
    strip, hdr = ip.readfits(stripfile)
    
    ny, nx = strip.shape
    z1, z2 = ip.zscale(strip)
    
    # fitting the base level 
    snr = 10
    ysum = np.average(strip[(ny/2-spix):(ny/2+spix),:], axis=0)
    ll = np.arange(nx)
    nrejt = 0
    while (1):
        yc = np.polynomial.polynomial.polyfit(ll, ysum[ll], fdeg)
        yfit = np.polynomial.polynomial.polyval(np.arange(nx), yc)
        ll, = np.where(ysum < yfit*(1.0 + 1.0/snr))
        #print nrejt 
        if nrejt == len(ll): break 
        nrejt = len(ll)

    ysignal = np.array(peak_find(ysum, thres=thres, dpix=dpix ))
      
    # define figure object 
    fig = plt.figure(figsize=(14,4))
    ax = fig.add_subplot(111)
    
    # draw the strip image         
    ax.imshow(strip, cmap='gray',vmin=z1, vmax=z2, aspect='auto')
    ax.set_xlim(0,nx)
    ax.set_ylim(-5,ny+5)
    # mark the number of lines 
    for i, ypeak in enumerate(ysignal): 
        if i % 2 == 1:
            ytext = ny
        else:
            ytext = -5
        ax.text(ypeak, ytext, i+1, fontsize=15)
        
    # find the emission feature in each row 
    ysteps = np.linspace(spix, ny-spix, npoints)
    # make the array for emission feature points 
    xx, yy = ([], [])
    for ypos in ysteps:
        irow = np.mean(strip[(ypos-spix/2):(ypos+spix/2),:],axis=0)
        #a2.plot(irow)
        # find the peak positions 
        xpos_list = peak_find(irow, dpix=dpix, thres=thres)
        # draw and save the peak positions (x, y)  
        for xpos in xpos_list:
            ax.plot(xpos, ypos, 'go', markersize=5, alpha=0.5)
            xx.append(xpos)
            yy.append(ypos)
    # convert into the numpy array         
    xx = np.array(xx, dtype=np.float)
    yy = np.array(yy, dtype=np.float)
    
    # generate the wavelength array for each line 
    ywave = np.zeros(len(ysignal))
    ycoeff = np.zeros([len(ysignal),fdeg+1])
    fig.canvas.mpl_connect('key_press_event', key_press)
    print '[a] add line and input wavelength'
    print '[r] reset'
    print '[q] quit from this aperture'
    plt.show()

           
    print 'thanks'    

def line_plot(strip, thres, spix, dpix):
    pass 
'''
ap_tracing_strip(band, filename, npoints=40, spix=10, dpix=20, thres=1400, \
                start_col=None, ap_thres=None, degree=None, \
                target_path=MANUAL_PATH)
'''
 

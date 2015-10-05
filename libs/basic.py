
import math 
import itertools 
import os 
import glob
import numpy as np 
from scipy.optimize import curve_fit
from scipy.optimize import fmin
from scipy.ndimage import map_coordinates 
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata, interp2d, interp1d  
from scipy.stats import nanmean,nanmedian,nanstd
from H_series import H_series
import libs.fits as pyfits


#############################################################################
# FITS FILE I/O  
#############################################################################

def savefits(filename, data, header=None):
    '''
    Save to the FITS file 
    '''
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    if len(data.shape) == 1: data = data.reshape(-1,len(data))
    
    pyfits.writeto(filename, data, header, clobber=True)
    #print 'savefits, data.shape', data.shape
    
def readfits(filename):
    '''
    Read the data from FITS file 
    '''
    fobj = pyfits.open(filename)[0] 
    return fobj.data, fobj.header

def newheader():
    '''
    Make new FITS header 
    '''
    return pyfits.Header()
    
def fopen(fname):
    '''
    Open the FITS file 
    ''' 
    fits = pyfits.open(fname)[0]
    return fits.data, fits.header

def fwrite(fname, img, hdr=None):
    '''
    Write to the FITS file 
    '''
    if hdr == None:
        pyfits.writeto(fname, img, clobber=True) 
    else: 
        pyfits.writeto(fname, img, hdr, clobber=True)
        
def split_path(fname):
    '''
    Split file path into path, filename 
    '''
    return os.path.dirname(fname)+'/', os.path.basename(fname)

def exist_path(path):
    '''
    Check if the file/directory exists
    ''' 
    if path == None: 
        pbool = False 
    else: 
        pbool = os.path.exists(path)
    return pbool

def find_files(filter):
    return glob.glob(filter)
    
def read_lines(filename):
    '''
    Read the line list file 
     - outputs : wavelength, intensity 
    '''
    # read the line list 
    dat = np.loadtxt(filename)
    wav = dat[:,0]
    flx = dat[:,1]
    
    if wav[0] > 3.0:
        wav = wav/10000.0 
    return wav, flx


def read_orderinfo(desc):
    '''
    Read the order information of IGRINS 
     - outputs : order#, order description, start lambda, end lambda 
     - requirements : "IGRINS_WVRAN_H.dat", "IGRINS_WVRAN_H.dat" 
    '''
    
    # define band 
    band = desc[0]
    
    # read the wavelength information for each order
    dat = np.loadtxt('IGRINS_WVRAN_%s.dat' % (band,))
    worder = dat[:,0]
    wv_start = dat[:,1]
    wv_end = dat[:,2]   
    
    odesc = []
    for i, iord in enumerate(worder):
        odesc.append("%s%03d" % (band, iord))
    onum = worder 
    odesc = np.array(odesc) 
    owv1 = wv_start-0.005
    owv2 = wv_end+0.005
        
    if len(desc) == 1: 
        return onum, odesc, owv1, owv2 
    elif len(desc) == 4: 
        iord = np.int(desc[1:4])
        oo = np.where(onum == iord)[0]
        return owv1[oo], owv2[oo]
    
def save_fitting(desc, xcoeff, ycoeff, pdeg=[5,3], fitting_path='./fitdata/'):

    pass

def load_fitting(desc, pdeg=[5,3], fitting_path='./fitdata/'):
    
    pass 

def save_mapping(band, PA=0, offset=[1023.5, 1023.5], pscale=0.018, mapping_path='./mapdata/'):
    
    pass 

def load_mapping(band, mapping_path='./mapdata/'):
    
    pass 



#############################################################################
# IMAGE MAPPING FUNCTIONS  
#############################################################################

def rotate_coord(xx, yy, PA=0):
    '''
    Rotate the coordinates variables around (0,0)
    '''
    rPA = -np.deg2rad(PA)
    txx = xx*np.cos(rPA) - yy*np.sin(rPA)
    tyy = xx*np.sin(rPA) + yy*np.cos(rPA)
    return txx, tyy  


def xy2pix(xx, yy, offset=[1023.5,1023.5], PA=0, pscale=0.018):
    '''
    Convert (x, y) positions in mm into (X, Y) in pixel positions
     - (0,0) => (1024, 1024)
     - apply the rotating transform with (x, y) before scaling 
    '''
    sxx = xx/pscale
    syy = yy/pscale 
     
    txx, tyy = rotate_coord(sxx, syy, PA=PA)
    
    rxx, ryy = (txx+offset[0], tyy+offset[1]) 
    
    return rxx, ryy


def pix2xy(xx, yy, offset=[1023.5,1023.5], PA=0, pscale=0.018):
    '''
    Convert the (X, Y) pixel coordinate into (x, y) in mm scale
     - the center of xx, yy  
    '''
    xx = xx - offset[0]
    yy = yy - offset[1]  
    
    sxx = xx*pscale
    syy = yy*pscale 
     
    rxx, ryy = rotate_coord(sxx, syy, PA=-PA)
    
    return rxx, ryy    


#############################################################################
# 2D-ARRAY IMAGE PROCESSING  
#############################################################################

def imcombine(img_list, mode='median'):
    '''
    Combine images (imgges might have NaN values)
    '''
    if mode.startswith('me'):
        rimg = np.median(img_list, axis=0)
    elif mode.startswith('su'): 
        rimg = np.sum(img_list, axis=0)
    elif mode.startswith('av'):
        rimg = np.average(img_list, axis=0)
#     if mode.startswith('me'):
#         rimg = nanmedian(img_list, axis=0)
#     elif mode.startswith('su'): 
#         rimg = np.nansum(img_list, axis=0)
#     elif mode.startswith('av'):
#         rimg = nanmean(img_list, axis=0)
        
    return rimg 

def imarith(img1, img2, op='-'):
    '''
    Operation with images 
    '''
    if op == '-':
        rimg = img1 - img2 
    elif op == '+':
        rimg = img1 + img2
    elif op == '*':
        rimg = img1 * img2 
    elif op == '/': 
        rimg = img1 / img2 
    return rimg

def imextract(img, px, py, order=2, mode='constant'):
    '''
    Extract (px, py) pixels from img by mapping 
    '''
    return map_coordinates(img,[py, px], order=order, mode=mode)

def imblur(img, gpix):
    '''
    Perform Gaussian convolution for image with pixel size  
    '''
    return gaussian_filter(img, gpix)

def iminterpol(img, ix, iy, ox, oy, method='nearest'):
    '''
    Find the value for (ox, oy) in the image by interpolation
     - inputs : img (data), ix (image x coord.), iy (image y coord.)
     - inputs : ox, oy for outputs 
     - output : the image data corresponding to (ox, oy)  
    '''
    val = np.array(img.flatten(), dtype=np.double)
    lx = np.array(ix.flatten(), dtype=np.double)
    ly = np.array(iy.flatten(), dtype=np.double) 
    oimg = griddata((lx, ly), val, (ox, oy), method=method)
    return oimg 

def iminterp1d(ox, x, y, method='linear'):
    x = np.array(x.flatten(), dtype=np.double)
    y = np.array(y.flatten(), dtype=np.double)
    f = interp1d(x, y, kind=method, bounds_error=False)
    
    return f(ox)

def iminterp2d(img, x, y, ox, oy, method='linear'):
    x = np.array(x.flatten(), dtype=np.double)
    y = np.array(y.flatten(), dtype=np.double)
    z = np.array(img.flatten(), dtype=np.double)
    f = interp2d(x, y, z, kind=method, bounds_error=False)
    
    return f(ox, oy)

#############################################################################
# Image Processing 
#############################################################################

def fixpix(inputdata, inputmask, intertype, wbox=10):
    '''
    Fix the bad-pixels by using the mask     
    '''
    mask = inputmask != 0   # Create mask
    
    test = inputdata.copy()  # Copy data()
    badfixindices = np.argwhere(mask)
           
    if intertype == 'cinterp':

        for badpixpos in badfixindices:
            x =  badpixpos[0] 
            y =  badpixpos[1] 

            y_0 = 0
            y_1 = 0

            # Nearest good pixels point
            try:
                for k in range(1,wbox):
                    if mask[x,y-k] == False:
                        y_0 = k
                        break
            except IndexError:
                y_0 = 0
            
            try:
                for z in range(1,wbox):
                   if mask[x,y+z] == False:
                        y_1 = z
                        break
            except IndexError:
                y_1 = 0
        
            # Replace with nearest good pixels
            test[x,y] = test[x,y-y_0] + ((test[x,y+y_1] - test[x,y-y_0])* y_0)/(y_0+y_1)
            

    elif intertype == 'linterp':
        
        for badpixpos in badfixindices:
            x =  badpixpos[0] 
            y =  badpixpos[1] 

            y_0 = 0
            y_1 = 0

            # Nearest good pixels point
            try:
                for k in range(1,wbox):
                    if mask[x-k,y] == False:
                        y_0 = k
                        break
            except IndexError:
                y_0 = 0
            
            try:
                for z in range(1,wbox):
                   if mask[x+z,y] == False:
                        y_1 = z
                        break
            except IndexError:
                y_1 = 0
        
            # Replace with nearest good pixels
            test[x,y] = test[x-y_0,y] + ((test[x+y_1,y] - test[x-y_0,y])* y_0)/(y_0+y_1)

    return test    

def cosmicrays(data, threshold=1000, flux_ratio=0.2, wbox=3):
    '''
    Correct the cosmic-ray pixels by using threshold and flux_ratio factors 
    '''
    # Create threshold
    mask = data > threshold

    # Copy data()
    test = data.copy()
    cosmicindices = np.argwhere(mask)
    #print 'Indices = ', cosmicindices

#    w = test.shape[0]
#    h = test.shape[1]
  
    for cosmicpos in cosmicindices:
        x =  cosmicpos[0] 
        y =  cosmicpos[1] 
        cutoff = test[(x-wbox):(x+wbox),(y-wbox):(y+wbox)]  # 8 surrounding pixels
        cutoff = cutoff[cutoff != test[x,y]]
        aver =  np.median(cutoff) # np.average(cutoff)  
        fluxratio = aver / test[x,y]
        if fluxratio < flux_ratio:      # Apply threshold = 0.2
            test[x,y] = aver
        else:
            pass

    return test    

def sigma_clip(data, sig=3, iters=1, cenfunc=np.median, varfunc=np.var,
               axis=None, copy=True):
    """FROM ASTROPY 
       Perform sigma-clipping on the provided data.

    This performs the sigma clipping algorithm - i.e. the data will be iterated
    over, each time rejecting points that are more than a specified number of
    standard deviations discrepant.

    .. note::
        `scipy.stats.sigmaclip` provides a subset of the functionality in this
        function.

    Parameters
    ----------
    data : array-like
        The data to be sigma-clipped (any shape).
    sig : float
        The number of standard deviations (*not* variances) to use as the
        clipping limit.
    iters : int or None
        The number of iterations to perform clipping for, or None to clip until
        convergence is achieved (i.e. continue until the last iteration clips
        nothing).
    cenfunc : callable
        The technique to compute the center for the clipping. Must be a
        callable that takes in a masked array and outputs the central value.
        Defaults to the median (numpy.median).
    varfunc : callable
        The technique to compute the standard deviation about the center. Must
        be a callable that takes in a masked array and outputs a width
        estimator.  Defaults to the standard deviation (numpy.var).
    axis : int or None
        If not None, clip along the given axis.  For this case, axis=int will
        be passed on to cenfunc and varfunc, which are expected to return an
        array with the axis dimension removed (like the numpy functions).
        If None, clip over all values.  Defaults to None.
    copy : bool
        If True, the data array will be copied.  If False, the masked array
        data will contain the same array as `data`.  Defaults to True.

    Returns
    -------
    filtered_data : `numpy.masked.MaskedArray`
        A masked array with the same shape as `data` input, where the points
        rejected by the algorithm have been masked.

    Notes
    -----
     1. The routine works by calculating

            deviation = data - cenfunc(data [,axis=int])

        and then setting a mask for points outside the range

            data.mask = deviation**2 > sig**2 * varfunc(deviation)

        It will iterate a given number of times, or until no further points are
        rejected.

     2. Most numpy functions deal well with masked arrays, but if one would
        like to have an array with just the good (or bad) values, one can use::

            good_only = filtered_data.data[~filtered_data.mask]
            bad_only = filtered_data.data[filtered_data.mask]

        However, for multidimensional data, this flattens the array, which may
        not be what one wants (especially is filtering was done along an axis).

    Examples
    --------

    This will generate random variates from a Gaussian distribution and return
    a masked array in which all points that are more than 2 *sample* standard
    deviation from the median are masked::

        >>> from astropy.stats import sigma_clip
        >>> from numpy.random import randn
        >>> randvar = randn(10000)
        >>> filtered_data = sigma_clip(randvar, 2, 1)

    This will clipping on a similar distribution, but for 3 sigma relative to
    the sample *mean*, will clip until converged, and does not copy the data::

        >>> from astropy.stats import sigma_clip
        >>> from numpy.random import randn
        >>> from numpy import mean
        >>> randvar = randn(10000)
        >>> filtered_data = sigma_clip(randvar, 3, None, mean, copy=False)

    This will clip along one axis on a similar distribution with bad points
    inserted::

        >>> from astropy.stats import sigma_clip
        >>> from numpy.random import normal
        >>> from numpy import arange, diag, ones
        >>> data = arange(5)+normal(0.,0.05,(5,5))+diag(ones(5))
        >>> filtered_data = sigma_clip(data, axis=0, sig=2.3)

    Note that along the other axis, no points would be masked, as the variance
    is higher.

    """

    if axis is not None:
        cenfunc_in = cenfunc
        varfunc_in = varfunc
        cenfunc = lambda d: np.expand_dims(cenfunc_in(d, axis=axis), axis=axis)
        varfunc = lambda d: np.expand_dims(varfunc_in(d, axis=axis), axis=axis)

    filtered_data = np.ma.array(data, copy=copy)

    if iters is None:
        i = -1
        lastrej = filtered_data.count() + 1
        while(filtered_data.count() != lastrej):
            i += 1
            lastrej = filtered_data.count()
            do = filtered_data - cenfunc(filtered_data)
            filtered_data.mask |= do * do > varfunc(filtered_data) * sig ** 2
        iters = i + 1
    else:
        for i in range(iters):
            do = filtered_data - cenfunc(filtered_data)
            filtered_data.mask |= do * do > varfunc(filtered_data) * sig ** 2

    return filtered_data
    #filtered_data.data[filtered_data.mask] = np.nan 
    #return filtered_data.data


def A0Vsclip(spec,fit,low=4., high=4.):#, vmask=False):
    '''
    2013-12-26 CKSIM - sigma clip for fitting A0V star with gaussian H absorption profiles
    '''        
    specnew = spec.copy()
    c = spec - fit
    size = c.size
    delta = 1    
    while delta:
        c_std = nanstd(c)  
        c_median = nanmedian(c) #median
        oldsize = size
        critlower = c_median - c_std*low
        critupper = c_median + c_std*high
        v1 = np.where(c <= critlower)        
        v2 = np.where(c >= critupper)
        
        if len(v1[0]) > 0:
            c[v1[0]] = np.nan
            specnew[v1[0]] = np.nan
        if len(v2[0]) > 0:
            c[v2] = np.nan
            specnew[v2[0]] = np.nan        
        vv = np.where(np.isnan(c)==False)[0]
        size = len(vv)
        delta = oldsize - size          
        
#    if vmask == False:
#        return specnew
#    elif vmask == True:
#        return vv
    
    return specnew, vv


#TODO Note scipy dependency
            
            
#############################################################################
# 2D-POLYNOMIAL FITTING 
#############################################################################

def polyfit2d(x, y, z, deg=(5,5), rcond=1E-20):
    '''
    Find 2D polynomial fitting of z data with (x, y) 
    '''
    x = np.array(x.flatten(), dtype=np.double) 
    y = np.array(y.flatten(), dtype=np.double)
    z = np.array(z.flatten(), dtype=np.double) 
    ncols = (deg[0] + 1)*(deg[1] + 1)
    G = np.zeros((x.size, ncols), dtype=np.double)
    ij = itertools.product(range(deg[0]+1), range(deg[1]+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z, rcond)
    return m

def polyval2d(x, y, m, deg=(5,5)):
    '''
    Evaluate the value of (x, y) by fitting function coefficients  
    '''
    x = np.array(x, dtype=np.double)
    y = np.array(y, dtype=np.double)
    ij = itertools.product(range(deg[0]+1), range(deg[1]+1))
    z = np.zeros_like(x, dtype=np.double)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

def polycoeff2d(m, deg=(5,5)):
    ij = itertools.product(range(deg[0]+1), range(deg[1]+1))
    for a, (i,j) in zip(m, ij):
        print 'coeff. of x^%d y^%d = %12.8f' % (i, j, a)

#############################################################################
# GAUSSAIN FITTING  
#############################################################################
     
def gauss(x, *p):
    '''
    Generate the simple Gaussian function 
    '''
    peak, center, width = p
    return peak*np.exp(-(x-center)**2 / (2.0*width**2)  )
 
def gauss_fit(x, y, p0=[1.,0.,1.]):
    '''
    Find the simple Gaussian fitting function  
    '''
    coeff, var_matrix = curve_fit(gauss, x, y, p0=p0)
    return coeff 

def gaussian_convolve(y, size):
    '''
    Perform the Gaussian convolution for the spectrum 
    '''
    x = np.mgrid[-size:size+1]
    g = np.exp(-(x**2/np.float64(size)))
    kernal = g / g.sum()
    result = np.convolve(y, kernal, mode='same')
    return result 

def gauss_mixture(x,*p):
    '''
    The sum of Gaussian absorption profiles
    2013-12-26 CKSIM
    '''
    if isinstance(p[0],np.ndarray): p = p[0]
    if isinstance(p[0],tuple): p = p[0]
    gfit=np.zeros_like(x)    
    for i in range(len(p)/3):
        
        peak, center, width = p[i*3], p[i*3+1], p[i*3+2]
        
        width = sigma(0.004) #2013-12-31 cksim. for development stage only
           
        gfit += peak*np.exp(-(x-center)**2 / (2.0*width**2)  )
     
    return gfit  

def sigma(FWHM):
    '''
    2013-12-26 CKSIM
    '''
    import math
    
    sigma = FWHM / (2.*math.sqrt(2*math.log(2)))
    
    return sigma

#############################################################################
# SIMPLEX   
#############################################################################

def simplex(points, values, p0):
    global val, pts
    val = values 
    pts = points 
    def fgrid(x): 
        fval = griddata(pts, val, tuple(x), method='nearest')
        print x 
        print fval 
        return fval 

    res = fmin(fgrid, p0, ftol=1e-12, xtol=10., disp=1) 

    return res 

#############################################################################
# ARRAY PROCESSING  
#############################################################################

def shift(a,n):
    a = np.array(a)
    return np.concatenate([a[n:],a[:n]])

def fshift(a,dx):
    intx = np.floor(dx)
    fracx = dx - intx 
    a1 = shift(a,intx)
    a2 = a1*(1.-fracx) + shift(a1,1)*fracx 
    return a2     
    
#############################################################################
# NUMERICAL DERIVATIVES 
#############################################################################

def nderiv(x,y,smoother=5):
    '''
    Find the numerical derivatives 
    '''
    
    if x.size != y.size:
        print 'size of x, y is not the same'
        return 
#    if mask != None:
#        if mask.size != x.size:
#            print 'size of mask is not the same to x, y'
#            return  
#        mm = np.where(mask != 1)
#        tx = x[mm].copy()
#        ty = y[mm].copy()
#    else:

    tx = x.copy()
    ty = gaussian_convolve(y.copy(), smoother)
    dx = np.diff(tx)
    dy = np.diff(ty)
    
    dy1 = dy/dx 
    dx1 = tx[:-1] + dx/2.
    
        
    result = np.interp(tx, dx1, dy1)
    np.interp
    return result

def make_points_dpix(nx, dpix):
    '''
    Find the points having with difference = dpix in the array of size = nx 
    - input : x size , delta pixel 
    - output : cx (indices in array) , cwidth (for width between cx) 
    '''
    cwidth = (dpix + (dpix % 2))/2 - 1  
    # middle point of slit direction
    midx = nx/2 
    # pixel size for correlation 
    t1 = np.arange(midx,0,-dpix)
    t2 = np.arange(midx,nx,dpix)
    cx = np.concatenate([t1[::-1],t2[1:]])
    
    return cx, cwidth 

#############################################################################
# Spectrum utilities ===========================================================
# from https://bitbucket.org/nhmc/pyserpens/src/7826e643ad71/utilities.py
# nhmc / pyserpens by Neil Crighton
#############################################################################

def find_edges_true_regions(condition, dpix=1):
    """ Finds the indices for the edges of contiguous regions where
    condition is True.

    Examples
    --------
    >>> a = np.array([3,0,1,4,6,7,8,6,3,2,0,3,4,5,6,4,2,0,2,5,0,3])
    >>> ileft, iright = find_edges_true_regions(a > 2)
    >>> zip(ileft, iright)
    [(0, 0), (3, 8), (11, 15), (19, 19), (21, 21)]

    """
#    indices, = condition.nonzero() #2013-11-22 cksim commented this
    indices = condition.nonzero()[0] #2013-11-22 cksim
    if not len(indices):
        return None, None
    iright = (indices[1:] - indices[:-1] > dpix).nonzero()[0]
    ileft = iright + 1
    iright = np.concatenate( (iright, [len(indices)-1]) )
    ileft = np.concatenate( ([0], ileft) )
    return indices[ileft], indices[iright]

def find_signif_peaks(array, thres=200):
    '''
    Find the start, peak significance and end pixel indices of
    absorption and emission features.
    Note ew must be positive for absorption.
     - inputs : 
        1. 1d-array = intensity 
        2. ndetect = absolute intensity threshold for detecting lines 
     - outputs :
        set of indices of pixels over ndetect     
    '''
    # gradually decrease the detection level from the significance of
    # the maximum feature to the minimum level, adding features as they
    # rise above the detection level.
    signif = array
    detectlevels = np.sort(signif)

    lines = set()
    for level in reversed(detectlevels):
        if level < thres: break
        condition = (signif >= level)
        # find the edges of contiguous regions
        ind0,ind1 = find_edges_true_regions(condition, dpix=1)
        # add to lines where we have the tip of a feature
        if ind0 != None : lines.update(ind0[ind0-ind1 == 0])

    if not lines:
        return None

    return sorted(lines)

def cr_reject(flux, error, nsigma=15.0, npix=2, verbose=False):
    """ Given flux and errors, rejects cosmic-ray type or dead
    pixels. These are defined as pixels that are more than
    nsigma*sigma above or below the median of the npixEL pixels on
    either side.

    Returns newflux,newerror where the rejected pixels have been
    replaced by the median value of npix to either side, and the
    error has been set to NaN.

    The default values work ok for S/N~20, Resolution=500 spectra.
    """
    if verbose:  print nsigma,npix
    flux,error = list(flux), list(error)  # make copies
    i1 = npix
    i2 = len(flux) - npix
    for i in range(i1, i2):
        # make a list of flux values used to find the median
        fl = flux[i-npix:i] + flux[i+1:i+1+npix]
        er = error[i-npix:i] + error[i+1:i+1+npix]
        fl = [f for f,e in zip(fl,er) if e > 0]
        er = [e for e in er if e > 0]
        medfl = np.median(fl)
        meder = np.median(er)
        if np.abs((flux[i] - medfl) / meder) > nsigma:
            flux[i] = medfl
            error[i] = np.nan
            if verbose:  print len(fl), len(er)

    return np.array(flux), np.array(error)

#def find_features(x, y, thres=250, gpix=3, i=0): #2013-11-21 cksim commented 
def find_features(x, y, thres=250, gpix=3, i=0): #2013-11-21 cksim
    '''
    Find the emission features in 1d-array 
     - inputs : 
       1. thres = threshold to detect the emissions
       2. gpix = pixel difference to be identified as the same group 
       
    '''
    # find peak indices in the spectrum flx  
    pidx = find_signif_peaks(y, thres=thres)
    if pidx == None: 
        return 0, 0
    else:
        pidx = np.array(pidx)
    # set the arrays with 1 for selected points 
    a = np.zeros(y.size)
    a[pidx] = 1
    # grouping the peaks within "dpix" again 
    ileft, iright = find_edges_true_regions((a > 0), dpix=gpix)
    # calculate the pixel centers of peaks for each group in average   
    cx = [] 
    cy = [] 
    for i1, i2 in zip(ileft, iright):
        # add x2 range + 1  for indexing 
        if i1 == i2:
            tx, ty = (x[i1], y[i1])
        else:
            tt = np.where((pidx >= i1) & (pidx <= i2))[0]
            tidx = pidx[tt]
            mm = np.nanargmax(y[tidx])
            tx = x[tidx[mm]]
            ty = y[tidx[mm]]
            
            #tx = np.nansum(x[pidx[tt]]*y[pidx[tt]])/np.nansum(y[pidx[tt]])
            #ty = np.nanmax(y[pidx[tt]])
        cx.append(tx)
        cy.append(ty)
    cx = np.array(cx)
    cy = np.array(cy)

    return cx, cy
#############################################################################
# zscale =======================================================================
#############################################################################

def zscale(image, nsamples=1000, contrast=0.25, bpmask=None, zmask=None):
    MAX_REJECT = 0.5
    MIN_NPIXELS = 5
    GOOD_PIXEL = 0
    BAD_PIXEL = 1
    KREJ = 2.5
    MAX_ITERATIONS = 5
    def _zscale (image, nsamples=1000, contrast=0.25, bpmask=None, zmask=None):
        """Implement IRAF zscale algorithm
        nsamples=1000 and contrast=0.25 are the IRAF display task defaults
        bpmask and zmask not implemented yet
        image is a 2-d numpy array
        returns (z1, z2)
        """
    
        # Sample the image
        samples = zsc_sample (image, nsamples, bpmask, zmask)
        npix = len(samples)
        samples.sort()
        zmin = samples[0]
        zmax = samples[-1]
        # For a zero-indexed array
        center_pixel = (npix - 1) / 2
        if npix%2 == 1:
            median = samples[center_pixel]
        else:
            median = 0.5 * (samples[center_pixel] + samples[center_pixel + 1])
    
        #
        # Fit a line to the sorted array of samples
        minpix = max(MIN_NPIXELS, int(npix * MAX_REJECT))
        ngrow = max (1, int (npix * 0.01))
        ngoodpix, zstart, zslope = zsc_fit_line (samples, npix, KREJ, ngrow,
                                                 MAX_ITERATIONS)
    
        if ngoodpix < minpix:
            z1 = zmin
            z2 = zmax
        else:
            if contrast > 0: zslope = zslope / contrast
            z1 = max (zmin, median - (center_pixel - 1) * zslope)
            z2 = min (zmax, median + (npix - center_pixel) * zslope)
        return z1, z2
    
    def zsc_sample (image, maxpix, bpmask=None, zmask=None):
        
        # Figure out which pixels to use for the zscale algorithm
        # Returns the 1-d array samples
        # Don't worry about the bad pixel mask or zmask for the moment
        # Sample in a square grid, and return the first maxpix in the sample
        nc = image.shape[0]
        nl = image.shape[1]
        stride = max (1.0, math.sqrt((nc - 1) * (nl - 1) / np.float64(maxpix)))
        stride = int (stride)
        samples = image[::stride,::stride].flatten()
        return samples[:maxpix]
        
    def zsc_fit_line (samples, npix, krej, ngrow, maxiter):
    
        #
        # First re-map indices from -1.0 to 1.0
        xscale = 2.0 / (npix - 1)
        xnorm = np.arange(npix)
        xnorm = xnorm * xscale - 1.0
    
        ngoodpix = npix
        minpix = max (MIN_NPIXELS, int (npix*MAX_REJECT))
        last_ngoodpix = npix + 1
    
        # This is the mask used in k-sigma clipping.  0 is good, 1 is bad
        badpix = np.zeros(npix, dtype="int32")
    
        #
        #  Iterate
    
        for niter in range(maxiter):
    
            if (ngoodpix >= last_ngoodpix) or (ngoodpix < minpix):
                break
            
            # Accumulate sums to calculate straight line fit
            goodpixels = np.where(badpix == GOOD_PIXEL)
            sumx = xnorm[goodpixels].sum()
            sumxx = (xnorm[goodpixels]*xnorm[goodpixels]).sum()
            sumxy = (xnorm[goodpixels]*samples[goodpixels]).sum()
            sumy = samples[goodpixels].sum()
            sum = len(goodpixels[0])
    
            delta = sum * sumxx - sumx * sumx
            # Slope and intercept
            intercept = (sumxx * sumy - sumx * sumxy) / delta
            slope = (sum * sumxy - sumx * sumy) / delta
            
            # Subtract fitted line from the data array
            fitted = xnorm*slope + intercept
            flat = samples - fitted
    
            # Compute the k-sigma rejection threshold
            ngoodpix, mean, sigma = zsc_compute_sigma (flat, badpix, npix)
    
            threshold = sigma * krej
    
            # Detect and reject pixels further than k*sigma from the fitted line
            lcut = -threshold
            hcut = threshold
            below = np.where(flat < lcut)
            above = np.where(flat > hcut)
    
            badpix[below] = BAD_PIXEL
            badpix[above] = BAD_PIXEL
            
            # Convolve with a kernel of length ngrow
            kernel = np.ones(ngrow,dtype="int32")
            badpix = np.convolve(badpix, kernel, mode='same')
    
            ngoodpix = len(np.where(badpix == GOOD_PIXEL)[0])
            
            niter += 1
    
        # Transform the line coefficients back to the X range [0:npix-1]
        zstart = intercept - slope
        zslope = slope * xscale
    
        return ngoodpix, zstart, zslope
    
    def zsc_compute_sigma (flat, badpix, npix):
    
        # Compute the rms deviation from the mean of a flattened array.
        # Ignore rejected pixels
    
        # Accumulate sum and sum of squares
        goodpixels = np.where(badpix == GOOD_PIXEL)
        sumz = flat[goodpixels].sum()
        sumsq = (flat[goodpixels]*flat[goodpixels]).sum()
        ngoodpix = len(goodpixels[0])
        if ngoodpix == 0:
            mean = None
            sigma = None
        elif ngoodpix == 1:
            mean = sumz
            sigma = None
        else:
            mean = sumz / ngoodpix
            temp = sumsq / (ngoodpix - 1) - sumz*sumz / (ngoodpix * (ngoodpix - 1))
            if temp < 0:
                sigma = 0.0
            else:
                sigma = math.sqrt (temp)
    
        return ngoodpix, mean, sigma
    return _zscale(image, nsamples=nsamples, contrast=contrast, bpmask=bpmask, zmask=zmask)

#############################################################################
# WCS 
#############################################################################

#cond2r = np.deg2rad(1.0) # math.pi/180.
# allowed projections
#ctypes = ("-sin","-tan","-arc","-ncp", "-gls", "-mer", "-ait", "-stg")


def getwcs(hdr=None, filename=None,ext=0,rot='crota2',cd=False):
   """Get the WCS geometry from a fits file and return as a dictionary
   
      filename - name of the fits file
      ext      - extension number to read for header info
      rot      - name for rotation angle header keyword.  Can also be an angle
                 in degrees
      cd       - set to true to use CD matrix instead of rot."""
   
   wcs = {'rot' : None, 'cd' : None}
   
   if (filename == None) and (hdr != None):
       hdr = hdr 
   elif (filename != None) and (hdr == None):    
       img = pyfits.open(filename)
       hdr = img[ext].header
       img.close()
   else:
       print 'No header and filename!'
       return None 
       
   hdrkeys = hdr.ascardlist().keys() # list of existing keywords
   wcs['crval1'] = np.float64(hdr['crval1'])
   wcs['crval2'] = np.float64(hdr['crval2'])
   try:
       wcs['cdelt1'] = np.float64(hdr['cdelt1'])
       wcs['cdelt2'] = np.float64(hdr['cdelt2'])
   except:
       wcs['cdelt1'] = 1
       wcs['cdelt2'] = 1   
   wcs['crpix1'] = np.float64(hdr['crpix1'])
   wcs['crpix2'] = np.float64(hdr['crpix2'])
   wcs['proj']   = hdr['ctype1'][-4:] # just want projection
   if 'CD1_1' in hdrkeys: cd = True
   if cd is True: # use CD matrix
      cd = [0,0,0,0]
      if 'CD1_1' in hdrkeys:
         cd[0] = np.float64(hdr['cd1_1'])
      if 'CD1_2' in hdrkeys:
         cd[1] = np.float64(hdr['cd1_2'])
      if 'CD2_1' in hdrkeys:
         cd[2] = np.float64(hdr['cd2_1'])
      if 'CD2_2' in hdrkeys:
         cd[3] = np.float64(hdr['cd2_2'])
      wcs['cd'] = cd
   else: # use rotation
      if isinstance(rot,str): # is a keyword
         if rot in hdrkeys:
            wcs['rot'] = np.float64(hdr[rot])
         else:
            #sys.stderr.write("### Warning! Cannot find rotation angle keyword %s.  Assuming angle=0\n" %rot)
            wcs['rot'] = 0
      else:
         wcs['rot'] = rot
   
   
   return wcs

def sky2xy(ra,dec,wcs):
   """Convert ra,dec into x,y pixels
   
      ra,dec - can be either single values or iterable list/tuples/etc
      wcs - a dictionary containing the wcs as returned by getwcs()"""
   
   def _xypix(xpos, ypos, xref, yref, xrefpix, yrefpix, xinc, yinc, proj, 
      rot=None, cd=None, deps=1.0e-5):
      """Convert input ra,dec to x,y pixels
   
         xpos    : x (RA) coordinate (deg)
         ypos    : y (dec) coordinate (deg)
         xref    : x reference coordinate value (deg) (CRVAL1)
         yref    : y reference coordinate value (deg) (CRVAL2)
         xrefpix : x reference pixel (CRPIX1)
         yrefpix : y reference pixel (CRPIX2)
         xinc    : x coordinate increment (deg) (CDELT1)
         yinc    : y coordinate increment (deg) (CDELT2)
         proj    : projection type code e.g. -SIN (CTYPE1/2)
         rot     : rotation (deg)  (from N through E)
         cd      : list/tuple of four values (the cd_matrix)
         
         Note, you MUST define either rot or cd.  If both are given, cd is 
         used by default.
         
         returns the x,y pixel positions, or raises ValueError for angles too
         large for projection, ValueError if xinc or yinc is zero, and
         Arithmetic error in one instance. """
   
      proj = proj.lower()
      
      # check axis increments - bail out if either 0
      if (xinc == 0.0) or (yinc == 0.0):
         raise ValueError("Input xinc or yinc is zero!")
   
      # 0h wrap-around tests added by D.Wells 10/12/94:
      dt = (xpos - xref)
      if (np.max(dt) > +180):
         xpos -= 360
      if (np.min(dt) < -180):
         xpos += 360
   
      if cd is not None and proj in ('-ait','-mer'):
         raise ValueError('cd matrix cannot be used with -AIT or -MER projections!')
      elif rot is not None:
         cosr = np.cos(rot * (np.pi/180.0))
         sinr = np.sin(rot * (np.pi/180.0))
   
      # Non linear position
      ra0  = xref * (np.pi/180.0)
      dec0 = yref * (np.pi/180.0)
      ra   = xpos * (np.pi/180.0)
      dec  = ypos * (np.pi/180.0)
   
      # compute direction cosine
      coss = np.cos(dec)
      sins = np.sin(dec)
      l    = np.sin(ra-ra0) * coss
      sint = sins * np.sin(dec0) + coss * np.cos(dec0) * np.cos(ra-ra0)
   
      if proj == '-sin':
         if np.min(sint) < 0.0:
            raise ValueError("Angle too large for projection!")
         m = sins * np.cos(dec0) - coss * np.sin(dec0) * np.cos(ra-ra0)
      elif proj == '-tan':
         if np.min(sint) <= 0.0:
            raise ValueError("Angle too large for projection!")
         m = sins * np.sin(dec0) + coss * np.cos(dec0) * np.cos(ra-ra0)
         l = l / m
         m = (sins*np.cos(dec0) - coss*np.sin(dec0) * np.cos(ra-ra0)) / m
      elif proj == '-arc':
         m = sins * np.sin(dec0) + coss * np.cos(dec0) * np.cos(ra-ra0)
         if m < -1.0:
            m = -1.0
         elif m > 1.0:
            m = 1.0
         m = np.arccos(m)
         if m != 0:
            m = m / np.sin(m)
         else:
            m = 1.0
         l = l * m
         m = m*(sins*np.cos(dec0) - coss*np.sin(dec0) * np.cos(ra-ra0))
      elif proj == '-ncp': # North celestial pole
         if dec0 == 0.0:
            raise ValueError("Angle too large for projection!") # can't stand the equator
         else:
            m = (np.cos(dec0) - coss * np.cos(ra-ra0)) / np.sin(dec0)
      elif proj == '-gls': # global sinusoid
         dt = ra - ra0
         if np.max(abs(dec)) > np.pi/2.:
            raise ValueError("Angle too large for projection!")
         if abs(dec0) > np.pi/2.:
            raise ValueError("Angle too large for projection!")
         m = dec - dec0
         l = dt * coss
      elif proj == '-mer': # mercator
         dt = yinc * cosr + xinc * sinr
         if dt == 0.0:
            dt = 1.0
         dy = (yref/2.0 + 45.0) * (np.pi/180.0)
         dx = dy + dt / 2.0 * (np.pi/180.0)
         dy = np.log(np.tan(dy))
         dx = np.log(np.tan(dx))
         geo2 = dt * (np.pi/180.0) / (dx - dy)
         geo3 = geo2 * dy
         geo1 = np.cos(yref*(np.pi/180.0))
         if geo1 <= 0.0:
            geo1 = 1.0
         dt = ra - ra0
         l  = geo1 * dt
         dt = dec / 2.0 + np.pi/4.
         dt = np.tan(dt)
         if np.min(dt) < deps:
            raise ArithmeticError("dt < %f" %deps)
         m = geo2 * np.log(dt) - geo3
      elif proj == '-ait': # Aitoff
         l = 0.0
         m = 0.0
         da = (ra - ra0) / 2.0
         if np.max(abs(da)) > np.pi/2.:
            raise ValueError("Angle too large for projection!")
         dt = yinc*cosr + xinc*sinr
         if dt == 0.0:
            dt = 1.0
         dt = dt * (np.pi/180.0)
         dy = yref * (np.pi/180.0)
         dx = np.sin(dy+dt)/np.sqrt((1+np.cos(dy+dt))/2.) - \
              np.sin(dy)/np.sqrt((1+np.cos(dy))/2.)
         if dx == 0.0:
            dx = 1.0
         geo2 = dt / dx
         dt = xinc*cosr - yinc* sinr
         if dt == 0.0:
            dt = 1.0
         dt = dt * (np.pi/180.0)
         dx = 2.0 * np.cos(dy) * np.sin(dt/2.0)
         if dx == 0.0:
            dx = 1.0
         geo1 = dt * np.sqrt((1.0+np.cos(dy)*np.cos(dt/2.0))/2.) / dx
         geo3 = geo2 * np.sin(dy) / np.sqrt((1+np.cos(dy))/2.)
         dt = np.sqrt ((1 + np.cos(dec) * np.cos(da))/2.)
         if np.min(abs(dt)) < deps:
            raise ZeroDivisionError("dt < %f" %deps)
         l = 2.0 * geo1 * np.cos(dec) * np.sin(da) / dt
         m = geo2 * np.sin(dec) / dt - geo3
      elif proj == '-stg': # Sterographic
         da = ra - ra0
         if np.max(abs(dec)) > np.pi/2.:
            raise ValueError("Angle too large for projection!")
         dd = 1.0 + sins * np.sin(dec0) + coss * np.cos(dec0) * np.cos(da)
         if np.min(abs(dd)) < deps:
            raise ValueError("Angle too large for projection!")
         dd = 2.0 / dd
         l = l * dd
         m = dd * (sins * np.cos(dec0) - coss * np.sin(dec0) * np.cos(da))
      else: # linear
         l = (np.pi/180.0)*(xpos - xref)
         m = (np.pi/180.0)*(ypos - yref)
   
      # back to degrees 
      dx = l / (np.pi/180.0)
      dy = m / (np.pi/180.0)
         
      if cd is not None:
         if len(cd) != 4:
            raise IndexError("You must give four values for the cd matrix!")
         dz = dx*cd[0] + dy*cd[1]
         dy = dx*cd[2] + dy*cd[3]
         dx = temp
      elif rot is not None:
         # correct for rotation
         dz = dx*cosr + dy*sinr
         dy = dy*cosr - dx*sinr
         dx = dz
   
         # correct for xinc,yinc
         dx = dx / xinc
         dy = dy / yinc
      else: # both are None
         raise ValueError("You must define either rot or cd keywords!")
   
      # convert to pixels
      xpix = dx + xrefpix
      ypix = dy + yrefpix
      return xpix, ypix
   
   if wcs['cd'] is not None:
      x,y = _xypix(ra, dec, wcs['crval1'], wcs['crval2'], wcs['crpix1'], 
      wcs['crpix2'], wcs['cdelt1'], wcs['cdelt2'], wcs['proj'],
      cd=wcs['cd'])
   elif wcs['rot'] is not None:
      x,y = _xypix(ra, dec, wcs['crval1'], wcs['crval2'], wcs['crpix1'], 
      wcs['crpix2'], wcs['cdelt1'], wcs['cdelt2'], wcs['proj'],
      rot=wcs['rot'])
   else:
      raise KeyError("Either rot or cd must be specified with wcs!")
   
   return x, y  

def xy2sky(x,y,wcs):
   """Convert x,y into ra,dec
   
      x,y - can be either single values or iterable list/tuples/etc
      wcs - a dictionary containing the wcs as returned by getwcs()"""
          
   def _worldpos(xpix, ypix, xref, yref, xrefpix, yrefpix, xinc, yinc, proj,
      rot=None, cd=None, deps=1.0e-5):
      """Convert x,y pixel value to world coordinates in degrees
   
         xpix    : x pixel number
         ypix    : y pixel number
         xref    : x reference coordinate value (deg) (CRVAL1)
         yref    : y reference coordinate value (deg) (CRVAL2)
         xrefpix : x reference pixel (CRPIX1)
         yrefpix : y reference pixel (CRPIX2)
         xinc    : x coordinate increment (deg) (CDELT1)
         yinc    : y coordinate increment (deg) (CDELT2)
         proj    : projection type code e.g. "-SIN"
         rot     : rotation (deg)  (from N through E)
         cd      : list/tuple of four values (the cd_matrix)
         
         Note, you MUST define either rot or cd.  If both are given, cd is 
         used by default.
         
         returns the two coordinates, raises ValueError if the angle is too
         large for projection, and ValueError if xinc or yinc is zero."""
   
      proj = proj.lower()
         
      # check axis increments - bail out if either 0
      if (xinc == 0.0) or (yinc == 0.0):
         raise ValueError("Input xinc or yinc is zero!")
   
      # Offset from ref pixel
      dx = xpix - xrefpix
      dy = ypix - yrefpix
   
      if cd is not None:
         if len(cd) != 4:
            raise IndexError("You must give four values for the cd matrix!")
         if proj in ('-ait','-mer'):
            raise ValueError('cd matrix cannot be used with -AIT or -MER projections!')
         temp = dx*cd[0] + dy*cd[1]
         dy   = dx*cd[2] + dy*cd[3]
         dx   = temp
      elif rot is not None:
         # scale by xinc
         dx = dx * xinc
         dy = dy * yinc
         
         # Take out rotation
         cosr = np.cos(rot*(np.pi/180.0))
         sinr = np.sin(rot*(np.pi/180.0))
         if (rot != 0.0):
            temp = dx * cosr - dy * sinr
            dy   = dy * cosr + dx * sinr
            dx   = temp
      else: # both are None
         raise ValueError("You must define either rot or cd keywords!")
   
      # convert to radians
      ra0    = xref * (np.pi/180.0)     
      dec0   = yref * (np.pi/180.0)
      l      = dx * (np.pi/180.0)
      m      = dy * (np.pi/180.0)
      sins   = l*l + m*m
      decout = 0.0
      raout  = 0.0
      cos0   = np.cos(dec0)
      sin0   = np.sin(dec0)
   
      if proj == '-sin':
         if (np.max(sins) > 1.0):
            raise ValueError("Angle too large for projection!")
         coss = np.sqrt(1.0 - sins)
         dt = sin0 * coss + cos0 * m
         if (np.max(abs(dt)) > 1):
            raise ValueError("Angle too large for projection!")
         dect = np.arcsin(dt)
         rat = cos0 * coss - sin0 * m
         #if ((rat==0.0) and (l==0.0)):
         #   raise ValueError("Angle too large for projection!")
         rat = np.arctan2(l, rat) + ra0;
         
      elif proj == '-tan':
         if (np.max(sins) > 1.0):
            raise ValueError("Angle too large for projection!")
         dect = cos0 - m * sin0
         if (np.max(dect) == 0.0):
            raise ValueError("Angle too large for projection!")
         rat = ra0 + np.arctan2(l, dect)
         dect = np.arctan(np.cos(rat-ra0) * (m * cos0 + sin0) / dect)
      elif proj == '-arc':
         if (sins >= np.pi**2):
            raise ValueError("Angle too large for projection!")
         sins = np.sqrt(sins)
         coss = np.cos(sins)
         
         sins[(sins != 0)] = np.sin(sins[(sins != 0)]) / sins[(sins != 0)]
         sins[(sins == 0)] = 1.0 
         
         dt = m * cos0 * sins + sin0 * coss
         if (np.max(abs(dt)) > 1):
            raise ValueError("Angle too large for projection!")
         dect = np.arcsin(dt)
         da = coss - dt * sin0
         dt = l * sins * cos0
         dtest, = np.where((da == 0.0) & (dt == 0.0))
         if len(dtest) > 0: 
            raise ValueError("Angle too large for projection!")
         rat = ra0 + np.arctan2(dt, da)
      elif proj == '-ncp': # north celestial pole
         dect = cos0 - m * sin0
         if dect == 0.0:
            raise ValueError("Angle too large for projection!")
         rat = ra0 + np.arctan2(l, dect)
         dt = np.cos(rat-ra0)
         if 0 in dt:
            raise ValueError("Angle too large for projection!")
         dect = dect / dt
         if np.max(abs(dect)) > 1.0:
            raise ValueError("Angle too large for projection!")
         dect = np.arccos(dect)
         if dec0 < 0.0:
            dect = -dect
      elif proj == '-gls': # global sinusoid
         dect = dec0 + m
         if np.max(abs(dect)) > np.pi/2.:
            raise ValueError("Angle too large for projection!")
         coss = np.cos(dect)
         if np.max(abs(l)) > np.pi*coss:
            raise ValueError("Angle too large for projection!")
         rat = np.zeros(l.shape) + ra0
         rat[coss > deps] = rat[coss > deps] + l / coss[coss > deps]
      elif proj == '-mer': # mercator
         dt = yinc * cosr + xinc * sinr
         if dt == 0.0:
            dt = 1.0
         dy = (yref/2.0 + 45.0) * (np.pi/180.0)
         dx = dy + dt / 2.0 * (np.pi/180.0)
         dy = np.log(np.tan(dy))
         dx = np.log(np.tan(dx))
         geo2 = dt * (np.pi/180.0) / (dx - dy)
         geo3 = geo2 * dy
         geo1 = np.cos(yref*(np.pi/180.0))
         if geo1 <= 0.0:
            geo1 = 1.0
         rat = l / geo1 + ra0
         if np.max(abs(rat - ra0)) > 2*np.pi:
            raise ValueError("Angle too large for projection!") # added 10/13/94 DCW/EWG
         dt = 0.0
         if geo2 != 0.0:
            dt = (m + geo3) / geo2
         dt = np.exp(dt)
         dect = 2.0 * np.arctan(dt) - np.pi/2.
      elif proj == '-ait': # Aitoff
         dt = yinc*cosr + xinc*sinr
         if dt == 0.0:
            dt = 1.0
         dt = dt * (np.pi/180.0)
         dy = yref * (np.pi/180.0)
         dx = np.sin(dy+dt)/np.sqrt((1.0 + np.cos(dy+dt))/2.0) - \
              np.sin(dy)/np.sqrt((1.0 + np.cos(dy))/2.0)
         if dx == 0.0:
            dx = 1.0
         geo2 = dt / dx
         dt = xinc*cosr - yinc* sinr
         if dt == 0.0:
            dt = 1.0
         dt = dt * (np.pi/180.0)
         dx = 2.0 * np.cos(dy) * np.sin(dt/2.0)
         if dx == 0.0:
            dx = 1.0
         geo1 = dt * np.sqrt((1 + np.cos(dy)* np.cos(dt/2.0))/2.0) / dx
         geo3 = geo2 * np.sin(dy) / np.sqrt((1 + np.cos(dy))/2.0)
         
         dz = 4.0 - l*l/(4.0*geo1*geo1) - ((m+geo3)/geo2)**2
         if (np.max(dz) > 4.0) or (np.min(dz) < 2.0):
            raise ValueError("Angle too large for projection!")
         dz = 0.5 * np.sqrt(dz)
         dd = (m + geo3) * dz / geo2
         if np.max(abs(dd)) > 1.0:
            raise ValueError("Angle too large for projection!")
         dd = np.arcsin(dd)
         if np.min(abs(np.cos(dd))) < deps:
            raise ValueError("Angle too large for projection!")
         da = l * dz / (2.0 * geo1 * np.cos(dd))
         if np.max(abs(da)) > 1.0:
            raise ValueError("Angle too large for projection!")
         da = np.arcsin(da)
         rat = ra0 + 2.0 * da
         dect = dd
         
         '''
         rat  = ra0
         dect = dec0
         if (l == 0.0) and (m == 0.0):
            pass
         else:
            dz = 4.0 - l*l/(4.0*geo1*geo1) - ((m+geo3)/geo2)**2
            if (dz > 4.0) or (dz < 2.0):
               raise ValueError("Angle too large for projection!")
            dz = 0.5 * math.sqrt(dz)
            dd = (m + geo3) * dz / geo2
            if abs(dd) > 1.0:
               raise ValueError("Angle too large for projection!")
            dd = math.asin(dd)
            if abs(math.cos(dd)) < deps:
               raise ValueError("Angle too large for projection!")
            da = l * dz / (2.0 * geo1 * math.cos(dd))
            if abs(da) > 1.0:
               raise ValueError("Angle too large for projection!")
            da = math.asin(da)
            rat = ra0 + 2.0 * da
            dect = dd
         '''
      elif proj == '-stg': # Sterographic
         dz = (4.0 - sins) / (4.0 + sins)
         if np.max(abs(dz)) > 1.0:
            raise ValueError("Angle too large for projection!")
         dect = dz * sin0 + m * cos0 * (1.0+dz) / 2.0
         if np.max(abs(dect)) > 1.0:
            raise ValueError("Angle too large for projection!")
         dect = np.arcsin(dect)
         rat  = np.cos(dect)
         if np.min(abs(rat)) < deps:
            raise ValueError("Angle too large for projection!")
         rat = l * (1.0+dz) / (2.0 * rat)
         if np.max(abs(rat)) > 1.0:
            raise ValueError("Angle too large for projection!")
         rat = np.arcsin(rat)
         mg = 1.0 + np.sin(dect) * sin0 + np.cos(dect) * cos0 * np.cos(rat)
         if np.min(abs(mg)) < deps:
            raise ValueError("Angle too large for projection!")
         mg = 2*(np.sin(dect)*cos0 - np.cos(dect)*sin0*np.cos(rat)) / mg
         
         rat[abs(mg-m) > deps] = np.pi - rat[abs(mg-m) > deps]
         rat = ra0 + rat
      else: # default is linear
         rat  =  ra0 + l
         dect = dec0 + m
   
      # return ra in range
      raout = rat;
      decout = dect;
      #if raout-ra0 > math.pi:
      #   raout = raout - 2*math.pi
      #elif raout-ra0 < -math.pi:
      #   raout = raout + 2*math.pi
      #if raout < 0.0:
      #   raout += 2*math.pi # added by DCW 10/12/94
   
      # correct units back to degrees
      xpos  = raout  / (np.pi/180.0)
      ypos  = decout  / (np.pi/180.0)
      return xpos,ypos
  
   #print wcs['cd'] 
   if wcs['cd'] is not None:
      ra,dec = _worldpos(x, y, wcs['crval1'], wcs['crval2'], wcs['crpix1'], 
               wcs['crpix2'], wcs['cdelt1'], wcs['cdelt2'], wcs['proj'],
               cd=wcs['cd'])
   elif wcs['rot'] is not None:
      ra,dec = _worldpos(x, y, wcs['crval1'], wcs['crval2'], wcs['crpix1'], 
              wcs['crpix2'], wcs['cdelt1'], wcs['cdelt2'], wcs['proj'],
              rot=wcs['rot'])
   else:
      raise KeyError("Either rot or cd must be specified with wcs!")
  
   return ra, dec 
  

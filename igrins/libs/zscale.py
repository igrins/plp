#############################################################################
# zscale =======================================================================
#############################################################################

import numpy as np
import math

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

if __name__ == "__main__":
    im = np.random.randn(10000).reshape((100, 100))
    print zscale(im)

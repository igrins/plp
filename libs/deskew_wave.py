'''
Distortion correction and wavelength calibration
Created on 2013.11.25

Last updated on 2014.04.03

by Huynh Anh

'''
import time, os
import basic as ip
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, OldScalarFormatter
import manual as mn
from scipy import interpolate
from scipy import *
from scipy.optimize import curve_fit, leastsq
from scipy import optimize

import extract as ex
import pyfits

import ohlines_h as hband
import ohlines_kv2 as kband

# Functions

# Gaussian fitting function to find the centers

#------------------------------------------------------------------------------------------------

"""1. Gauss Fitting for one absorption line Routine"""

def gaussian(height,center,width,background):
    """ Returns a gaussian function with the given parameters"""
    width = float(width)
	# I added the background parameter in this function 
    return lambda x: height * exp(-(((x - center)/width)**2/2)) + background

def moments(data):
    """ Returns (height, x, width),
    the Gaussian parameters of a 1D distribution by calculating its moments"""
    total = data.sum()
    X = indices(data.shape)
    x = (X * data).sum()/total
    width = sqrt(abs((arange(data.size) - x)**2*data).sum()/data.sum())
    height = data.max()
    """ We can determine background value by median of background array """
    backgroundArray1 = data[0 : int((len(data) - 4*int(width))/2)]
    backgroundArray2 = data[int((len(data) + 4*int(width))/2) : len(data)]
    backgroundArray = list(backgroundArray1) + list(backgroundArray2)
    backgroundArray = array(backgroundArray)
    background = median(backgroundArray)
    return height, x, width, background

def fitgaussian(data):
    """ Return (height, x, width)
    the gaussian parameters of a 1D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: ravel(gaussian(*p)(*indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params, maxfev=200)
    return p

def interpo(image, delta):

    '''
    Linear interpolate function
    input = image
    output = image
    delta = shifting value in pixels units.
    '''
    
    yp = np.zeros(len(image))
    for i in range(len(image)):
        pos = delta[i] + i
        if i == len(image)-1:
            #print 'here' #2013-02-19 cksim
            yp[i] = image[pos]
        else:
            yp[i] = image[np.floor(pos)] + (image[np.floor(pos)+1] - image[np.floor(pos)] ) * \
            ((pos) - np.floor(pos)) / (np.floor(pos)+1 - np.floor(pos))

        #print 'i, delta, py', i, delta[i], yp[i]
        
    return yp

def nearest(image, delta):

    '''
    Nearest interpolate function
    input = image
    output = image
    delta = shifting value in pixels units.
    '''

    yp = np.zeros(len(image))
    for i in range(len(image)):
        pos = delta[i] + i
        if i == len(image)-1:
            yp[i] = image[pos]
        else:
            yp[i] = image[round(pos)]

        #print 'i, delta, py', i, delta[i], yp[i]
        
    return yp


def transform_p(img, delta):

    '''
    Function to transform image using linear interpolate
    input = image
    delta: delta shift values in pixels units.
    output = transform image
    '''

    ny, nx = img.shape
    timg = np.zeros([ny,nx])
    for i in range(ny):
        # Linear interpolate
        timg[i,:] = interpo(img[i,:], delta[i])  
        # Nearest interpolate
        #f = interpolate.interp1d(np.arange(nx), img[i,:], kind='nearest', axis=0, \
        #                     copy=True, bounds_error=False, fill_value=True)
        #timg[i,:] = f(np.arange(nx) + delta[i])
        #timg[i,:] = nearest(img[i,:], delta[i])
        
    return timg

def transform_strip(img, delta):

    '''
    Function to transform image using linear interpolate
    input = image
    delta: delta shift values in pixels units.
    output = transform image
    '''

    ny, nx = img.shape
    timg = np.zeros([ny,nx])
    delta_s = np.zeros([ny,nx])
    for j in range(ny):
        delta_s[j] = delta 

    for i in range(ny):
        # Linear interpolate
        timg[i,:] = interpo(img[i,:], delta_s[i])  

    return timg

def reidentify(image, npix, lxx):

    '''
    - Identify the center of emission lines using
    Gaussian fitting function gaussfunction()

    - Strip image in this function is seperated into 6 1d
    spectrum --> these are 6 centers of each emission line
    will be found.

    - npix = range of each emission lines, the range will be
    applied to each line as [center-npix, center+npix]
    '''

    ny, nx = image.shape 
    line = np.arange(0,nx)
    #print 'colum, row = ', ny, nx

    #x = np.arange(0, nx)
    step = int(ny/6)  # Seperate the strip image into 6 1d spectrum
    #print 'step', step

    # An arbitrary constant number to show the sum from 10 lines of strip
    # image with different value from y-axis.
        
    k = 20000   
    peak = np.zeros(len(lxx))

    #fig = plt.figure(1, figsize=(14,7))
    #a1 = fig.add_subplot(211, title='strip')
    #a2 = fig.add_subplot(212, title='line')


    # Reidentify lines
    
    peak1 = [[] for _ in range(len(lxx))]
    print 'npix', npix
    for st in range(0, len(lxx)):
        for j in range(ny/step): # range(1,9):
            for i in range(0, step):
               line = line + image[:,:][i+ j*step]  # whole the row with 2048 pixels
            #print 'line', line
            # define figure object 
                               
            # Finding center using Gaussian function
            
            line1 =  line[lxx[st]- npix[st]: lxx[st] + npix[st]]/step  # identify range of each line

            #print 'emission line', line1
               
            para = fitgaussian(line1)
            '''
            print 'parameters'
            print 'height', para[0]
            print 'center', para[1]
            print 'width', para[2]
            '''

            gaussfunction = gaussian(*para)

            fitting = gaussfunction(*indices(line1.shape))
            #ff = plt.figure(11, figsize=(14,7))
            #a = ff.add_subplot(111)
            #a.plot(line1, 'k', fitting, 'r')
            #ff.savefig(TDIR+ str(lxx[st]) + str(j)+ str(i)+ '.png')
            #plt.close('all')
            #plt.show()
            
            peak[st] = para[1]
            peak1[st].append(peak[st] + lxx[st]- npix[st])
                        
    return peak1

def peak_reidentify(image, npix, lxx):

    '''
    - Identify the center of emission lines using
    maximum value of each emission lines

    - Strip image in this function is seperated into 6 1d
    spectrum --> these are 6 centers of each emission line
    will be found.

    - npix = range of each emission lines, the range will be
    applied to each line as [center-npix, center+npix]
    '''
    
    ny, nx = image.shape 
    line = np.arange(0,nx)
    #print 'colum, row = ', ny, nx

    #x = np.arange(0, nx)
    step = int(ny/6)
    #print 'step', step

    k = 20000
    peak = np.zeros(len(lxx))
    linewidth = np.zeros(len(lxx))

    #fig = plt.figure(1, figsize=(14,7))
    #a1 = fig.add_subplot(211, title='strip')
    #a2 = fig.add_subplot(212, title='line')


    # Reidentify lines
    
    peak1 = [[] for _ in range(len(lxx))]
    width = [[] for _ in range(len(lxx))]
    
    for st in range(0, len(lxx)):
        for j in range(ny/step): # range(1,9):
            for i in range(0, step):
               line = line + image[:,:][i+ j*step]
            #print 'line', line
            # define figure object 
                               
            # Finding center using Gaussian function

            line1 =  line[lxx[st]- npix: lxx[st] + npix]/step
            total = line1.sum()
            X = indices(line1.shape)
            x = (X * line1).sum()/total

            linewidth[st] = sqrt(abs((arange(line1.size) - x)**2*line1).sum()/line1.sum())
            width[st].append(linewidth[st])
            
            peak[st] = np.where([line[lxx[st]- npix: lxx[st] + npix]/step == \
                             max(line[lxx[st]- npix: lxx[st] + npix]/step)])[1][0]
            peak1[st].append(peak[st] + lxx[st]- npix)
            

    return peak1, width

def residual(peak, vmedium):

    '''
    - Residual plot of delta x in pixels unit
    - Delta x = center of each emission line - medium_center
    emission line
    '''

    fig = plt.figure(1, figsize=(10,8))
    a = fig.add_subplot(111)

    for i in range(len(peak)):
        delta_x_p = np.array(peak[i]) - vmedium[i] #np.median(peak[i]) #peak[i][2] #np.average(peak[i])
        #print 'line', peak[i], peak[i][2]
        #a.plot(np.zeros(len(peak[i])) + peak[i][4], delta_x_p, 'k.')

    #a.set_xlim([0, nx])
    #a.set_ylim([-0.7, 0.7])

    #plt.title('Distortion correction')
    #plt.ylabel('Residual (pixels)')
    #plt.xlabel('X-positions (pixels)')

    #plt.show()


def linefitting(img, peak, lxx):

    '''
    - Fitting emission line using polynomial fitting
    - The second order poly fitting is applied to 6 peaks
    possition found from reidentify() function.
    '''
    ny, nx = img.shape
    step = ny/6
    
    fit_x = [[] for _ in range(len(lxx))]
    middle = [[] for _ in range(len(lxx))]
    for f in range(0, len(peak)):
        
        yy = range(5, ny, step)

        col = range(0, ny)
        coeff = np.polynomial.polynomial.polyfit(yy, peak[f], 2)
        x_fitting = np.polynomial.polynomial.polyval(col, coeff)

        #print 'fitting of x peak', x_fitting, len(x_fitting)

        # shifting the row to correct distortion.
        fit_x[f] = x_fitting
        middle[f] = np.average(fit_x[f])
        #print 'x fitting', x_fitting
        #print 'middle', middle[f]
        
        # Draw the strip image         
        #f2 = plt.figure(2, figsize=(14,3))
        #ax2 = f2.add_subplot(111)
         
        #z1, z2 = ip.zscale(img)         
        #ax2.imshow(img, cmap='gray',vmin=z1, vmax=z2, aspect='auto')
        #ax2.plot(x_fitting, col, 'r-', peak[f], yy , 'bo', linewidth=2)
        #ax2.set_xlim(0,nx)
        #ax2.set_ylim(-10,ny+10)
        
    #plt.show()         

    
    
    return fit_x, middle

# Find the nearst values

def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def calibration(aperture, stripfile, lxx, lwave, datpath='', datapath='', outputname=''):

    imgv1, ahdr = ip.readfits(stripfile)
    col, line = imgv1.shape
    img = imgv1[5:(line-5), :]
    ny, nx = img.shape 

    # Identify emission lines
    # Make range of the emission line (When we have identified the imission line
    # peak, then range of the line will be [peak-npix, peak+npix]

    npix = 13  # This number will be updated or edit by appropriate formular


    # This will be identify the peak of the emission line
    # from the range has applied for the line

    i_peakline, lwidth = peak_reidentify(img, npix, lxx)

    npix = np.zeros(len(lxx))
    for i in range(len(lxx)):
       npix[i] = round(np.average(lwidth[i]))*2 + 1
    
    print 'npix', npix
    
    ilxx = np.zeros(len(lxx))
    for i in range(len(lxx)):
        ilxx[i] = i_peakline[i][2]

    #print 'identify lxx', ilxx 

    # Using Gaussian fitting to find the center of the line.
    peakline = reidentify(img, npix, ilxx)

    # Check peak line stable of Gaussian profile
    for i in range(len(peakline)):
        for j in range(len(peakline[i])):
            if (peakline[i][j] - np.median(peakline[i])) > 3:
               peakline[i][j] =  np.median(peakline[i])
            if (peakline[i][j] - np.median(peakline[i])) < -3:
               peakline[i][j] =  np.median(peakline[i])
    
    # Using polynomial to fit the emission lines

    fit_x, vmiddle = linefitting(img, peakline, lxx)
    #print 'fit x', fit_x
    #print 'value middle', vmiddle
    
    # Test distortion of emission lines
    residual(peakline, vmiddle)

    # Shifting rows with delta x from identify lines and fitting.

    xx = np.zeros(len(lwave))
    for i in range(0, len(lwave)):
        xx[i] = peakline[i][2]  # Sum of 10 lines in the middle stripfile
        #print 'center of identify lines', xx

    line_xx = list(xx)


    # Correct distortion correction by delta_x values

    k = [[] for _ in range(ny)]
    #print 'k', k, len(k)
    for i in range(len(k)):
        for c in range(len(fit_x)):
            k[i].append(fit_x[c][ny-1]-fit_x[c][i])
            #print 'k index', i, c

    #print 'k -test', k

    delta_x = np.zeros(ny)
    for i in range(ny):
        delta_x[i] = k[i][0] #np.average(k[i])

    #print 'delta x', delta_x

    '''
    f = open('distor_table_t.dat','w')
    A = np.arange(len(k))
    for i in A:
        temp = "%.4d %.6f \n" %(i, delta_x[i])
        f.write(temp)
    f.close()
    '''

    xx = line_xx  # Using line_xx as list file

    # Wavelength calibration

    npixel = range(nx)  # Array with len of 2047 pixels

    wavel = np.array(lwave) # Make array of wavelength

    
    # Polynomial function
    coeff = np.polynomial.polynomial.polyfit(wavel, xx, 2)

    #print 'cofficient fitting before transform', coeff


    # Finding lambda min and max from coefficient fitting
    # X = f(lamda) = a_0 + a_1 * X + a_2 * X^2
    
    # Second order 
    p_min = [coeff[2], coeff[1], coeff[0]]
    value_min = np.roots(p_min)

    #print 'lamda min values', np.roots(p_min)

    p_max = [coeff[2], coeff[1], coeff[0] - 2047]
    value_max = np.roots(p_max)

    lamda_min = find_nearest(value_min, min(wavel))
    lamda_max = find_nearest(value_max, max(wavel))

    # Position array
    position = np.where(value_max == lamda_max)[0]

    #print 'lamda max values', np.roots(p_max)

    # Check in case of complex values
    
    if (np.iscomplexobj(lamda_min)):
        lamda_min = np.real(lamda_min)
    elif (np.iscomplexobj(lamda_max)):
        lamda_max = np.real(lamda_max)

    #print 'position', position
    #print 'lambda min and max', lamda_min, lamda_max

    ptlamda = np.zeros(nx)
    for i in range(nx):
        pt = [coeff[2], coeff[1], coeff[0] - i]
        if (np.iscomplexobj(np.roots(pt))):
            temp = np.real(np.roots(pt))
            ptlamda[i] = temp[position]
        else:
            ptlamda[i] = np.roots(pt)[position]
    #print 'ptlamda', ptlamda    
      
    '''
    
    # Or using 4th order poly function
    # Polynomial function
    coeff = np.polynomial.polynomial.polyfit(wavel, xx, 4)

    print 'cofficient fitting before transform', coeff

    p_min = [coeff[4], coeff[3], coeff[2], coeff[1], coeff[0]]
    value_min = np.roots(p_min)
    

    p_max = [coeff[4], coeff[3], coeff[2], coeff[1], coeff[0] - 2047]
    value_max = np.roots(p_max)
    
    lamda_min = find_nearest(value_min, min(wavel))
    lamda_max = find_nearest(value_max, max(wavel))

    print 'position', np.where(value_min == lamda_min)[0], np.where(value_max == lamda_max)[0]    
    print 'lambda min and max', lamda_min, lamda_max
          
    
    #Solve 4th oder poly function to find the output solution
    #for lambda min and max values
    
    ptlamda = np.zeros(nx)
    for i in range(nx):
         pt = [coeff[4], coeff[3], coeff[2], coeff[1], coeff[0] - i]
         ptlamda[i] = np.roots(pt)[np.where(value_min == lamda_min)[0][0]]
   '''

    '''
    Plot the fitting before transform
    '''
         
    #fig = plt.figure(3, figsize=(10,8))
    #a = fig.add_subplot(111)
    #a.plot(ptlamda, range(nx), 'r', wavel, xx, 'ko')
    #plt.xlabel('Wavelength [microns]')
    #plt.ylabel('X-position [pixels]')
    #plt.legend(['Polyfit', 'X-before transform'])
    #plt.show()


    # Finding linear equation from poly fitting function
    # g(lamda) = b_0 + b1 * lamda
    # Solve equation with two varible
    # b_0 + b_1 x lamda_min = 0
    # b_0 + b_1 x lamda_max = 2047

    # Define linear function
    def func(b):
        f = [b[0] + b[1] * lamda_min, b[0] + b[1]*lamda_max - 2047]
        return f

    b = optimize.fsolve(func, [0, 0])

    #print 'linear coeff', b

    # Linear equation has value of b[0] and b[1]
    # X' = b[0] + b[1] * lambda

    # Delta lambda wavelength

    delta_lamda = (lamda_max - lamda_min) / 2047
    #print 'delta_lamda', delta_lamda

    # Linear wavelength from the lambda min and max values

    linear_wave = np.zeros(nx)
    for i in range(nx):
        linear_wave[i] = lamda_min + i*delta_lamda

    #lwave = np.linspace(lamda_min, lamda_max, nx)
    #print 'linear wavelength', linear_wave

    # Create text file with i=0-2047, lamda, x, x'

    x_fit = np.zeros(len(linear_wave))  # x position values calculated from the 4th order function.
    lx_fit = np.zeros(len(linear_wave)) # x position values calculated from linear function.
    
    # Second order
    for i in range(len(linear_wave)):
        x_fit[i] = coeff[0] + coeff[1] * linear_wave[i] + coeff[2] * pow(linear_wave[i],2)
        lx_fit[i] = b[0] + b[1] * linear_wave[i]
    '''
    # 4nd order
    
    for i in range(len(linear_wave)):
        x_fit[i] = coeff[0] + coeff[1] * linear_wave[i] + coeff[2] * pow(linear_wave[i],2) + \
                   coeff[3] * pow(linear_wave[i],3) + coeff[4] * pow(linear_wave[i],4) 
                
        lx_fit[i] = b[0] + b[1] * linear_wave[i]
    '''
    # Transform x to x', delta is the values have to be shifted to convert 4th order
    # poly function to linear function

    delta = x_fit - lx_fit
    #print 'delta x', delta

    # Combine distortion correction and wavelength calbration
    # Delta_s are the values have to be applied to the transform
    # function.

    delta_s = np.zeros([ny,nx])
    for j in range(ny):
        if delta_x[j] < 0:
            delta_s[j] = delta + delta_x[j] 
        if delta_x[j] > 0:
            delta_s[j] = delta - delta_x[j] 
  
     
    f = open(datpath + 'wavemap_H_02_ohline.' + str(aperture) + '.dat','w')
    A = np.arange(len(linear_wave))
    for i in A:
        temp = "%.4d %.6f %.6f %.6f %.6f \n" %(i, linear_wave[i], x_fit[i], lx_fit[i], delta[i])
        f.write(temp)
    f.close()
    

    # x-position after transform
    #print 'xx', xx
    lxx_t = np.zeros(len(lxx))
    for i in range(len(lxx)):
        lxx_t[i] = xx[i] - delta[lxx[i]]
        #print 'delta', delta[lxx[i]]
    #print 'lxx transform', lxx_t
    
    tstrip = transform_p(img, delta_s) # Tranfsorm function
    #z1, z2 = ip.zscale(tstrip)   
    #plt.imshow(tstrip, cmap='hot',vmin=z1, vmax=z2, aspect='auto')
    #plt.show()

    thdr = ahdr.copy()
    thdr.update('TRN-TIME', time.strftime('%Y-%m-%d %H:%M:%S'))
    # WCS header ========================================
    thdr.update('WAT0_001', 'system=world')
    thdr.update('WAT1_001', 'wtype=linear label=Wavelength units=microns units_display=microns')
    # wavelength axis header ============================= 
    thdr.update('CTYPE1', 'LINEAR  ')
    thdr.update('LTV1',   1)
    thdr.update('LTM1_1', 1.0)
    thdr.update('CRPIX1', 1.0)
    thdr.update('CDELT1', delta_lamda)
    thdr.update('CRVAL1', min(linear_wave))
    thdr.update('LCOEFF1', b[0])
    thdr.update('LCOEFF2', b[1])

    ip.savefits(datapath + outputname + str(aperture) + '.fits', tstrip, header=thdr)
    np.savetxt(datpath + outputname + str(aperture) + '.wave', linear_wave)
    ip.savefits(datpath + 'wavemap_H_02_ohlines.' +  str(aperture) + '.fits', delta_s, thdr)

    return tstrip, delta_s, coeff, b, linear_wave, lxx_t

def check(stripfile, aperture, tstrip, lxx_t, lwave, b, linear_wave, pngpath='', datpath='', outputname=''):

    # Input images
    img, ahdr = ip.readfits(stripfile)
    ny, nx = img.shape
    

    wavel = np.array(lwave)
    npix = 9 # Range pixel of each identify lines
    
    # The belove steps are used to check the transform processes.

    ti_peakline, tlwidth = peak_reidentify(tstrip, npix, lxx_t)

    npix = np.zeros(len(lxx_t))
    for i in range(len(lxx_t)):
       npix[i] = round(np.average(tlwidth[i]))*2 + 1

    tilxx = np.zeros(len(lxx_t))
    for i in range(len(lxx_t)):
        tilxx[i] = ti_peakline[i][2]

    #print 'identify lxx', tilxx 

    peak_w = reidentify(tstrip, npix, list(tilxx))

    # Check peak line stable of Gaussian profile
    for i in range(len(peak_w)):
        for j in range(len(peak_w[i])):
            if (peak_w[i][j] - np.median(peak_w[i])) > 3:
               peak_w[i][j] =  np.median(peak_w[i])
            if (peak_w[i][j] - np.median(peak_w[i])) < -3:
               peak_w[i][j] =  np.median(peak_w[i])

    # Using polynomial to fit the emission lines

    tfit_x, tvmiddle = linefitting(tstrip, peak_w, lxx_t)

    # Test distortion of emission lines
    residual(peak_w, tvmiddle)

    # txx values transform

    step = int(ny/6)
    txx = np.zeros(len(lxx_t))
    for i in range(0, len(lxx_t)):
        txx[i] = peak_w[i][2]  # Sum of 10 lines in the middle stripfile

    #print 'center of identify lines after transform', txx

    # Second order
    coeff2 = np.polynomial.polynomial.polyfit(wavel, txx, 2)
    #print 'cofficient fitting linear wave', coeff2

    tp_min = [coeff2[2], coeff2[1], coeff2[0]]
    tvalue_min = np.roots(tp_min)

    tp_max = [coeff2[2], coeff2[1], coeff2[0] - 2047]
    tvalue_max = np.roots(tp_max)

    tlamda_min = find_nearest(tvalue_min, min(wavel))
    tlamda_max = find_nearest(tvalue_max, max(wavel))

    #print 'position', np.where(tvalue_min == tlamda_min)[0], np.where(tvalue_max == tlamda_max)[0]    
    #print 'lambda min and max', tlamda_min, tlamda_max

    # Position array
    position = np.where(tvalue_max == tlamda_max)[0]
    
    # Check in case of complex values
    
    if (np.iscomplexobj(tlamda_min)):
        lamda_min = np.real(tlamda_min)
    elif (np.iscomplexobj(tlamda_max)):
        lamda_max = np.real(tlamda_max)
        
    tlamda = np.zeros(nx)
    for i in range(nx):
        pt = [coeff2[2], coeff2[1], coeff2[0] - i]
        if (np.iscomplexobj(np.roots(pt))):
            temp = np.real(np.roots(pt))
            tlamda[i] = temp[position]
        else:
            tlamda[i] = np.roots(pt)[position]

    

    # 4nd order
    '''
    coeff2 = np.polynomial.polynomial.polyfit(wavel, txx, 4)
    print 'cofficient fitting linear wave', coeff2

    tp_min = [coeff2[4], coeff2[3], coeff2[2], coeff2[1], coeff2[0]]
    tvalue_min = np.roots(tp_min)

    tp_max = [coeff2[4], coeff2[3], coeff2[2], coeff2[1], coeff2[0] - 2047]
    tvalue_max = np.roots(tp_max)

    tlamda_min = find_nearest(tvalue_min, min(wavel))
    tlamda_max = find_nearest(tvalue_max, max(wavel))

    print 'position', np.where(tvalue_min == tlamda_min)[0], np.where(tvalue_max == tlamda_max)[0]    
    print 'lambda min and max', tlamda_min, tlamda_max
    
    print 'lambda min and max after transform', tlamda_min, tlamda_max

    tlamda = np.zeros(nx)
    for i in range(nx):
         p = [coeff2[4], coeff2[3], coeff2[2], coeff2[1], coeff2[0] - i]
         tlamda[i] = np.roots(p)[np.where(tvalue_min == tlamda_min)[0]]
    '''

    #fig = plt.figure(4, figsize=(10,8))
    #a = fig.add_subplot(111)
    #a.plot(tlamda, range(nx), 'r', wavel, txx, 'ko')
    #a.plot(wavel, txx, 'ko')
    #plt.ylabel('X-position [pixels]')
    #plt.xlabel('Wavelength [microns]')
    #plt.legend(['Polyfit', 'X-after transform'])
    #plt.show()

    # Test file of results

    # wavelength calculate from the linear equation from the
    # line lists
    f_linear = np.zeros(len(wavel))
    for i in range(len(wavel)):
        f_linear[i] = b[0] + b[1] * wavel[i]
    #print 'linear coeff', b

    
    f = open(datpath + outputname + str(aperture) + '.dat','w')
    A = np.arange(len(f_linear))
    for j in range(len(peak_w[0])):
       for i in A:
          temp = "%.6f %.6f %.6f %.6f \n" %(wavel[i], f_linear[i], np.array(peak_w[i][j]), \
                                              np.array(peak_w[i][j]) - f_linear[i])
          f.write(temp)

    f.close()
           
    # Draw the strip image         
    f6 = plt.figure(None, figsize=(14,6))
    ax6= f6.add_subplot(311, title= outputname + str(aperture))
    z1, z2 = ip.zscale(tstrip)   
    ax6.imshow(img, cmap='hot',vmin=z1, vmax=z2, aspect='auto')
    ax6.set_xlim(0,nx)
    ax6.set_ylim(-10,ny+10)
    plt.ylabel('Y [pixels]')
    #plt.xlabel('X [pixels]')


    #f7 = plt.figure(7, figsize=(14,3))
    ax7= f6.add_subplot(312)
    ax7.imshow(tstrip, cmap='hot',vmin=z1, vmax=z2, aspect='auto')
    #ax7.plot(range(nx), img_shift[ny/2,:], 'k.', trxx, tstrip[ny/2,:], 'r.')
    #plt.legend(['before transform', 'after transform'])
    ax7.set_xlim(0,nx)
    ax7.set_ylim(-10,ny+10)
    #plt.xlabel('X [pixels]')


    #f8 = plt.figure(8, figsize=(14,3))
    ax8 = f6.add_subplot(313, title='')
    ax8.plot(linear_wave, tstrip[ny/2,:], 'b')

    #ax4.set_xlim(0, len(tstrip[1,:]))
    #ax4.set_ylim(-10,ny+10)
    #ax4.set_xlim(min(w_fitting), max(w_fitting))
    ax8.set_xlim(min(linear_wave), max(linear_wave))
    plt.xlabel('Wavelength [microns]')

    plt.savefig(pngpath + outputname + str(aperture) + '.png')
    plt.close()

    return coeff2

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run >>>>>>>>>>>>>>>>>>>>>>>>
'''

DIR = './FITS/'
TDIR = './PNG/Gaussfit/'



aperture = range(23)

for i in range(23):

    #Strip file input
    if aperture[i] < 10:
        stripfile = DIR + 'IGRINS_H_ohlines.00' + str(aperture[i]) + '.fits'
        # Referent lines
        lxx = pixel[aperture[i]]
        lwave = wavelength[aperture[i]]
    else:
        stripfile = DIR + 'IGRINS_H_ohlines.0' + str(aperture[i]) + '.fits'
        # Referent lines
        lxx = pixel[aperture[i]]
        lwave = wavelength[aperture[i]]

    # Read image size 

    #img, header = ip.readfits(stripfile)
    #ny, nx = img.shape 

    t_image, delta_shift, coeff_transform, linear_par, linear_wave, lxx_tr = calibration(aperture[i], stripfile, lxx, lwave)
    check(stripfile, aperture[i], t_image, lxx_tr, lwave, linear_par, linear_wave)

    # Plot results
    final = plt.figure(1, figsize=(16,6))
    ax = final.add_subplot(111)
    a = np.genfromtxt('./DAT/' + 'result_linear_' + str(aperture[i]) + '.dat')
    ax.plot(a[:,0], a[:,3], 'k.')
    plt.xlabel('Wavelength [um]')
    plt.ylabel('Delta X [pixels]')
    plt.title('Distortion correction and wavelength calibration')
    
    x_majorLocator = MultipleLocator(0.02)
    x_majorFormatter = FormatStrFormatter('%0.3f')
    x_minorLocator = MultipleLocator(0.004)
    y_majorLocator = MultipleLocator(0.2)
    y_majorFormatter = FormatStrFormatter('%0.1f')
    y_minorLocator = MultipleLocator(0.04)

    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_major_formatter(x_majorFormatter)
    ax.xaxis.set_minor_locator(x_minorLocator)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_major_formatter(y_majorFormatter)
    ax.yaxis.set_minor_locator(y_minorLocator)
    

plt.show()

'''





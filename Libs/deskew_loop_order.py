'''
Test fine-tuning result
'''



import scipy
import pyfits
import glob, time, os 
import time, os
import itertools 
import numpy as np 
import matplotlib.pyplot as plt 
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.interpolate import griddata 
import matplotlib.cm as cm
import basic as ip
#import extract
from numpy import poly1d 
import math
import pylab
import pyfits
from numpy import *
from matplotlib.pylab import *
from scipy import *
from scipy.optimize import curve_fit, leastsq
from scipy import optimize
from scipy.stats import nanmean


#band = 'H'

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
    try: backgroundArray1 = data[0 : int((len(data) - 4*int(width))/2)]
    except ValueError:
        return
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

#------------------------------------------------------------------------------------------------


def deskew_plot(band, target_path, name='', start=0, end=23):

    if band=="H":
        start = 0
        end = 23
    else:
        start = 0
        end = 20
    
    number = range(start + 1, end + 1)
    order =  range(start + 1, end + 1)
    for i in range(start, end):
	if i <=8:
		order[i] = '00' + str(number[i])
	else:
		order[i] = '0' + str(number[i])

    step = 0
    
    for index in range(start, end):
        print 'index', index
        step = step + 1.
        img, header = ip.readfits(target_path + name + '.' + order[index]+  '.fits')
        ny, nx = img.shape
        #print 'image', img, len(img)
        #x = arange(0,nx/16)
        x = np.linspace(0, 1, nx/100)  # Change by Huynh Anh

        # Plot column

        m = 0
        k = 100  # Change by Huynh Anh

        y_low = np.zeros(nx/k)
        y_up = np.zeros(nx/k)


        for j in range(0, nx/k):
           col = np.zeros(len(img))
           for i in range(0,len(img)):
              col[i] = np.average(img[i][m:m+k])
           m = m+k
           #print 'range', m, m+k

           # derivative
           xdata = range(0,len(col))
           #polycoeffs = scipy.polyfit(xdata, col, 10)
           #print polycoeffs
           #yfit = scipy.polyval(polycoeffs, xdata)
           #print 'col', col
           deri = np.gradient(col) # Consider this derivation function.
           #print 'derivative', deri, len(deri)
           
           try:
            # Find boundary 1
              bound1 = deri[0:len(deri)/2]
              para1 = fitgaussian(bound1)
           except TypeError:
                pass
          
           try:
               
               # Find boundary 2
               bound2 = deri[(len(deri)/2): len(deri)] * (-1)
               para2 = fitgaussian(bound2)
           except TypeError:
                pass
           
           try:
               y_low[j] = para1[1]
               y_up[j] = para2[1] + len(deri)/2
           except UnboundLocalError:
               pass
            
           
           #y_low[j] = np.where([deri == max(deri)])[1][0]
           #y_up[j]  = np.where([deri == min(deri)])[1][0]


        #print 'y low', y_low, len(y_low), np.std(y_low)
        #print 'y upp', y_up, len(y_up), np.std(y_up)

        # Constant value for y low and y up

        ylow_value = np.zeros(len(y_low))
        for i in range(0, len(ylow_value)):
            ylow_value[i] = np.round(nanmean(y_low))

        yup_value = np.zeros(len(y_up))
        for i in range(0, len(yup_value)):
            yup_value[i] = np.round(nanmean(y_up))

        #print 'ylow', ylow_value

        # Plot

        #f = plt.figure(num=1, figsize=(12,7))
        #ax1 = f.add_subplot(111)
        #ax1.imshow(img, cmap='hot')

        #ax.set_xlim(-20,nx+20)
        #ax.set_ylim(-20,ny+20)

        #ax1.set_xlim(0,nx)
        #ax1.set_ylim(0,ny)

        #f2 = plt.figure(num=2, figsize=(20,10))
        #ax2 = f2.add_subplot(111)

        #ax2.plot(x, y_low, 'k-.', x, y_up, 'r-.', linewidth=2, markersize=3)
        #ax2.plot(x, ylow_value, 'k-', x, yup_value, 'r-', linewidth=1)

	    up = (y_up - nanmean(y_up))
	    low = (y_low - nanmean(y_low))
        up[up>=5] = np.nan
        up[up<-5] = np.nan
        low[low>=5] = np.nan
        low[low<-5] = np.nan

        f3 = plt.figure(num=3, figsize=(8,6))
        ax3 = f3.add_subplot(111)
        ax3.plot(x + step, (up), 'bx', linewidth=1, markersize=3)
        ax3.plot(x + step, (low), 'rx', linewidth=1, markersize=3)
       #ax3,plot((x + step)[up[(up<=5) & (up>=-5)]], up[(up<=5) & (up>=-5)], 'b-.', linewidth=1, markersize=3) 
       #ax3,plot((x + step)[low[(up<=5) & (low>=-5)]], low[(low<=5) & (low>=-5)], 'r-.', linewidth=1, markersize=3)
	

        x_majorLocator = MultipleLocator(1.)
        x_majorFormatter = FormatStrFormatter('%d')

        ax3.set_xlim(0,24)
        ax3.set_ylim(-5,5)
        ax3.xaxis.set_major_locator(x_majorLocator)
        ax3.xaxis.set_major_formatter(x_majorFormatter)

        plt.legend(['Upper Residual Edge', 'Lower Residual Edge'])
        plt.title('Residual Edges')
        plt.xlabel('Order Number')
        plt.ylabel('Residual (pixel)')
        '''
        f4 = plt.figure(num=4, figsize=(20,10))
        ax4 = f4.add_subplot(111)
        ax4.plot(x + step, (y_low - nanmean(y_low)), 'k-.', linewidth=1, markersize=3)

        x_majorLocator = MultipleLocator(1.)
        x_majorFormatter = FormatStrFormatter('%d')
        ax4.xaxis.set_major_locator(x_majorLocator)
        ax4.xaxis.set_major_formatter(x_majorFormatter)
        ax4.set_xlim(0,24)

        plt.title('Lower Residual Edge')
        plt.xlabel('Order Number')
        plt.ylabel('Residual')
        '''
    plt.show()


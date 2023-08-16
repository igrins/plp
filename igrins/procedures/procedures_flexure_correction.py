from __future__ import print_function

import numpy as np
from scipy.ndimage import median_filter, zoom, gaussian_filter1d
from scipy.signal import fftconvolve
from .estimate_sky import estimate_background, get_interpolated_cubic
from ..procedures.destriper import destriper, stack128, stack64, get_stripe_pattern64
from ..igrins_libs.resource_helper_igrins import ResourceHelper
from astropy.io import fits
import glob
import copy

from numpy.polynomial import Polynomial

from scipy.ndimage import median_filter, gaussian_filter


#Use a series of median filters to isolate the sky lines while ignoring everything else
def isolate_sky_lines(data):
	data = data - np.nanmedian(data, 0) #Remove pattern
	data = data- np.nanmedian(data, 1)[:,np.newaxis]	
	data = median_filter(data - median_filter(data, [1, 35]), [11, 1]) #median filters to try to isolate sky lines
	data = data - median_filter(data, [9,19])

	std = np.std(data) #Zero out negative residuals from science signal
	data[data < -std] = 0.
	return data


def get_date_and_band(obsset):
	date, band = obsset.get_resource_spec()
	return date, band

def get_exptime(obsset):
	return obsset.recipe_entry['exptime']


def roll_along_axis(array_to_correct, correction, axis=0): #Apply flexure correction by numpy rolling along an axis and averaging between two rolled arrays to account for sub-pixel shifts
    axis = int(axis)
    integer_correction = np.round(correction) #grab whole number component of correction
    fractional_correction = correction - float(integer_correction) #Grab fractional component of correction (remainder after grabbing whole number out)
    rolled_array =  np.roll(array_to_correct, int(integer_correction), axis=axis) #role array the number of pixels matching the integer correction
    if fractional_correction > 0.: #For a positive correction
        rolled_array_plus_one = np.roll(array_to_correct, int(integer_correction+1), axis=axis) #Roll array an extra one pixel to the right
    else: #For a negative correction
        rolled_array_plus_one = np.roll(array_to_correct, int(integer_correction-1), axis=axis) #Roll array an extra one pixel to the left
    corrected_array = rolled_array*(1.0-np.abs(fractional_correction)) + rolled_array_plus_one*np.abs(fractional_correction) #interpolate over the fraction of a pixel
    return corrected_array


def cross_correlate(reference, data, zoom_amount=1000, maximum_pixel_search=10):
	interp_order = 1
	fx1 = zoom(np.nansum(reference, 0), zoom_amount, order=interp_order) #Zoom and collapse into 1D
	#fy1 = zoom(np.nansum(reference, 1), zoom_amount, order=interp_order)
	fx2 = zoom(np.nansum(data, 0), zoom_amount, order=interp_order) #Zoom and collapse into 1D
	#fy2 = zoom(np.nansum(data, 1), zoom_amount, order=interp_order)
	# fx2 = bn.nansum(zoom(data, [1, self.zoom_amount], order=1), 0) #Zoom and collapse into 1D
	# fy2 = bn.nansum(zoom(data, [self.zoom_amount, 1], order=1), 1)
	fx1[np.isnan(fx1)] = 0 #Zero out remaining nan pixels
	#fy1[np.isnan(fy1)] = 0
	fx2[np.isnan(fx2)] = 0 #Zero out remaining nan pixels
	#fy2[np.isnan(fy2)] = 0
	fft_result_x = fftconvolve(fx1, fx2[::-1]) #Perform FFIT cross correlation only in x and y
	#fft_result_y = fftconvolve(fy1, fy2[::-1])
	delta_sub_pixels = 0.5*maximum_pixel_search*zoom_amount #Cut FFT cross-correlation result to be within a maximum number of pixels from zero offset, this cuts out possible extraneous minima screwing up the maximum in the FFT result characterizing the true offset
	x1 = int((fft_result_x.shape[0]/2) - delta_sub_pixels)
	x2 = int((fft_result_x.shape[0]/2) + delta_sub_pixels)
	#y1 = int((fft_result_y.shape[0]/2) - delta_sub_pixels)
	#y2 = int((fft_result_y.shape[0]/2) + delta_sub_pixels)  
	fft_sub_result_x = fft_result_x[x1:x2]
	#fft_sub_result_y = fft_result_y[y1:y2]

	#Mask out large trends
	n = len(fft_sub_result_x)
	fit_x_array = np.arange(n)
	# pfit = Polynomial.fit(fit_x_array, fft_sub_result_x, 20) #Fit a 20 order polynomial and use the first derivitive test to find the peak of the CCF while ignoring large scale trends
	# roots = pfit.deriv().roots()
	# possible_peaks = (roots[np.isreal(roots)].real -  (n/2))/zoom_amount
	# if sum(roots[np.isreal(roots)] < 1.5) > 1:
	# 	breakpoint()
	# fft_dx_result = possible_peaks[np.abs(possible_peaks) == np.min(np.abs(possible_peaks))][0] #Find peak closest to zero flexure
	#fft_sub_result_x = fft_sub_result_x - pfit(fit_x_array)
	# pfit = Polynomial.fit(fit_x_array,  fft_sub_result_y, 20)
	# roots = pfit.deriv().roots()
	# possible_peaks = (roots[np.isreal(roots)].real -  (n/2))/zoom_amount

	# fft_dy_result = possible_peaks[np.abs(possible_peaks) == np.min(np.abs(possible_peaks))][0] #Find peak closest to zero flexure

	#fft_sub_result_y = fft_sub_result_y - pfit(fit_x_array)

	fft_sub_result_mask = (fit_x_array < n*0.25) | (fit_x_array > n*0.75)
	pfit = Polynomial.fit(fit_x_array[fft_sub_result_mask], fft_sub_result_x[fft_sub_result_mask], 2)
	fft_sub_result_x = fft_sub_result_x - pfit(fit_x_array)
	# pfit = Polynomial.fit(fit_x_array[fft_sub_result_mask], fft_sub_result_y[fft_sub_result_mask], 1)
	# fft_sub_result_y = fft_sub_result_y - pfit(fit_x_array)

	find_shift_from_maximum_x = np.unravel_index(np.argmax(fft_sub_result_x), fft_sub_result_x.shape[0]) #Find pixels with strongest correlation
	#find_shift_from_maximum_y = np.unravel_index(np.argmax(fft_sub_result_y), fft_sub_result_y.shape[0])
	fft_dx_result = (find_shift_from_maximum_x[0] -  (fft_sub_result_x.shape[0]/2))/zoom_amount #Calcualte the offset from the pixels with the strongest correlation
	#fft_dy_result = (find_shift_from_maximum_y[0] -  (fft_sub_result_y.shape[0]/2))/zoom_amount
	# fft_dx_result = (x_peak -  (n/2))/zoom_amount #Calcualte the offset from the pixels with the strongest correlation
	# fft_dy_result = (y_peak -  (n/2))/zoom_amount	

	if np.abs(fft_dx_result) > 1.5: # or np.abs(fft_dy_result) > 1.5:
		breakpoint()


	return fft_dx_result#, fft_dy_result #Returns the difference in x pixels and y pixels between the reference and data frames


#Create reference frames to flexure correct everything to
#for the H and K bands. This is the first SKY frame
def set_reference_frame(obsset):
	if obsset.recipe_name == 'SKY':
		#band = get_band(obsset)
		exptime = get_exptime(obsset)
		#Grab sky frame data.  If exposures are short, stack them, otherwise just use first frame
		if exptime >= 100.0:  
			print('Sky frames exp time > 30 s.  Using the first frame.')
			hdus = obsset.get_hdus([obsset.get_obsids()[0]]) #Grab first sky frame
			data = hdus[0].data
		else:
			print('Sky frames exp time <= 30 s.  Combining all sky frames.')
			hdus = obsset.get_hdus(obsset.get_obsids())
			data_list = [hdu.data for hdu in hdus]
			data = np.sum(data_list, axis=0)
		data = isolate_sky_lines(data)
		data /= exptime #Normalize by exposure time
		hdus_out = obsset.get_hdul_to_write(([], data)) #Store processed for flexure correction sky frames
		obsset.store('FLEXCORR_FITS', data=hdus_out)


#Check 
def check_telluric_shift(obsset, datalist):
	date, band = get_date_and_band(obsset) #Grab date and band we are working in
	xr = [400,1900]
	zoom_amount = 1000.0
	interp_order=1
	maximum_pixel_search = 10
	if band == 'H':
		orders = [119, 120, 121]
	elif band == 'K':
		orders = [72, 73, 74]
	filename = glob.glob('calib/primary/'+date+'/SKY_SDC'+band+'_'+date+'*order_map.fits')[0] #Load order map
	order_map = fits.getdata(filename)
	filtered_data1 = datalist[0] - np.nanmedian(datalist[0], 0) - np.nanmedian(datalist[0], 1)[:,np.newaxis] #Cross correlate each dataframe with the first data frame in the list
	filtered_data1 -= median_filter(filtered_data1, [35,1])
	for j in range(1, len(datalist)):
		filtered_data2 =  datalist[j] - np.nanmedian(datalist[j], 0) - np.nanmedian(datalist[j], 1)[:,np.newaxis]
		filtered_data2 -= median_filter(filtered_data2, [35,1])
		dx_results = []

		for i in orders:
		    cut_data1 = copy.deepcopy(filtered_data1) #Collapse data in an order into 1D
		    cut_data2 = copy.deepcopy(filtered_data2)
		    cut_data1[~(order_map == i)] = np.nan
		    cut_data2[~(order_map == i)] = np.nan
		    collapsed_data1 = np.nansum(cut_data1[:,xr[0]:xr[1]], 0)
		    collapsed_data2 = np.nansum(cut_data2[:,xr[0]:xr[1]], 0)
		    collapsed_data1 /= np.nansum(collapsed_data1) #Normalize both frames
		    collapsed_data2 /= np.nansum(collapsed_data2)
		    fx1 = zoom(collapsed_data1, zoom_amount, order=interp_order) #Zoom and collapse into 1D
		    fx2 = zoom(collapsed_data2, zoom_amount, order=interp_order) #Zoom and collapse into 1D
		    fx1[np.isnan(fx1) | (fx1 < -np.std(fx1))] = 0 #Zero out remaining nan pixels
		    fx2[np.isnan(fx2) | (fx2 < -np.std(fx2))] = 0 #Zero out remaining nan pixels
		    fft_result_x = fftconvolve(fx1, fx2[::-1]) #Perform FFIT cross correlation only in x and y
		    delta_sub_pixels = 0.5*maximum_pixel_search*zoom_amount #Cut FFT cross-correlation result to be within a maximum number of pixels from zero offset, this cuts out possible extraneous minima screwing up the maximum in the FFT result characterizing the true offset
		    x1 = int((fft_result_x.shape[0]/2) - delta_sub_pixels)
		    x2 = int((fft_result_x.shape[0]/2) + delta_sub_pixels)
		    fft_sub_result_x = fft_result_x[x1:x2]
		    find_shift_from_maximum_x = np.unravel_index(np.argmax(fft_sub_result_x), fft_sub_result_x.shape[0]) #Find pixels with strongest correlation
		    fft_dx_result = (find_shift_from_maximum_x[0] -  (fft_sub_result_x.shape[0]/2))/zoom_amount #Calc
		    dx_results.append(fft_dx_result)
		dx = np.nanmedian(dx_results)

		#if abs(dx) > 0.0: #threshold for telluric shift
		if True:
			with open('outdata/'+date+"/telluric_shift_"+band+".csv", "a") as f: #Output flexure corrections to the textfile flexure.txt 
				f.write(str(obsset.obsids[0])+', '+str(obsset.obsids[j])+', '+str(dx)+'\n')


def estimate_flexure(obsset, data, exptime):
	#exptime = get_exptime(obsset)
	date, band = get_date_and_band(obsset) #Grab date and band we are working in
	flexure_corrected_data = [] #Create a list to store the flexure corrected data

	filename = glob.glob('calib/primary/'+date+'/SDC'+band+'_'+date+'_*.sky_flexcorr.fits')[0] #Load reference frame created with recipe_flexure_setup
	refframe = fits.getdata(filename)

	# if exptime >= 20.0: #Load mask to isolate sky lines , for long exposures estimate flexure for each frame seperately
	mask = (fits.getdata('master_calib/'+band+'-band_sky_mask.fits') == 1.0)
	refframe[~mask] = np.nan
	#for dataframe in data:
	for i in range(len(data)):
		dataframe = data[i]
		cleaned_dataframe = isolate_sky_lines(dataframe) / exptime #Apply median filters to isolate sky lines from other signal and normalize by exposure time
		#if obsset.obsids[i] == 99:
		#	breakpoint()
		cleaned_dataframe[~mask] = np.nan #Apply mask to isolate sky lines on detector
		#dx, dy = cross_correlate(refframe, cleaned_dataframe) #Estimate delta-x and delta-y difference in pixels between the reference and data frames
		dx = cross_correlate(refframe, cleaned_dataframe) #Estimate delta-x and delta-y difference in pixels between the reference and data frames

		#shifted_dataframe = roll_along_axis(dataframe, dy, axis=0)			
		#shifted_dataframe = roll_along_axis(shifted_dataframe, dx, axis=1) #Apply flexure correction
		shifted_dataframe = roll_along_axis(dataframe, dx, axis=1)	

		flexure_corrected_data.append(shifted_dataframe)
		#print('dx =', dx, 'dy =', dy)
		#breakpoint()
		with open('outdata/'+date+"/flexure_"+band+".csv", "a") as f: #Output flexure corrections to the textfile flexure.txt 
			f.write(band+', '+str(obsset.obsids[i])+', '+str(dx)+'\n')



	# else: #For short exposures, estimate flexure for all the frames stacked
	# 	mask = (fits.getdata('master_calib/'+band+'-band_limited_sky_mask.fits') == 1.0)  #(note we use a more conservative mask for short exposures)
	# 	refframe[~mask] = np.nan
	# 	stacked_dataframe =  np.sum(data, axis=0) #Stack data since exposure time is low to increase signal-to-noise on the sky lines before estimating flexure
	# 	cleaned_stacked_dataframe = isolate_sky_lines(stacked_dataframe) / (exptime * len(data)) #Apply median filters to isolate sky lines from other signal and normalize by exposure time
	# 	if obsset.obsids[0] == 99:
	# 		breakpoint()		
	# 	cleaned_stacked_dataframe[~mask] = np.nan #Apply mask to isolate sky lines on detector	
	# 	dx, dy = cross_correlate(refframe, cleaned_stacked_dataframe) #Estimate delta-x and delta-y difference in pixels between the reference and data frames
	# 	#print('dx =', dx, 'dy =', dy)
	# 	for dataframe in data:
	# 		shifted_dataframe = roll_along_axis(dataframe, dx, axis=1) #Apply flexure correction
	# 		shifted_dataframe = roll_along_axis(shifted_dataframe, dy, axis=0)
	# 		flexure_corrected_data.append(shifted_dataframe)
	# 	with open('outdata/'+date+"/flexure_"+band+".csv", "a") as f: #Output flexure corrections to the textfile flexure.txt 
	# 		obsids = ''
	# 		for obsid in obsset.obsids:
	# 			obsids = str(obsid)+ ' '
	# 		obsids = obsids[:-1] #Remove trailing space
	# 		f.write(band+', '+obsids+', '+str(dx)+', '+str(dy)+'\n')

	return flexure_corrected_data

def estimate_flexure_short_exposures(obsset, data_a, data_b, exptime):
	date, band = get_date_and_band(obsset) #Grab date and band we are working in
	filename = glob.glob('calib/primary/'+date+'/SDC'+band+'_'+date+'_*.sky_flexcorr.fits')[0] #Load reference frame created with recipe_flexure_setup
	refframe = fits.getdata(filename)
	mask = (fits.getdata('master_calib/'+band+'-band_limited_sky_mask.fits') == 1.0)  #(note we use a more conservative mask for short exposures)
	refframe[~mask] = np.nan
	combined_data = data_a + data_b - np.abs(data_a - data_b)
	cleaned_combined_data = isolate_sky_lines(combined_data) / (exptime * len(obsset.get_obsids())) #Apply median filters to isolate sky lines from other signal and normalize by exposure time
	cleaned_combined_data[~mask] = np.nan #Apply mask to isolate sky lines on detector	
	#dx, dy = cross_correlate(refframe, cleaned_combined_data) #Estimate delta-x and delta-y difference in pixels between the reference and data frames
	dx = cross_correlate(refframe, cleaned_combined_data) #Estimate delta-x and delta-y difference in pixels between the reference and data frames
	shifted_data_a = roll_along_axis(data_a, dx, axis=1) #Apply flexure correction
	#shifted_data_a = roll_along_axis(shifted_data_a, dy, axis=0)
	shifted_data_b = roll_along_axis(data_b, dx, axis=1) #Apply flexure correction
	#shifted_data_b = roll_along_axis(shifted_data_b, dy, axis=0)

	# print('OBSID:', obsset.obsids[0])
	# breakpoint()

	with open('outdata/'+date+"/flexure_"+band+".csv", "a") as f: #Output flexure corrections to the textfile flexure.txt 
		#f.write(band+', '+str(obsset.obsids[0])+', '+str(dx)+', '+str(dy)+'\n')
		f.write(band+', '+str(obsset.obsids[0])+', '+str(dx)+'\n')

	return shifted_data_a, shifted_data_b

if __name__ == "__main__":
    pass

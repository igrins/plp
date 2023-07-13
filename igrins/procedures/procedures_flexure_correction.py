from __future__ import print_function

#from collections import namedtuple

import numpy as np
#import scipy.ndimage as ni
from scipy.ndimage import median_filter, zoom, gaussian_filter1d
from scipy.signal import fftconvolve
#from ..procedures.readout_pattern_guard import remove_pattern_from_guard
from ..igrins_recipes.recipe_combine import remove_pattern
#from ..igrins_libs.storage_descriptions import FLEXCORR_FITS_DESC
from .estimate_sky import estimate_background, get_interpolated_cubic
from ..procedures.destriper import destriper, stack128, stack64, get_stripe_pattern64
from ..igrins_libs.resource_helper_igrins import ResourceHelper
from ..procedures.sky_spec import _get_combined_image, _destripe_sky

from scipy.ndimage import median_filter, gaussian_filter


def get_band(obsset):
	_, band = obsset.rs.get_resource_spec()
	return band

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



# class flexure:
# 	def __init__(self, path_to_sky_frame, path_to_sky_mask, median_filter_size=21, zoom_amount=10): #Set up initial object with a sky frame
# 		sky_data = fits.getdata(path_to_sky_frame) #Get data from sky frame
# 		sky_exptime = float(fits.getval(path_to_sky_frame, 'EXPTIME'))
# 		#sky_data -= gaussian_filter1d(sky_data, sigma=10, axis=1)
# 		sky_data -= median_filter(sky_data, size=[1,median_filter_size]) #Use a horizontal median filter to remove most background and continnuum, leaving mostly OH sky emission lines

# 		sky_data = sky_data - np.nanmedian(sky_data, 0)
# 		sky_data = sky_data- np.nanmedian(sky_data, 1)[:,np.newaxis]
		

# 		sky_mask = (fits.getdata(path_to_sky_mask) == 1.0) #Create a sky mask
# 		sky_data[~sky_mask] = np.nan
# 		self.sky_mask = sky_mask
# 		self.sky_data = sky_data
# 		self.sky_exptime = sky_exptime
# 		self.summed_masked_sky_data = np.nansum(sky_data[sky_mask])
# 		self.median_filter_size = median_filter_size
# 		self.zoom_amount = zoom_amount
# 		fx1 = zoom(np.nansum(sky_data, 0), self.zoom_amount, order=1) #Zoom and collapse into 1D
# 		fy1 = zoom(np.nansum(sky_data, 1), self.zoom_amount, order=1)
# 		# fx1 = np.nansum(zoom(sky_data, [1, self.zoom_amount], order=1), 0) #Zoom and collapse sky frame into 1D
# 		# fy1 = np.nansum(zoom(sky_data, [self.zoom_amount, 1], order=1), 1)
# 		fx1[np.isnan(fx1)] = 0 #Zero out remaining nan pixels
# 		fy1[np.isnan(fy1)] = 0		
# 		self.fx1 = fx1
# 		self.fy1 = fy1



# 	def find_oned_offset(self, path_to_frame, maximum_pixel_search=10): #Find offset in x and y between input frame and sky frame
# 		data = fits.getdata(path_to_frame) #Get data from frame
# 		exptime = float(fits.getval(path_to_frame, 'EXPTIME'))
# 		#data -= bn.nanmedian(data, 0) + bn.nanmedian(data, 1)[:, np.newaxis] #Clean out the readout pattern in an IGRINS frame for better processing
# 		#data -= median_filter(data, size=[1,self.median_filter_size]) #Use a horizontal median filter to remove most background and continnuum, leaving mostly OH sky emission lines
		
# 		data -= gaussian_filter1d(data, sigma=10, axis=1)
# 		data -= median_filter(data, size=[1,self.median_filter_size]) #Use a horizontal median filter to remove most background and continnuum, leaving mostly OH sky emission lines


# 		data = data - np.nanmedian(data, 0)
# 		data = data- np.nanmedian(data, 1)[:,np.newaxis]

# 		data[~self.sky_mask] = np.nan #Mask out everything but brightest pixels, presumeably mostly the OH sky lines
# 		#print('scale = ', self.summed_masked_sky_data / bn.nansum(data))
# 		data *= self.sky_exptime / exptime #Scale data to match the sky data (roughly) before cross-correlation
# 		fx2 = zoom(bn.nansum(data, 0), self.zoom_amount, order=1) #Zoom and collapse into 1D
# 		fy2 = zoom(bn.nansum(data, 1), self.zoom_amount, order=1)
# 		# fx2 = bn.nansum(zoom(data, [1, self.zoom_amount], order=1), 0) #Zoom and collapse into 1D
# 		# fy2 = bn.nansum(zoom(data, [self.zoom_amount, 1], order=1), 1)
# 		fx2[np.isnan(fx2)] = 0 #Zero out remaining nan pixels
# 		fy2[np.isnan(fy2)] = 0
# 		fft_result_x = fftconvolve(self.fx1, fx2[::-1]) #Perform FFIT cross correlation only in x and y
# 		fft_result_y = fftconvolve(self.fy1, fy2[::-1])
# 		delta_sub_pixels = 0.5*maximum_pixel_search*self.zoom_amount #Cut FFT cross-correlation result to be within a maximum number of pixels from zero offset, this cuts out possible extraneous minima screwing up the maximum in the FFT result characterizing the true offset
# 		x1 = int((fft_result_x.shape[0]/2) - delta_sub_pixels)
# 		x2 = int((fft_result_x.shape[0]/2) + delta_sub_pixels)
# 		y1 = int((fft_result_y.shape[0]/2) - delta_sub_pixels)
# 		y2 = int((fft_result_y.shape[0]/2) + delta_sub_pixels)  
# 		fft_sub_result_x = fft_result_x[x1:x2]
# 		fft_sub_result_y = fft_result_y[y1:y2]
# 		find_shift_from_maximum_x = np.unravel_index(np.argmax(fft_sub_result_x), fft_sub_result_x.shape[0]) #Find pixels with strongest correlation
# 		find_shift_from_maximum_y = np.unravel_index(np.argmax(fft_sub_result_y), fft_sub_result_y.shape[0])
# 		fft_dx_result = (find_shift_from_maximum_x[0] -  (fft_sub_result_x.shape[0]/2))/self.zoom_amount #Calcualte the offset from the pixels with the strongest correlation
# 		fft_dy_result = (find_shift_from_maximum_y[0] -  (fft_sub_result_y.shape[0]/2))/self.zoom_amount
# 		return fft_dx_result, fft_dy_result
# 	def correct(self, path_to_frame, dx, dy): #Apply a correction for a given offset [NEEDS A LOT OF WORK!!!]
# 		hdul = fits.open(path_to_frame, mode='update') #Open fits files
# 		data = hdul[0].data
# 		data = roll_along_axis(data, dx, axis=1)
# 		data = roll_along_axis(data, dy, axis=0)
# 		hdul[0].data = data #Replace old data with new data
# 		hdul.close() #Close fits file when finished





#Create reference frames to flexure correct everything to
#for the H and K bands. This is the first SKY frame
def set_reference_frame(obsset):
	if obsset.recipe_name == 'SKY':
		band = get_band(obsset)
		exptime = get_exptime(obsset)
		#Grab sky frame data.  If exposures are short, stack them, otherwise just use first frame
		if exptime > 30.0:  
			print('Sky frames exp time > 30 s.  Using the first frame.')
			hdus = obsset.get_hdus([obsset.get_obsids()[0]]) #Grab first sky frame
			data = hdus[0].data
		else:
			print('Sky frames exp time <= 30 s.  Combining all sky frames.')
			hdus = obsset.get_hdus(obsset.get_obsids())
			data_list = [hdu.data for hdu in hdus]
			data = np.sum(data_list, axis=0)
		#Subtract background
		helper = ResourceHelper(obsset)
		destripe_mask = helper.get("destripe_mask")
		
		
		#data = data -  median_filter(data, size=[100,1])
		# xc, yc, v, std = estimate_background(data, destripe_mask,
		#                                  di=48, min_pixel=40)
		# nx = ny = 2048
		# ZI3 = get_interpolated_cubic(nx, ny, xc, yc, v)
		# ZI3 = np.nan_to_num(ZI3)
		#data = data -  ZI3
		# breakpoint()


		# data = _destripe_sky(data, destripe_mask, subtract_bg=True)

		# data = data - destriper.get_stripe_pattern64(data, mask=destripe_mask,
        #                                      concatenate=True,
        #                                      remove_vertical=False)
		#Remove readout pattern
		#bias_mask = obsset.load_resource_for("bias_mask")
		# if band == 'H':
		# 	#data = destriper.get_destriped(data, remove_vertical=True, pattern=128,
		# 	#								hori=True)
		# 	# data = destriper.get_destriped(data, remove_vertical=True, pattern=128,
		# 	# 								hori=False)
		# 	data = destriper.get_destriped(data, remove_vertical=True, pattern=64,
		# 									hori=False)
		# else:
		# 	# data = destriper.get_destriped(data, remove_vertical=False, pattern=128,
		# 	# 								hori=False)
		# 	data = destriper.get_destriped(data, remove_vertical=False, pattern=64,
		# 									hori=False)
		#data = remove_pattern(data, remove_level=1, remove_amp_wise_var=False)
		# data = data -  median_filter(data, size=[1,50])
		#data = data -  median_filter(data, size=[1,151])  
		#data = #data - gaussian_filter(data, [15,100])
		#data = #data - gaussian_filter(data, [100,15])
		#data = data - median_filter(data- median_filter(data, size=[51,1]), size=[1,15]) 

		data = median_filter(data - median_filter(data, [1, 35]), [11, 1])

		data = data - np.nanmedian(data, 0)
		data = data- np.nanmedian(data, 1)[:,np.newaxis]

		std = np.std(data)
		data[data < -std] = 0.

		#data = data -  median_filter(data- median_filter(data, size=[51,1]), size=[1,15]) 

		#data = remove_pattern(data, mask=destripe_mask, remove_level=2, remove_amp_wise_var=True)

		#data = data -  median_filter(data, size=[400,1])

#		data = remove_pattern(data, mask=destripe_mask, remove_level=2, remove_amp_wise_var=False)
		#data = data -  median_filter(data, size=[1,15])

		#Normalize by exposure time
		#data = data / exptime
		#Store processed for flexure correction sky frames
		hdus_out = obsset.get_hdul_to_write(([], data))

		obsset.store('FLEXCORR_FITS', data=hdus_out)

		#breakpoint()


def estimate_flexure(obsset):
	print('Extimating flexure')
	breakpoint()
	print('Running extimate flexure for', obsset)


if __name__ == "__main__":
    pass

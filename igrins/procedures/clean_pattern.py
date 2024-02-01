#Clean detector readout pattern from the raw data that was noticable in the data from Gemini

import numpy as np 
import copy
#from scipy.ndimage import median_filter
from scipy.signal import medfilt2d




#Stack rows of 128x2048 pixels, and the mirror image of those rows, in order to isolate the pattern (which repeats every 64 pixels, and mirrors itself every other row of 64 pixels)
def stack_rows(masked_data, horizontal_mask=0): 
    dpix = int(128) #128 pixels tall tiles
    half_dpix = int(dpix/2) #Half tile 64 pixels (mirrors itself every other 64 pixel row)
    nchunks = int(2048/dpix) #Number of 128x2048 pixel tiles up the detector
    stackeddata = np.zeros([32, 64, 2048]) #Createa datacube of 32 64x2048 pixel half tiles that will hold the inter-order pixels which we will collapse to get the median pattern
    for i in range(nchunks): #Loop through each half tile
        stackeddata[i*2,:,:] = masked_data[int(dpix*(i)):int((i+0.5)*dpix),:] #Unmirroed row
        stackeddata[i*2+1,:,:] = masked_data[int(dpix*(i+0.5)):int(dpix*(i+1)), :][::-1,:] #Mirroed row
        if horizontal_mask > 0:
            stackeddata[i*2,:,:] -= medfilt2d(stackeddata[i*2,:,:], [1,horizontal_mask])
            stackeddata[i*2+1,:,:] -= medfilt2d(stackeddata[i*2+1,:,:], [1,horizontal_mask])
    stackeddata[stackeddata == 0] = np.nan #Set any pixel not filled to nan before we collapse the 
    median_stacked_data = np.nanmedian(stackeddata,  axis=0) #Median collapse stack of half tiles rows to isolate the readout pattern, taking advantage of the fact it repeats many times and we can thake the mdian of multiple interorder background subtraced pixels    
    reconstructed_tile = np.zeros([dpix, 2048]) #Create an idealized 128x2048 "tile"
    reconstructed_tile[0:half_dpix, :] = median_stacked_data #Painth in the first half tile
    reconstructed_tile[half_dpix:dpix, :] = median_stacked_data[::-1,:] #Paint in the mirroed half tile
    return np.tile(reconstructed_tile, [int(nchunks),1]) #Use np.fkle to recreate pattern over entire 2048x2048 frame, which repeats 16 times the recreated 128x2048 tile



#Clean up the detector pattern and returned the cleaned image
def clean_detector_pattern(data, median_filter_length=23):
    stddev = np.nanstd(data)
    order_mask = (data >  1.0*stddev) & (data > 100)
    cleaned_data = copy.deepcopy(data)
    for i in [7, 9, 11, 13]:
        masked_cleaned_data = copy.deepcopy(cleaned_data) #Remask the cleaned pattern for the next iteration
        masked_cleaned_data[order_mask] = np.nan
        pattern = stack_rows(masked_cleaned_data, horizontal_mask=i) #Get the first iteration of the pattern by stacking the 
        cleaned_data = cleaned_data - pattern #Remove found pattern from this iteration
    # if stddev < 2000: #Run for non-flats or non-super bright sources
    masked_cleaned_data = copy.deepcopy(cleaned_data) #Remask the cleaned pattern for the next iteration
    masked_cleaned_data[order_mask] = np.nan
    smoothed_horizontal_data = medfilt2d(masked_cleaned_data, [1,61]) #Use a horizontal running median filter to characterize the large scale horizontal pattern across the detector
    smoothed_horizontal_data[order_mask] = np.nan #Mask the result
    cleaned_data = cleaned_data - np.nanmin(stack_rows(smoothed_horizontal_data), axis=1)[:,np.newaxis] #Remove the remaining horizontal pattern by nanmin collapsing horizontally to get the remaining horizontal pattern across the whole detector    
    found_nans = ~np.isfinite(cleaned_data)
    cleaned_data[found_nans] = data[found_nans] #fill in any nans (if they exist) with original data to clean up any remaining holes
    return cleaned_data






# def create_order_masks(obsset):
#     if obsset.recipe_name == 'FLAT':

#         obsset_on = obsset.get_subset("ON")
#         obsset_off = obsset.get_subset("OFF")

#         hdus = obsset.get_hdus(obsset_on.get_obsids()) #Stack flat ons
#         data_list = [hdu.data for hdu in hdus]
#         data_on_stacked = np.nansum(data_list, axis=0)

#         hdus = obsset.get_hdus(obsset_off.get_obsids()) #Stack flat offs
#         data_list = [hdu.data for hdu in hdus]
#         data_off_stacked = np.nansum(data_list, axis=0)
#         flat_on_minus_off = data_on_stacked - data_off_stacked

#         thresh = np.nanpercentile(flat_on_minus_off, 75) #Set threshold to 90th percentile
#         data = (flat_on_minus_off > thresh*0.1) #Create order mask where flat is bright
        
#         data[0:4,  :] = True #Mask overscan
#         data[2044:2048,  :] = True
#         data[:, 0:4] = True
#         data[:, 2044:2048] = True

#         obsset.store('PATTERNMASK_FITS', data=data)
#         breakpoint()




# #Stack rows of 128x2048 pixels, and the mirror image of those rows, in order to isolate the pattern (which repeats every 64 pixels, and mirrors itself every other row of 64 pixels)
# def stack_rows(masked_data): 
#     dpix = int(128) #128 pixels tall tiles
#     half_dpix = int(dpix/2) #Half tile 64 pixels (mirrors itself every other 64 pixel row)
#     nchunks = int(2048/dpix) #Number of 128x2048 pixel tiles up the detector
#     stackeddata = np.zeros([32, 64, 2048]) #Createa datacube of 32 64x2048 pixel half tiles that will hold the inter-order pixels which we will collapse to get the median pattern
#     for i in range(nchunks): #Loop through each half tile
#     	stackeddata[i*2,:,:] = masked_data[int(dpix*(i)):int((i+0.5)*dpix),:] #Unmirroed row
#     	stackeddata[i*2+1,:,:] = masked_data[int(dpix*(i+0.5)):int(dpix*(i+1)), :][::-1,:] #Mirroed row
#     stackeddata[stackeddata == 0] = np.nan #Set any pixel not filled to nan before we collapse the 
#     median_stacked_data = np.nanmedian(stackeddata,  axis=0) #Median collapse stack of half tiles rows to isolate the readout pattern, taking advantage of the fact it repeats many times and we can thake the mdian of multiple interorder background subtraced pixels    
#     reconstructed_tile = np.zeros([dpix, 2048]) #Create an idealized 128x2048 "tile"
#     reconstructed_tile[0:half_dpix, :] = median_stacked_data #Painth in the first half tile
#     reconstructed_tile[half_dpix:dpix, :] = median_stacked_data[::-1,:] #Paint in the mirroed half tile
#     return np.tile(reconstructed_tile, [int(nchunks),1]) #Use np.fkle to recreate pattern over entire 2048x2048 frame, which repeats 16 times the recreated 128x2048 tile



# #Clean up the detector pattern and returned the cleaned image
# def clean_detector_pattern(data, median_filter_length=25):
#     order_mask = data >  np.nanpercentile(data, 40) + 40
#     # order_mask[0:4,:] = True #mask overscan
#     # order_mask[:,0:4] = True
#     # order_mask[2044:2048,:] = True
#     # order_mask[:,2044:2048] = True
#     cleaned_data = copy.deepcopy(data)
#     masked_cleaned_data = copy.deepcopy(data)
#     masked_cleaned_data[order_mask] = np.nan
#     for i in range(5): #Iterate several times
#         masked_cleaned_data -= medfilt2d(masked_cleaned_data, [1,median_filter_length+2*(i-5)]) #Horizontal median filter to remove science data and background
#         pattern = stack_rows(masked_cleaned_data) #Get the first iteration of the pattern by stacking the masked background+science subtraced image
#         cleaned_data = cleaned_data - pattern #Remove found pattern from this iteration
#         masked_cleaned_data = copy.deepcopy(cleaned_data) #Remask the cleaned pattern for the next iteration
#         masked_cleaned_data[order_mask] = np.nan
#     smoothed_horizontal_data = medfilt2d(masked_cleaned_data, [1,61]) #Use a horizontal running median filter to characterize the large scale horizontal pattern across the detector
#     smoothed_horizontal_data[order_mask] = np.nan #Mask the result
#     cleaned_data = cleaned_data - np.nanmin(stack_rows(smoothed_horizontal_data), axis=1)[:,np.newaxis] #Remove the remaining horizontal pattern by nanmin collapsing horizontally to get the remaining horizontal pattern across the whole detector
#     found_nans = ~np.isfinite(cleaned_data)
#     cleaned_data[found_nans] = data[found_nans] #fill in any nans (if they exist) with original data to clean up any remaining holes
#     return cleaned_data

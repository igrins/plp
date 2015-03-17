#Test script for creating extra corrections 
import tell_wvsol

bands = ['H','K']
date = '20141023'
datadir = 'outdata/'+date+'/'
#calibdir = 'calib/'+date+'/prima

import libs.recipes as recipes #Import library for reading in table of night from directory recipe_logs

night = recipes.Recipes('recipe_logs/'+date+'.recipes') #Load up table for a night
framenos = [] #Set up list to store numbers for first frame of each A0V standadr star
for found_a0v in night.recipe_dict['A0V_AB']: #Loop through dictionary7
	framenos.append( '%.4d' % found_a0v[0][0]) #Append the first frame found for each A0V star in a night

print framenos
			  
for band in bands:
	for frame in framenos:
		tell_wvsol.run(datadir+'SDC'+band+'_'+date+'_'+frame+'.spec.fits')


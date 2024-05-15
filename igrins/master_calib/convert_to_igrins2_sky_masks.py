#Simple script to convert sky masks from IGRINS 1 to IGRINS 2

import numpy as np
from scipy.ndimage import binary_dilation
from astropy.io import fits

#Convert H-band

d = fits.getdata('H-band_sky_mask.fits')
d2 = binary_dilation(np.roll(np.roll(d, 30, axis=0), -140, axis=1), iterations=10)*1
fits.writeto('H-band_sky_mask_igrins2.fits', d2, overwrite=True)

d = fits.getdata('H-band_limited_sky_mask.fits')
d2 = binary_dilation(np.roll(np.roll(d, 30, axis=0), -140, axis=1), iterations=10)*1
fits.writeto('H-band_limited_sky_mask_igrins2.fits', d2, overwrite=True)

#Convert K-band

d = fits.getdata('K-band_sky_mask.fits')
d2 = binary_dilation(np.roll(np.roll(d, 85, axis=0), -40, axis=1), iterations=5)*1
fits.writeto('K-band_sky_mask_igrins2.fits', d2, overwrite=True)

d = fits.getdata('K-band_limited_sky_mask.fits')
d2 = binary_dilation(np.roll(np.roll(d, 85, axis=0), -40, axis=1), iterations=5)*1
fits.writeto('K-band_limited_sky_mask_igrins2.fits', d2, overwrite=True)
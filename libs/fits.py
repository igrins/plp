import os
from astropy.io.fits import *  # generally import * is horrid, but is marginally acceptable in this context
from astropy.io.fits import open as fits_open_original

# Purpose of this module is to provide a wrapper to astropy.io.fits which will 
# automatically read in either FILENAME.fits *or* FILENAME.fits.gz, whichever
# one exists on disk.

# Note that astropy.io.fits already handles gzip'd files well, so the purpose of this wrapper is
# really just to hide the *.gz suffix from the rest of the IGRINS PLP code.

# TODO someday:  automatically gzip output as well

# By replacing throughout igrins plp:
#     import astropy.io.fits as pyfits
# with
#     import libs.fits as pyfits

def open(name, **kwargs):
    """
    Simple wrapper to astropy.io.fits.open that receives FILENAME.fits in name, 
    and then:
        if FILENAME.fits.gz exists on disk, opens FILENAME.fits.gz
    otherwise
        proceeds with opening FILENAME.fits
    """
    if os.path.isfile(name + '.gz'):
        return fits_open_original(name + '.gz', **kwargs)
    else:
        return fits_open_original(name, **kwargs)


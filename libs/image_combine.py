import libs.fits as pyfits
from stsci_helper import stsci_median

def make_combined_image_thar(helper, band, obsids):
    """
    simple median combine with destripping. Suitable for sky.
    """
    filenames, basename, master_obsid = helper.get_base_info(band, obsids)

    hdu_list = [pyfits.open(fn)[0] for fn in filenames]
    _data = stsci_median([hdu.data for hdu in hdu_list])

    from destriper import destriper
    data = destriper.get_destriped(_data)

    return data

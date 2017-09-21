import os
import astropy.io.fits as pyfits

def get_first_science_hdu(hdu_list):
    if hdu_list[0].data is None:
        return hdu_list[1]
    else:
        return hdu_list[0]

candidate_generators = [(lambda x: os.path.extsep.join([x, "gz"]),
                         pyfits.open)]

def find_fits(fn):

    if os.path.exists(fn):
        return fn

    fn_search_list = [fn]

    for gen_candidate, open_fits in candidate_generators:
        fn1 = gen_candidate(fn)
        if os.path.exists(fn1):
            return fn1

        fn_search_list.append(fn1)
    else:
        raise RuntimeError("No candidate files are found : %s" 
                           % fn_search_list)

def open_fits(fn):

    if os.path.exists(fn):
        return pyfits.open(fn)

    fn_search_list = [fn]

    for gen_candidate, open_fits in candidate_generators:
        fn1 = gen_candidate(fn)
        if os.path.exists(fn1):
            return open_fits(fn1)

        fn_search_list.append(fn1)
    else:
        raise RuntimeError("No candidate files are found : %s" 
                           % fn_search_list)


def load_fits_data(fn):
    hdu_list = open_fits(fn)
    hdu = get_first_science_hdu(hdu_list)
    return hdu


# Below are helper funtions

from astropy.io.fits import Card

def get_hdus(helper, band, obsids):
    _ = helper.get_base_info(band, obsids)
    filenames = _[0]

    hdus = [load_fits_data(fn_) for fn_ in filenames]

    return hdus

# data_list = [hdu.data for hdu in hdu_list]

# return data_list


def get_combined_image(hdus): #, destripe=True):
    # destripe=True):

    data_list = [hdu.data for hdu in hdus]

    from .stsci_helper import stsci_median
    im = stsci_median(data_list)

    return im

if 0:
    cards = []

    if destripe:
        from destriper import destriper
        im = destriper.get_destriped(im)

        cards.append(Card("HISTORY", "IGR: image destriped."))

    return im, cards


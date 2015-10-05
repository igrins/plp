import os
import numpy as np
import libs.fits as pyfits

import matplotlib.pyplot as plt

from libs.path_info import IGRINSPath, IGRINSFiles
from libs.recipes import load_recipe_list, make_recipe_dict
from libs.products import PipelineProducts, ProductPath, ProductDB

if 1:
    band = "K"

    utdate, obsid = "20140524", 45 # Serpens 15
    utdate, obsid = "20140526", 158 # Serpens 2
    utdate, obsid = "20140526", 104 # GSS 30
    utdate, obsid = "20140707", 149 # S140 N3
    utdate, obsid = "20140707", 161 # S140 N13

    igr_path = IGRINSPath(utdate)

    igrins_files = IGRINSFiles(igr_path)

    fn = "%s.recipes" % utdate
    recipe_list = load_recipe_list(fn)
    recipe_dict = make_recipe_dict(recipe_list)

    abba = [rd[-1] for rd in recipe_dict["STELLAR_AB"] if obsid in rd[0]]

    objname = abba[-1][0]
    #obsids = abba[0]
    #frametypes = abba[1]


    obj_filenames = igrins_files.get_filenames(band, [obsid])
    obj_path = ProductPath(igr_path, obj_filenames[0])
    obj_master_obsid = obsid

    fn = obj_path.get_secondary_path("spec.fits")
    s_list = list(pyfits.open(fn)[0].data)

    if 1:

        sky_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
                                        "sky.db"))

        basename = sky_db.query(band, obj_master_obsid)
        sky_path = ProductPath(igr_path, basename)
        fn = sky_path.get_secondary_path("wvlsol_v1")
        wvlsol_products = PipelineProducts.load(fn)

        orders_w_solutions = wvlsol_products["orders"]
        wvl_solutions = wvlsol_products["wvl_sol"]


        A0V_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
                                        "A0V.db"))

        basename = A0V_db.query(band, obj_master_obsid)
        A0V_path = ProductPath(igr_path, basename)
        # fn = A0V_path.get_secondary_path("spec_flattened.fits")
        # telluric_cor = list(pyfits.open(fn)[0].data)
        fn = A0V_path.get_secondary_path("spec.fits")
        telluric_cor = list(pyfits.open(fn)[0].data)

        thar_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
                                           "thar.db"))

        basename = thar_db.query(band, obj_master_obsid)

        thar_path = ProductPath(igr_path, basename)
        fn = thar_path.get_secondary_path("median_spectra")
        thar_products = PipelineProducts.load(fn)
        fn = thar_path.get_secondary_path("orderflat")
        orderflat_products = PipelineProducts.load(fn)

        i1i2_list = orderflat_products["i1i2_list"]

    # if 1:
    #     ii = 8
    new_orders = orderflat_products["orders"]

    for ii, s_ in enumerate(s_list):
        o = orders_w_solutions[ii]
        o_new_ind = np.searchsorted(new_orders, o)
        i1, i2 = i1i2_list[o_new_ind]
        sl = slice(i1, i2)

        ax1 = plt.subplot(211)
        ax1.plot(wvl_solutions[ii][sl], s_list[ii][sl])

        ax2 = plt.subplot(212, sharex=ax1)
        ss = s_list[ii]/telluric_cor[ii]
        yy = np.median(ss[800:1200])
        import scipy.ndimage as ni
        ssyy = ni.median_filter(ss/yy, 5)
        ax2.plot(wvl_solutions[ii][sl], ssyy[sl])

    ax2.set_ylim(0.2, 2.)


    for lam in [2.22112, 2.22328, 2.22740, 2.23106]:
        ax1.axvline(lam)
        ax2.axvline(lam)

    ax1.set_title(objname)
    #ax1.set_xlim(2.211, 2.239)
    #ax.set_ylim(0.69, 1.09)

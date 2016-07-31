import numpy as np


def save_wat_header(helper, band, obsids):

    caldb = helper.get_caldb()

    master_obsid = obsids[0]
    basename = (band, master_obsid)

    d = caldb.load_item_from(basename, "SKY_WVLSOL_JSON")
    # d = dict(orders=orders,
    #          wvl_sol=wvl_sol)

    orders = d["orders"]
    wvl_sol = d["wvl_sol"]

    fit_results = caldb.load_item_from(basename,
                                       "SKY_WVLSOL_FIT_RESULT_JSON")

    # fit_results = dict(xyz=[xl[msk], yl[msk], zl[msk]],
    #                    fit_params=fit_params,
    #                    fitted_model=poly_2d)

    modeul_name, class_name, serialized = fit_results["fitted_model"]
    from libs.astropy_poly_helper import deserialize_poly_model
   
    p = deserialize_poly_model(modeul_name, class_name, serialized)

    # save as WAT fits header
    xx = np.arange(0, 2048)
    xx_plus1 = np.arange(1, 2048+1)

    from astropy.modeling import models, fitting

    # We convert 2d chebyshev solution to a seriese of 1d
    # chebyshev.  For now, use naive (and inefficient)
    # approach of refitting the solution with 1d. Should be
    # reimplemented.

    p1d_list = []
    for o in orders:
        oo = np.empty_like(xx)
        oo.fill(o)
        wvl = p(xx, oo) / o * 1.e4  # um to angstrom

        p_init1d = models.Chebyshev1D(domain=[1, 2048],
                                      degree=p.x_degree)
        fit_p1d = fitting.LinearLSQFitter()
        p1d = fit_p1d(p_init1d, xx_plus1, wvl)
        p1d_list.append(p1d)

    from libs.iraf_helper import get_wat_spec, default_header_str
    wat_list = get_wat_spec(orders, p1d_list)

    # cards = [pyfits.Card.fromstring(l.strip()) \
    #          for l in open("echell_2dspec.header")]
    import astropy.io.fits as pyfits
    cards = [pyfits.Card.fromstring(l.strip())
             for l in default_header_str]

    wat = "wtype=multispec " + " ".join(wat_list)
    char_per_line = 68
    num_line, remainder = divmod(len(wat), char_per_line)
    for i in range(num_line):
        k = "WAT2_%03d" % (i+1,)
        v = wat[char_per_line*i:char_per_line*(i+1)]
        #print k, v
        c = pyfits.Card(k, v)
        cards.append(c)

    if remainder > 0:
        i = num_line
        k = "WAT2_%03d" % (i+1,)
        v = wat[char_per_line*i:]
        #print k, v
        c = pyfits.Card(k, v)
        cards.append(c)

    # save fits with empty header

    # header = pyfits.Header(cards)
    # hdu = pyfits.PrimaryHDU(header=header,
    #                         data=np.array([]).reshape((0,0)))

    caldb.store_image((band, master_obsid),
                      "SKY_WVLSOL_FITS", np.array(wvl_sol),
                      card_list=cards)

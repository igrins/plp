
# old code

basename = extractor.basenames["flat_on"]
flat = FlatHelper(extractor.pr, basename)

bottomup_solutions = flat.centroid_sol["bottom_up_solutions"]
orders = range(len(bottomup_solutions))

from libs.apertures import Apertures
ap =  Apertures(orders, bottomup_solutions)
#ap.extract_spectra_v2()

order_map2 = pyfits.open("calib/primary/20150525/SKY_SDCH_20150525_0052.order_map.fits")[0].data > 0
#order_map2 = ap.make_order_map(mask_top_bottom=True)
bias_mask = (flat.mask.data & (order_map2 > 0)) | extractor.pix_mask

#extractor.bias_mask = bias_mask

if 0:
    # make dark image
    obslist = map(int, "2 3 4 5 6 7 8 9 10 11".split())
    extractor0 = Extractor(20150525, "H",
                          obsids=obslist, frametypes="A"*len(obslist),
                          config=config,
                          ab_mode=False)

    data_list = []
    for i in range(len(obslist)):
        datai = extractor0.get_data1(i, destripe_pattern=False,
                                    bias_mask=None, hori=False, vert=False)
        data_list.append(datai)

    dark10 = np.mean(data_list, axis=0)
    dark1 = dark10 / 30.

    dark20 = np.median(data_list, axis=0)
    dark2 = dark20 / 30.


    destriper = Destriper()
    d2 = destriper.get_stripe_pattern64(dark2, concatenate=True, remove_vertical=False)

    dark = ni.median_filter(dark2-d2, (11,11))
    ss = np.median(dark, axis=0)
    ss2 = np.median(dark-ss, axis=1)
    ddd = dark - ss - ss2[:, np.newaxis]



if 0:
    data0 = extractor.get_data1(0, destripe_pattern=False, bias_mask=bias_mask, hori=False, vert=False)
    data1 = extractor.get_data1(1, destripe_pattern=False, bias_mask=bias_mask, hori=False, vert=False)
    data = extractor.get_data1(0, destripe_pattern=False, bias_mask=bias_mask, hori=True, vert=True)
    data = extractor.get_data1(0, destripe_pattern=64, bias_mask=bias_mask, hori=False, vert=False)
    #data = extractor.get_data1(0, bias_mask=bias_mask, hori=True, vert=False)

if 1:
    from libs.destriper import Destriper
    destriper = Destriper()
    i = 1
    d = extractor.get_data1(i, destripe_pattern=False, bias_mask=bias_mask, hori=False, vert=False)
    datai0 = d-dark2*1200
    d2 = destriper.get_stripe_pattern64(datai0, mask=bias_mask, concatenate=True, remove_vertical=False)
    datai = datai0 - d2

if 0:
    #d = data0 #-data1

    obsids = [85, 86, 87, 88, 89, 90, 91, 92]
    obsids = [51, 52, 53, 54]
    band = "H"

    sub_pattern = band in ["H"] # ["H", "K"]

    extractor = Extractor(20150525, band,
                          #obsids=[51, 52, 53, 54], frametypes="ABBA",
                          obsids=obsids, frametypes=["A"] * len(obsids),
                          config=config,
                          ab_mode=False)


    from libs.destriper import Destriper
    destriper = Destriper()
    h_dict = {}

    i_range = range(len(obsids))

    for i in i_range:
        d = extractor.get_data1(i, destripe_pattern=False, bias_mask=bias_mask, hori=False, vert=False)

        sky = extractor.estimate_sky(d, extractor.sky_mask)

        bbb = bias_mask|~np.isfinite(d)
        if sub_pattern == True:
            d2 = destriper.get_stripe_pattern64(d-sky, mask=bbb,
                                                concatenate=True, remove_vertical=False)



            d2[:,-7:] = 0.
            d2[:,:8] = 0.

            d = d - d2

            sky = extractor.estimate_sky(d, extractor.sky_mask & np.isfinite(d))

        #d2 = destriper.get_destriped(d, mask=bias_mask, hori=None, pattern=64)

        h = d-sky

        h_dict[i] = h


    ####

    clf()
    if band == "K":
        bins = np.linspace(-15, 40, 100)
    else:
        bins = np.linspace(-10, 30, 100)
    for i in i_range:
        h = h_dict[i]
        m1 = extractor.sky_mask & np.isfinite(h)
        plt.hist(h[m1], bins=bins,
                 histtype="step")

    plt.legend(["%d" % i for i in i_range])





if 0:
    #d = data0 #-data1

    d0 = extractor.get_data1(0, destripe_pattern=False,
                             bias_mask=bias_mask, hori=False, vert=False)
    d1 = extractor.get_data1(1, destripe_pattern=False,
                             bias_mask=bias_mask, hori=False, vert=False)
    d2 = extractor.get_data1(2, destripe_pattern=False,
                             bias_mask=bias_mask, hori=False, vert=False)
    d3 = extractor.get_data1(3, destripe_pattern=False,
                             bias_mask=bias_mask, hori=False, vert=False)

    dd01 = d0 - d1
    dd23 = d2 - d3
    bbb = bias_mask #|~np.isfinite(d)
    p01 = destriper.get_stripe_pattern64(dd01, mask=bbb,
                                         concatenate=True, remove_vertical=False)
    p23 = destriper.get_stripe_pattern64(dd23, mask=bbb,
                                         concatenate=True, remove_vertical=False)

    dd01p = dd01-p01
    dd01p2 = dd01p - np.median(dd01p, axis=-1)[:,np.newaxis]

    sky01 = extractor.estimate_sky(dd01p2, extractor.sky_mask)

    dd23p = dd23-p23
    dd23p2 = dd23p - np.median(dd23p, axis=-1)[:,np.newaxis]

    sky23 = extractor.estimate_sky(dd23p2, extractor.sky_mask)

    ds92.view(dd01-p01, frame=1), ds92.view(dd23-p23, frame=2)

    sky = extractor.estimate_sky(dd-p, extractor.sky_mask)

    h = d-sky

    h_dict[i] = h


    ####

    clf()
    if band == "K":
        bins = np.linspace(-15, 40, 100)
    else:
        bins = np.linspace(-10, 30, 100)
    for i in i_range:
        h = h_dict[i]
        m1 = extractor.sky_mask & np.isfinite(h)
        plt.hist(h[m1], bins=bins,
                 histtype="step")

    plt.legend(["%d" % i for i in i_range])



###


if True:
    utdate = "20150801"
    band = "H"
    obsids = [59, 58, 61, 63]
    frametypes = "A B B A".split()

    extractor = RecipeExtractBase(utdate, band,
                                  obsids, frametypes,
                                  "recipe.config",
                                  ab_mode=False)

    _ = extractor.get_data_variance(destripe_pattern=64,
                                    use_destripe_mask=True,
                                    sub_horizontal_median=True)

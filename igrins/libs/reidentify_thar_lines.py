import numpy as np
from reidentify import reidentify_lines_all2


def match_orders(orders, s_list_src, s_list_dst):
    """
    try to math orders of src and dst
    """
    center_indx0 = int(len(s_list_src)/2)

    delta_indx_list = []
    dstep = 2
    for di in range(-dstep, dstep+1):
        center_indx = center_indx0 + di
        center_s = s_list_src[center_indx]

        s_std = np.nanstd(center_s)
        center_s_clip = np.clip(center_s, -0.1*s_std, s_std)/s_std

        from scipy.signal import correlate
        from scipy.ndimage import median_filter
        #ss5 = ni.median_filter(ss, 25)

        # TODO : it is not clear if this is a right algorithm

        # we have to clip the spectra so that the correlation is not
        # sensitive to bright lines in the target spectra.
        s_list_dst_filtered = [s - median_filter(s, 55) for s in s_list_dst]
        std_list = [np.nanstd(s) for s in s_list_dst_filtered]
        #std_list = [100 for s in s_list_dst_filtered]
        s_list_dst_clip = [np.clip(s, -0.1*s_std, s_std)/s_std for (s, s_std) \
                           in zip(s_list_dst_filtered, std_list)]
        cor_list = [correlate(center_s_clip, s, mode="same") for s in s_list_dst_clip]
        cor_max_list = [np.nanmax(cor) for cor in cor_list]

        center_indx_dst = np.nanargmax(cor_max_list)

        delta_indx = center_indx - center_indx_dst
        delta_indx_list.append(delta_indx)

    #print cor_max_list, delta_indx
    print "index diferences : ", delta_indx_list
    delta_indx = sorted(delta_indx_list)[dstep]
    center_indx_dst0 = center_indx0 + delta_indx

    orders_dst = np.arange(len(s_list_dst)) - center_indx_dst0 + orders[center_indx0]

    return delta_indx, orders_dst


def get_offset_transform(thar_spec_src, thar_spec_dst):

    from scipy.signal import correlate
    offsets = []
    cor_list = []
    center = 2048/2.

    for s_src, s_dst in zip(thar_spec_src, thar_spec_dst):
        cor = correlate(s_src, s_dst, mode="same")
        cor_list.append(cor)
        offset = center - np.argmax(cor)
        offsets.append(offset)

    #from skimage.measure import ransac, LineModel
    from skimage_measure_fit import ransac, LineModel

    xi = np.arange(len(offsets))
    data = np.array([xi, offsets]).T
    model_robust, inliers = ransac(data,
                                   LineModel, min_samples=3,
                                   residual_threshold=2, max_trials=100)

    outliers_indices = xi[inliers == False]
    offsets2 = [o for o in offsets]
    for i in outliers_indices:
        # reduce the search range for correlation peak using the model
        # prediction.
        ym = int(model_robust.predict_y(i))
        x1 = int(max(0, (center - ym) - 20))
        x2 = int(min((center - ym) + 20 + 1, 2048))
        print i, x1, x2
        ym2 = center - (np.argmax(cor_list[i][x1:x2]) + x1)
        #print ym2
        offsets2[i] = ym2


    def get_offsetter(o):
        def _f(x, o=o):
            return x+o
        return _f
    sol_list = [get_offsetter(offset_) for offset_ in offsets2]

    return dict(sol_type="offset",
                sol_list=sol_list,
                offsets_orig=offsets,
                offsets_revised=offsets2)



if 0:

    import json

    band = "K"

    date = "20140316"
    # load spec
    ref_date = date
    ref_spec_file = "arc_spec_thar_%s_%s.json" % (band, date)
    ref_id_file = "thar_identified_%s_%s.json" % (band, date)

    def get_master_calib_abspath(fn):
        import os
        return os.path.join("master_calib", fn)

    s_list_ = json.load(open(get_master_calib_abspath(ref_spec_file)))
    s_list_src = [np.array(s) for s in s_list_]

    # reference line list : from previous run
    ref_lines_list = json.load(open(ref_id_file))

    date = "20140525"
    # load spec
    s_list_ = json.load(open("arc_spec_thar_%s_%s.json" % (band, date)))
    s_list_dst = [np.array(s) for s in s_list_]


    # get offset function from source spectra to target specta.
    sol_list_transform = get_offset_transform(s_list_src, s_list_dst)

    reidentified_lines_with_id = reidentify_lines_all2(s_list_dst,
                                                       ref_lines_list,
                                                       sol_list_transform)

    r = dict(match_list=reidentified_lines_with_id,
             ref_date=ref_date,
             ref_spec_file=ref_spec_file,
             ref_id_file=ref_id_file)

    json.dump(r,
              open("thar_shifted_%s_%s.json" % (band, date),"w"))

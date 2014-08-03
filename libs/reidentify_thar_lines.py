import numpy as np
from reidentify import reidentify_lines_all2


def match_orders(orders, s_list_src, s_list_dst):
    """
    try to math orders of src and dst
    """
    center_indx = int(len(s_list_src)/2)

    center_s = s_list_src[center_indx]

    from scipy.signal import correlate

    # TODO : it is not clear if this is a right algorithm

    # we have to clip the spectra so that the correlation is not
    # sensitive to bright lines in the target spectra.
    s_list_dst_clip = [np.clip(s, -10, 100) for s in s_list_dst]
    cor_list = [correlate(center_s, s, mode="same") for s in s_list_dst_clip]
    cor_max_list = [np.nanmax(cor) for cor in cor_list]

    center_indx_dst = np.nanargmax(cor_max_list)

    delta_indx = center_indx - center_indx_dst

    print cor_max_list, delta_indx

    orders_dst = np.arange(len(s_list_dst)) - center_indx_dst + orders[center_indx]

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

    from skimage.measure import ransac, LineModel

    xi = np.arange(len(offsets))
    data = np.array([xi, offsets]).T
    model_robust, inliers = ransac(data,
                                   LineModel, min_samples=3,
                                   residual_threshold=2, max_trials=100)

    outliers_indices = xi[inliers == False]
    offsets2 = [o for o in offsets]
    for i in outliers_indices:
        ym = int(model_robust.predict_y(i))
        x1 = max(0, (center - ym) - 20)
        x2 = min((center - ym) + 20 + 1, 2048)
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

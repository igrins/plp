import numpy as np


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
        # ss5 = ni.median_filter(ss, 25)

        # TODO : it is not clear if this is a right algorithm

        # we have to clip the spectra so that the correlation is not
        # sensitive to bright lines in the target spectra.
        s_list_dst_filtered = [s - median_filter(s, 55) for s in s_list_dst]
        std_list = [np.nanstd(s) for s in s_list_dst_filtered]
        # std_list = [100 for s in s_list_dst_filtered]
        s_list_dst_clip = [np.clip(s, -0.1*s_std, s_std)/s_std for (s, s_std)
                           in zip(s_list_dst_filtered, std_list)]
        cor_list = [correlate(center_s_clip, s, mode="same") for s
                    in s_list_dst_clip]

        import warnings
        with warnings.catch_warnings():
            msg = r'All-NaN (slice|axis) encountered'
            warnings.filterwarnings('ignore', msg)

            cor_max_list = [np.nanmax(cor) for cor in cor_list]

        center_indx_dst = np.nanargmax(cor_max_list)

        delta_indx = center_indx - center_indx_dst
        delta_indx_list.append(delta_indx)

    # print cor_max_list, delta_indx
    # print "index diferences : ", delta_indx_list
    delta_indx = sorted(delta_indx_list)[dstep]
    center_indx_dst0 = center_indx0 + delta_indx

    orders_dst = (np.arange(len(s_list_dst))
                  - center_indx_dst0 + orders[center_indx0])

    return delta_indx, orders_dst

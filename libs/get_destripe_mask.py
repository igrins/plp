def get_pix_mask(helper, band, obsids):

    caldb = helper.get_caldb()
    master_obsid = obsids[0]

    d_hotpix = caldb.load_resource_for((band, master_obsid),
                                       "hotpix_mask")
    d_deadpix = caldb.load_resource_for((band, master_obsid),
                                        "deadpix_mask")
    pix_mask = d_hotpix.data | d_deadpix.data

    return pix_mask


def get_destripe_mask(helper, band, obsids, pix_mask=None):

    caldb = helper.get_caldb()
    master_obsid = obsids[0]

    if pix_mask is None:
        pix_mask = get_pix_mask(helper, band, obsids)

    d_bias = caldb.load_resource_for((band, master_obsid),
                                     "bias_mask")

    mask = d_bias.data
    mask[pix_mask] = True
    mask[:4] = True
    mask[-4:] = True

    return mask

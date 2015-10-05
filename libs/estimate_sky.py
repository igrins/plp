import numpy as np
#from jj_recipe_base import get_pr
#import libs.fits as pyfits
import scipy.ndimage as ni


def estimate_background(data, msk, di=24, min_pixel=10):
    import scipy.ndimage as ni

    def get_sky_points1(data, msk, di, i0):
        """
        return array of xc, yc, v, std

        """
        bar = data[:,i0:i0+di]
        bar_msk = ~msk[:,i0:i0+di]

        labeled, nlabels = ni.label(bar_msk)
        #ss = ni.find_objects(labeled)
        index = np.arange(1, nlabels+1)
        labeled_sum = ni.sum(bar_msk, labeled, index)

        object_slices = ni.find_objects(labeled)
        index_selected = index[labeled_sum > min_pixel]

        segmented_slices = []
        for ind in index_selected:
            sl_y, sl_x = object_slices[ind-1]
            height = sl_y.stop - sl_y.start

            if height > 2 * di: # further slice
                nh = height // di
                h, remainder = divmod(height , nh)
                h_array = np.zeros(nh, dtype="i") + h
                h_array[:remainder] += 1

                sl_y0 = sl_y.start
                for h in h_array:
                    sl_y = slice(sl_y0, sl_y0+h)
                    sl_y0 += h
                    segmented_slices.append((sl_y, sl_x, ind))
            else:
                segmented_slices.append((sl_y, sl_x, ind))



        sky_points = []
        for sl_y, sl_x, ind in segmented_slices:
            bar_sl = bar[sl_y, sl_x]
            labeled_sl = labeled[sl_y, sl_x]
            #msk_sl = bar_msk[sl_y, sl_x]

            msk_sl = (labeled_sl == ind)
            yc_sl, xc_sl = ni.center_of_mass(msk_sl)

            xc = i0 + sl_x.start + xc_sl
            yc = sl_y.start + yc_sl

            v = np.median(bar_sl[msk_sl])
            e = (bar_sl[msk_sl] - v)**2
            sky_points.append((xc, yc, v, e.sum()**.5))

        return sky_points



    #di = 24
    sky_points_all = []
    for i0 in range(0, 2048, di):
        sky_points1 = get_sky_points1(data, msk, di, i0)
        sky_points_all.extend(sky_points1)


    xc = np.array([_[0] for _ in sky_points_all])
    yc = np.array([_[1] for _ in sky_points_all])
    v = np.array([_[2] for _ in sky_points_all])
    std = np.array([_[3] for _ in sky_points_all])

    #m = np.isfinite(v)

    #return xc[m], yc[m], v[m], std[m]
    return xc, yc, v, std

if 0:
    #from scipy.interpolate import Rbf
    #rbf = Rbf(xc, yc, v, epsilon=2)
    ti = np.arange(0, 2048)
    XI, YI = np.meshgrid(ti, ti)



def get_interpolated_rbf(nx, ny, xc, yc, v, smooth=1.e-12, nsample=16, nr=1000):
    from scipy.interpolate import Rbf

    rbf = Rbf(xc, yc, v, smooth=smooth)
    nn = nsample

    XI, YI = np.meshgrid(np.arange(0, nx), np.arange(0, ny))

    xi_r4 = np.ravel(XI[::nn,::nn])
    yi_r4 = np.ravel(YI[::nn,::nn])

    d, r = divmod(len(xi_r4), nr)
    ii = [nr] * d + [r]

    zi_list = []
    for i in range(d+1):
        xi = xi_r4[i*nr:(i+1)*nr]
        yi = yi_r4[i*nr:(i+1)*nr]
        zi = rbf(xi, yi)
        zi_list.append(zi)

    ZI4 = np.concatenate(zi_list).reshape(XI[::nn,::nn].shape)
    ZI = ni.zoom(ZI4, nn)

    return ZI


def get_interpolated_alglib(nx, ny, xc, yc, v,
                            nsample=16,
                            radius=100, nlayer=1, par=1.e0):
    import xalglib
    model = xalglib.rbfcreate(2, 1)
    #xy = [[-1,0,2],[+1,0,3]]
    #xy = np.array([xc, yc, v]).T
    xy = map(list, zip(xc, yc, v))
    xalglib.rbfsetpoints(model, xy)
    #xalglib.rbfsetalgoqnn(model)

    # check the meaning of par
    xalglib.rbfsetalgomultilayer(model, radius, nlayer, par)

    rep = xalglib.rbfbuildmodel(model)
    tx = list(np.arange(0, nx, nsample)+nsample*.5)
    ty = list(np.arange(0, ny, nsample)+nsample*.5)
    #xti1 = xalglib.x_vector(512)
    #xalglib.x_from_list(xti1, ti, xalglib.DT_REAL, xalglib.X_CREATE)
    #xti2 = xalglib.x_vector(512)
    #xalglib.x_from_list(xti2, ti, xalglib.DT_REAL, xalglib.X_CREATE)
    vv1 = xalglib.rbfgridcalc2(model, tx, len(tx), ty, len(ty))
    vv4 = ni.zoom(np.array(vv1).T, nsample)

    return vv4


def get_interpolated_cubic(nx, ny, xc, yc, v,
                            ):
    XI, YI = np.meshgrid(np.arange(0, nx),
                         np.arange(0, ny))

    xi_r = np.ravel(XI)
    yi_r = np.ravel(YI)

    in_xy = np.array([xc, yc]).T
    out_xy = np.array([xi_r, yi_r]).T
    import scipy
    import scipy.interpolate
    out_v = scipy.interpolate.griddata(in_xy, v, out_xy,
                                       method="cubic").reshape(XI.shape)

    return out_v


if __name__ == "__main__":
    utdate=20140525
    #utdate=20140526
    band = "H"
    objids = [11, 12]
    frametypes = ["A", "B"]

    ap = MyAperture(utdate, band, objids, frametypes)

    #data = ap.get_data()
    #data0 = ap.get_data1(0, hori=False, vert=False)

    data = ap.get_data1(0, hori=False, vert=False)
    #data = ap.get_data(hori=False)
    data[ap.pix_mask] = np.nan

    #import scipy.ndimage as ni

    msk_ = (ap.ordermap_bpixed > 0) | ap.bias_mask

    msk = msk_
    #msk = ni.binary_dilation(msk_, iterations=0)
    #msk_[ap.pix_mask] = False
    #msk_[~np.isfinite(ap.orderflat)] = 0
    msk[ap.pix_mask] = True

    msk[:4] = True
    msk[-4:] = True
    msk[:,:4] = True
    msk[:,-4:] = True

if 0:
    xc, yc, v, std = estimate_background(data, msk, di=24, min_pixel=40)

    nx = ny = 2048
    # ZI = get_interpolated(nx, ny, xc, yc, v)

    ZI2 = get_interpolated_alglib(nx, ny, xc, yc, v, nlayer=2)

    ZI3 = get_interpolated_cubic(nx, ny, xc, yc, v)

    if 0:
        #ds9.view(np.ma.array(data, mask=msk).filled(np.nan))
        ds9.view(data)

    data_shifted = ap.get_shifted(data, divide_orderflat=True)

    if 0:
        s = ap.get_simple_spec(data_shifted[0]/data_shifted[1],
                               0, 1)
        import pickle
        pickle.dump(s, open("mean_sky_%s.pickle" % band,"w"))

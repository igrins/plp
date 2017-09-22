from __future__ import print_function

import os
import numpy as np

from igrins.libs.path_info import IGRINSPath
import astropy.io.fits as pyfits

from ..libs.products import PipelineProducts

from .argh_helper import argh

# def _run_order_main(o):
#     print o


def _run_order_main(args):
    o, x, y, s, g_list0, logi = args
    print(o)

    xmsk = (800 < x) & (x < 2048-800)

    # check if there is enough pixels to derive new slit profile
    if len(s[xmsk]) > 8000: # FIXME : ?? not sure if this was what I meant?
        from ..libs.slit_profile_model import derive_multi_gaussian_slit_profile

        g_list = derive_multi_gaussian_slit_profile(y[xmsk], s[xmsk])
    else:
        g_list = g_list0

    if len(x) < 1000:
        # print "skipping"
        # def _f(order, xpixel, slitpos):
        #     return g_list(slitpos)

        # func_dict[o] = g_list0
        return None

    if 0:

        from ..libs import slit_profile_model as slit_profile_model
        debug_func = slit_profile_model.get_debug_func()
        debug_func(g_list, g_list, y, s)

    from ..libs.slit_profile_2d_model import get_varying_conv_gaussian_model
    Varying_Conv_Gaussian_Model = get_varying_conv_gaussian_model(g_list)
    vcg = Varying_Conv_Gaussian_Model()

    if 0:
        def _vcg(y):
            centers = np.zeros_like(y) + 1024
            return vcg(centers, y)

        debug_func(_vcg, _vcg, y, s)

    from astropy.modeling import fitting

    fitter = fitting.LevMarLSQFitter()
    t = fitter(vcg, x, y, s, maxiter=100000)

    # func_dict[o] = t

    # print "saveing figure"

    if logi is not None:
        logi.submit("raw_data_scatter",
                    (x, y, s))

        logi.submit("profile_sub_scatter",
                    (x, y, s-vcg(x, y)),
                    label="const. model")

        logi.submit("profile_sub_scatter",
                    (x, y, s-t(x, y)),
                    label="varying model")

    # return t
    return g_list.parameters, t.parameters


def extractor_factory(recipe_name):
    @argh.arg("-b", "--bands", default="HK", choices=["H", "K", "HK"])
    @argh.arg("-s", "--starting-obsids", default=None)
    @argh.arg("-g", "--groups", default=None)
    @argh.arg("-c", "--config-file", default="recipe.config")
    @argh.arg("-d", "--debug-output", default=False)
    @argh.arg("--wavelength-increasing-order", default=False)
    @argh.arg("--cr-rejection-thresh", default=30.)
    @argh.arg("--frac-slit", default="0,1")
    @argh.arg("--fill-nan", default=None)
    @argh.arg("--lacosmics-thresh", default=None)
    @argh.arg("--lacosmic-thresh", default=None)
    @argh.arg("--conserve-2d-flux", default=True)
    @argh.arg("--slit-profile-mode", 
              #choices=['auto','simple', "gauss", "gauss2d"],
              default="auto")
    @argh.arg("--subtract-interorder-background", default=False)
    @argh.arg("--extraction-mode", choices=['auto','simple', "optimal"],
              default="auto")
    @argh.arg("--n-process", default=1)
    @argh.arg("--basename-postfix", default=None)
    @argh.arg("--height-2dspec", default=0)
    def extract(utdate, refdate="20140316", bands="HK",
                **kwargs
                ):
        abba_all(recipe_name, utdate, refdate=refdate, bands=bands,
                 **kwargs
                 )

    extract.__name__ = recipe_name.lower()
    return extract

a0v_ab = extractor_factory("A0V_AB")
a0v_onoff = extractor_factory("A0V_ONOFF")
stellar_ab = extractor_factory("STELLAR_AB")
stellar_onoff = extractor_factory("STELLAR_ONOFF")
extended_ab = extractor_factory("EXTENDED_AB")
extended_onoff = extractor_factory("EXTENDED_ONOFF")



def abba_all(recipe_name, utdate, refdate="20140316", bands="HK",
             **kwargs
             ):

    starting_obsids = kwargs.pop("starting_obsids")
    groups = kwargs.pop("groups")

    from ..libs.igrins_config import IGRINSConfig
    config = IGRINSConfig(kwargs.pop("config_file"))

    if not bands in ["H", "K", "HK"]:
        raise ValueError("bands must be one of 'H', 'K' or 'HK'")

    fn = config.get_value('RECIPE_LOG_PATH', utdate)
    from ..libs.recipes import Recipes #load_recipe_list, make_recipe_dict
    recipe = Recipes(fn, allow_duplicate_groups=False)

    if starting_obsids is not None:
        if groups is not None:
            raise ValueError("starting_obsid option is not allowed if groups option is used")
            
        starting_obsids = list(map(int, starting_obsids.split(",")))
        selected = recipe.select_fnmatch(recipe_name, starting_obsids)

    elif groups is not None:
        groups = [_.strip() for _ in groups.split(",")]

        selected = recipe.select_fnmatch_by_groups(recipe_name, groups)

    else:
        selected = recipe.select_fnmatch_by_groups(recipe_name)

    if not selected:
        print("no recipe of matching arguments is found")

    frac_slit = list(map(float, kwargs["frac_slit"].split(",")))
    if len(frac_slit) !=2:
        raise ValueError("frac_slit must be two floats separated by comma")

    kwargs["frac_slit"] = frac_slit

    lacosmics_thresh = kwargs.pop("lacosmics_thresh")
    if lacosmics_thresh is not None:
        msg = ("'--lacosmics-thresh' is deprecated, "
               "please use '--lacosmic-thresh'")

        if kwargs["lacosmic_thresh"] is None:
            kwargs["lacosmic_thresh"] = float(lacosmics_thresh)
            print(msg)
        else:
            raise ValueError(msg)
    else:
        if kwargs["lacosmic_thresh"] is None:
            kwargs["lacosmic_thresh"] = 0.
        else:
            kwargs["lacosmic_thresh"] = float(kwargs["lacosmic_thresh"])

    process_abba_band = ProcessABBABand(utdate, refdate,
                                        config,
                                        **kwargs).process

    if len(selected) == 0:
        print("No entry with given recipe is found : %s" % recipe_name)

    for band in bands:
        for s in selected:
            recipe_name, obsids, frametypes = s[:3]
            groupname = s[-1]["GROUP1"]
            process_abba_band(recipe_name, band,
                              groupname, obsids, frametypes,
                              #do_interactive_figure=interactive
                              )

from ..libs.products import PipelineStorage

class ProcessABBABand(object):
    def __init__(self, utdate, refdate, config,
                 frac_slit=None,
                 cr_rejection_thresh=100,
                 debug_output=False,
                 wavelength_increasing_order=False,
                 fill_nan=None,
                 lacosmic_thresh=0,
                 subtract_interorder_background=False,
                 conserve_2d_flux=False,
                 slit_profile_mode="auto",
                 extraction_mode="auto",
                 n_process=1,
                 basename_postfix=None,
                 height_2dspec=0):
        """
        cr_rejection_thresh : pixels that deviate significantly from the profile are excluded.
        """
        self.utdate = utdate
        self.refdate = refdate
        self.config = config

        self.igr_path = IGRINSPath(config, utdate)

        self.igr_storage = PipelineStorage(self.igr_path)

        self.frac_slit = frac_slit
        self.cr_rejection_thresh = cr_rejection_thresh
        self.lacosmic_thresh = lacosmic_thresh
        self.debug_output = debug_output

        self.fill_nan=fill_nan

        self.wavelength_increasing_order = wavelength_increasing_order

        self.subtract_interorder_background = subtract_interorder_background

        _ = self.parse_slit_profile_mode(slit_profile_mode)
        self.slit_profile_mode, self.slit_profile_options = _

        self.extraction_mode = extraction_mode

        self.n_process = n_process
        self.basename_postfix = basename_postfix

        self.height_2dspec = height_2dspec

    def parse_slit_profile_mode(self, slit_profile_mode):
        import re
        p = re.compile(r"^(\w+)(\([^\)]*\))?$")
        m = p.match(slit_profile_mode)
        if m:
            profile_mode, _profile_options = m.groups()
            if _profile_options is None:
                profile_options = {}
            else:
                try:
                    profile_options = eval("dict%s" % _profile_options)
                except Exception:
                    raise ValueError("unrecognized option: " + slit_profile_mode)
        else:
            raise ValueError("unrecognized option: " + slit_profile_mode)

        return profile_mode, profile_options

    def _estimate_slit_profile_1d(self, extractor, ap,
                                  data_minus_flattened,
                                  x1=800, x2=2048-800,
                                  do_ab=True):
        """
        return a profile function

        def profile(order, x_pixel, y_slit_pos):
            return profile_value

        """
        _ = extractor.extract_slit_profile(ap,
                                           data_minus_flattened,
                                           x1=x1, x2=x2)
        bins, hh0, slit_profile_list = _

        if do_ab:
            profile_x, profile_y = extractor.get_norm_profile_ab(bins, hh0)
        else:
            profile_x, profile_y = extractor.get_norm_profile(bins, hh0)

        self.store_profile(extractor.mastername, # obj_filenames[0],
                           ap.orders, slit_profile_list,
                           profile_x, profile_y)

        if do_ab:
            profile = extractor.get_profile_func_ab(profile_x, profile_y)
        else:
            profile = extractor.get_profile_func(profile_x, profile_y)

        return profile

    def _derive_data_for_slit_profile(self, extractor, ap,
                                      data_minus_flattened,
                                      spec1d):

        def expand_1dspec_to_2dspec(s1d, o2d, min_order=None):
            mmm = (o2d > 0) & (o2d < 999)
            xi = np.indices(mmm.shape)[-1]
            if min_order is None:
                min_order = o2d[mmm].min()
            indx = (o2d[mmm]-min_order)*2048 + xi[mmm]

            s2 = np.empty(mmm.shape, dtype=float)
            s2.fill(np.nan)
            s2[mmm] = np.take(s1d, indx)

            return s2

        omap = extractor.ordermap
        # omap = extractor.ordermap_bpixed (?)

        # correction factors for aperture width
        ds0 = np.array([ap(o, ap.xi, 1.) - ap(o, ap.xi, 0.) for o in ap.orders])
        ds = ds0 / 50. # 50 is just a typical width.

        # try to estimate threshold to mask the spectra
        s_max = np.nanpercentile(spec1d / ds0, 90) # mean counts per pixel
        s_cut = 0.03 * s_max # 3 % of s_max

        ss_cut = s_cut * ds0

        ss = np.ma.array(spec1d, mask=(spec1d < ss_cut)).filled(np.nan)

        # omap[0].data

        s2d = expand_1dspec_to_2dspec(ss/ds, omap)

        ods = data_minus_flattened/s2d

        return ods

    def _estimate_slit_profile_glist(self, extractor, ap,
                                     ods, 
                                     # spec1d,
                                     x1=800, x2=2048-800,
                                     do_ab=True):
        """
        return a profile function. This has a signature of

        def profile(y_slit_pos):
            return profile_value

        """


        slit_profile_options = self.slit_profile_options.copy()
        n_comp = slit_profile_options.pop("n_comp", None)
        stddev_list = slit_profile_options.pop("stddev_list", None)
        if slit_profile_options:
            msgs = ["unrecognized options: %s" 
                    % self.slit_profile_options,
                    "\n",
                    "Available options are: n_comp, stddev_list"]

            raise ValueError("".join(msgs))

        omap, slitpos = extractor.ordermap_bpixed, extractor.slitpos_map

        msk1 = np.isfinite(ods) # & bias_mask

        x_min, x_max = x1, x2
        y_min, y_max = 128, 2048-128

        xx = np.array([ap(o, 1024, .5) for o in ap.orders])
        i1, i2 = np.searchsorted(xx, [y_min, y_max])
        o1, o2 = ap.orders[i1], ap.orders[i2]
        msk2 = (o1 < omap) & (omap < o2)

        msk2[:, :x_min] = False
        msk2[:, x_max:] = False


        msk = msk1 & msk2 # & (slitpos < 0.5)

        ods_mskd = ods[msk]
        s_mskd = slitpos[msk]

        from ..libs.slit_profile_model import derive_multi_gaussian_slit_profile
        g_list0 = derive_multi_gaussian_slit_profile(s_mskd, ods_mskd,
                                                     n_comp=n_comp,
                                                     stddev_list=stddev_list)

        return g_list0


    def _estimate_slit_profile_gauss_2d(self, extractor, ap,
                                        ods,
                                        g_list0,
                                        # spec1d,
                                        x1=800, x2=2048-800,
                                        do_ab=True):
        _, xx = np.indices(ods.shape)

        msk1 = np.isfinite(ods) # & np.isfinite(ode) & bias_mask

        from ..libs.slit_profile_2d_model import Logger
        logger = Logger("test.pdf")

        omap, slitpos = extractor.ordermap_bpixed, extractor.slitpos_map

        # def oo(x_pixel, slitpos):
        #     return g_list0(slitpos)

        # def _run_order(o):
        #     print o

        # if 0:
        #     msk2 = omap == o

        #     msk = msk1 & msk2 # & (slitpos < 0.5)

        #     x = xx[msk]
        #     y = slitpos[msk]
        #     s = ods[msk]

        #     xmsk = (800 < x) & (x < 2048-800)

        #     # check if there is enough pixels to derive new slit profile
        #     if len(s[xmsk]) > 8000:
        #         from igrins.libs.slit_profile_model import derive_multi_gaussian_slit_profile

        #         g_list = derive_multi_gaussian_slit_profile(y[xmsk], s[xmsk])
        #     else:
        #         g_list = g_list0

        #     if len(x) < 1000:
        #         # print "skipping"
        #         # def _f(order, xpixel, slitpos):
        #         #     return g_list(slitpos)

        #         # func_dict[o] = g_list0
        #         return oo

        #     if 0:

        #         import igrins.libs.slit_profile_model as slit_profile_model
        #         debug_func = slit_profile_model.get_debug_func()
        #         debug_func(g_list, g_list, y, s)

        #     from igrins.libs.slit_profile_2d_model import get_varying_conv_gaussian_model
        #     Varying_Conv_Gaussian_Model = get_varying_conv_gaussian_model(g_list)
        #     vcg = Varying_Conv_Gaussian_Model()

        #     if 0:
        #         def _vcg(y):
        #             centers = np.zeros_like(y) + 1024
        #             return vcg(centers, y)

        #         debug_func(_vcg, _vcg, y, s)

        #     from astropy.modeling import fitting

        #     fitter = fitting.LevMarLSQFitter()
        #     t = fitter(vcg, x, y, s, maxiter=100000)

        #     # func_dict[o] = t

        #     # print "saveing figure"
        #     logi = logger.open("slit_profile_2d_conv_gaussian",
        #                        ("basename", o))

        #     logi.submit("raw_data_scatter",
        #                 (x, y, s))

        #     logi.submit("profile_sub_scatter",
        #                 (x, y, s-vcg(x, y)),
        #                 label="const. model")

        #     logi.submit("profile_sub_scatter",
        #                 (x, y, s-t(x, y)),
        #                 label="varying model")

        #     return t

        print("deriving 2d slit profiles")

        args = []
        for o in ap.orders:
            msk2 = omap == o

            msk = msk1 & msk2 # & (slitpos < 0.5)

            x = xx[msk]
            y = slitpos[msk]
            s = ods[msk]

            logi = logger.open("slit_profile_2d_conv_gaussian",
                               ("basename", o))

            args.append((o, x, y, s, g_list0, logi))


        # n_process = 4
        if self.n_process > 1:
            from multiprocessing import Pool
            #from multiprocessing.pool import ThreadPool as Pool
            p = Pool(self.n_process)
            _ = p.map(_run_order_main, args)
        else:
            _ = []
            for a in args:
                r = _run_order_main(a)
                _.append(r)

        print("done")
        # func_dict.update((k, v) )

        from ..libs.slit_profile_2d_model import get_varying_conv_gaussian_model

        func_dict = {}
        for o, v in zip(ap.orders, _):
            if v is None: continue

            g_list_parameters, vcg_parameters = v
            g_list = type(g_list0)()
            g_list.parameters = g_list_parameters

            Varying_Conv_Gaussian_Model = get_varying_conv_gaussian_model(g_list)
            vcg = Varying_Conv_Gaussian_Model()
            vcg.parameters = vcg_parameters

            func_dict[o] = vcg

        for key in sorted(logger.instances.keys()):
            logger.finalize(key)

        logger.pdf.close()

        # for o in ap.orders:
        #     print o

        #     logi.close()

        def profile(order, x_pixel, slitpos):
            # print "profile", order, func_dict.keys()
            def oo(x_pixel, slitpos):
                return g_list0(slitpos)

            return func_dict.get(order, oo)(x_pixel, slitpos)

        return profile


    def _extract_spec_using_profile(self, extractor,
                                    ap, profile_map,
                                    variance_map,
                                    variance_map0,
                                    data_minus_flattened,
                                    ordermap,
                                    slitpos_map,
                                    slitoffset_map,
                                    debug=False,
                                    mode="optimal"):

        # This is used to test spec-extraction without slit-offset-map

        # slitoffset_map_extract = "none"
        slitoffset_map_extract = slitoffset_map

        shifted = extractor.get_shifted_all(ap,
                                            profile_map,
                                            variance_map,
                                            data_minus_flattened,
                                            slitoffset_map_extract,
                                            debug=self.debug_output)

        # for khjeong
        weight_thresh = None
        remove_negative = False

        _ = extractor.extract_spec_stellar(ap, shifted,
                                           weight_thresh,
                                           remove_negative)

        s_list, v_list = _

        # if self.debug_output:
        #     self.save_debug_output()

        # make synth_spec : profile * spectra
        synth_map = extractor.make_synth_map(\
            ap, profile_map, s_list,
            ordermap=ordermap,
            slitpos_map=slitpos_map,
            slitoffset_map=slitoffset_map)


        # update variance map
        variance_map = extractor.get_updated_variance(\
            variance_map, variance_map0, synth_map)

        # get cosmicray mask
        sig_map = np.abs(data_minus_flattened - synth_map)/variance_map**.5

        cr_mask = np.abs(sig_map) > self.cr_rejection_thresh

        if self.lacosmic_thresh > 0:
            from ..libs.lacosmics import get_cr_mask

            # As our data is corrected for orderflat, it
            # actually amplifies order boundary so that they
            # can be more easily picked as CRs.  For now, we
            # use a workaround by multiplying by the
            # orderflat. But it would be better to improve it.

            cosmic_input = sig_map.copy() *extractor.orderflat
            cosmic_input[~np.isfinite(data_minus_flattened)] = np.nan
            cr_mask_cosmics = get_cr_mask(cosmic_input,
                                          readnoise=self.lacosmic_thresh)

            cr_mask = cr_mask | cr_mask_cosmics


        data_minus_flattened = np.ma.array(data_minus_flattened,
                                           mask=cr_mask).filled(np.nan)

        # extract spec

        # profile_map is not used for shifting.
        shifted = extractor.get_shifted_all(ap, profile_map,
                                            variance_map,
                                            data_minus_flattened,
                                            slitoffset_map,
                                            debug=False)

        _ = extractor.extract_spec_stellar(ap, shifted,
                                           weight_thresh,
                                           remove_negative)

        s_list, v_list = _

        if self.extraction_mode == "simple":

            print("doing simple extraction")
            if self.extraction_mode in ["optimal"]:
                msg = ("optimal extraction is not supported "
                       "for extended source")
                print(msg)

            synth_map = extractor.make_synth_map(
                ap, profile_map, s_list,
                ordermap=ordermap,
                slitpos_map=slitpos_map,
                slitoffset_map=slitoffset_map)

            nan_msk = ~np.isfinite(data_minus_flattened)
            data_minus_flattened[nan_msk] = synth_map[nan_msk]
            variance_map[nan_msk] = 0. # can we replace it with other values??

            # profile_map is not used for shifting.
            shifted = extractor.get_shifted_all(ap, profile_map,
                                                variance_map,
                                                data_minus_flattened,
                                                slitoffset_map,
                                                debug=False)

            _ = extractor.extract_spec_simple(ap, shifted)
            s_list, v_list = _

            # regenerate synth_map using the new s_list, which will be saved.
            synth_map = extractor.make_synth_map(
                ap, profile_map, s_list,
                ordermap=ordermap,
                slitpos_map=slitpos_map,
                slitoffset_map=slitoffset_map)

        elif self.extraction_mode in ["auto", "optimal"]:
            pass
        else:
            raise RuntimeError("")
            







        # assemble aux images that can be used by debug output
        aux_images = dict(sig_map=sig_map,
                          synth_map=synth_map,
                          shifted=shifted)


        return s_list, v_list, cr_mask, aux_images


    def process(self, recipe, band, groupname, obsids, frametypes,
                conserve_2d_flux=True):

        igr_storage = self.igr_storage


        if "_" in recipe:
            target_type, nodding_type = recipe.split("_")
        else:
            raise ValueError("Unknown recipe : %s" % recipe)

        if target_type in ["A0V", "STELLAR"]:
            IF_POINT_SOURCE = True
        elif target_type in ["EXTENDED"]:
            IF_POINT_SOURCE = False
        else:
            raise ValueError("Unknown recipe : %s" % recipe)

        if nodding_type not in ["AB", "ONOFF"]:
            raise ValueError("Unknown recipe : %s" % recipe)

        DO_STD = (target_type == "A0V")
        DO_AB = (nodding_type == "AB")



        from .recipe_extract_base import RecipeExtractBase

        # mastername = extractor.obj_filenames[0]
        mastername = self.igr_path.get_basename(band, groupname)

        extractor = RecipeExtractBase(self.utdate, band,
                                      mastername,
                                      obsids, frametypes,
                                      self.config,
                                      ab_mode=DO_AB)

        _ = extractor.get_data_variance(destripe_pattern=64,
                                        use_destripe_mask=True,
                                        sub_horizontal_median=True)

        data_minus, variance_map, variance_map0 = _

        if self.subtract_interorder_background:
            print("### doing sky subtraction")
            data_minus_sky = extractor.estimate_interorder_background(\
                data_minus, extractor.sky_mask)

            data_minus -= data_minus_sky

        data_minus_flattened = data_minus / extractor.orderflat
        data_minus_flattened_orig = data_minus_flattened.copy()

        ordermap_bpixed = extractor.ordermap_bpixed

        ap = extractor.get_aperture()

        ordermap = extractor.ordermap
        slitpos_map = extractor.slitpos_map

        slitoffset_map = extractor.slitoffset_map

        if 1:


            if IF_POINT_SOURCE: # if point source


                profile = self._estimate_slit_profile_1d(extractor, ap,
                                                         data_minus_flattened,
                                                         x1=800, x2=2048-800,
                                                         do_ab=DO_AB)
                profile_map = extractor.make_profile_map(ap,
                                                         profile,
                                                         frac_slit=self.frac_slit)


                # extract spec
                _ = self._extract_spec_using_profile(extractor,
                                                     ap, profile_map,
                                                     variance_map,
                                                     variance_map0,
                                                     data_minus_flattened,
                                                     ordermap,
                                                     slitpos_map,
                                                     slitoffset_map,
                                                     debug=False)

                s_list, v_list, cr_mask, aux_images = _


                if self.slit_profile_mode in ["gauss", "gauss2d"]:
                    # now try to derive the n-gaussian profile
                    print("updating profile using the multi gauss fit")

                    ods = self._derive_data_for_slit_profile(extractor, ap,
                                                             data_minus_flattened,
                                                             s_list)

                    glist = self._estimate_slit_profile_glist(
                        extractor, ap,
                        ods,
                        # s_list, # bias_mask,
                        x1=800, x2=2048-800,
                        do_ab=True)

                    if self.slit_profile_mode == "gauss2d":
                        profile = self._estimate_slit_profile_gauss_2d(extractor, ap,
                                                                       ods,
                                                                       glist,
                                                                       # s_list,
                                                                       x1=800, x2=2048-800,
                                                                       do_ab=True)
                    elif self.slit_profile_mode == "gauss":
                        def profile(order, x_pixel, slitpos):
                            return glist(slitpos)
                    else:
                        raise ValueError("unexpected slit_profile_mode: %s" % self.slit_profile_mode)

                    profile_map = extractor.make_profile_map(ap,
                                                             profile,
                                                             frac_slit=self.frac_slit)

                    _ = self._extract_spec_using_profile(extractor,
                                                         ap, profile_map,
                                                         variance_map,
                                                         variance_map0,
                                                         data_minus_flattened,
                                                         ordermap,
                                                         slitpos_map,
                                                         slitoffset_map,
                                                         debug=False)

                    s_list, v_list, cr_mask, aux_images = _


                sig_map = aux_images["sig_map"]
                synth_map = aux_images["synth_map"]
                shifted = aux_images["shifted"]

                if 0: # save aux files
                    synth_map = ap.make_synth_map(ordermap, slitpos_map,
                                                  profile_map, s_list,
                                                  slitoffset_map=slitoffset_map
                                                  )

                    shifted = extractor.get_shifted_all(ap, profile_map,
                                                        variance_map,
                                                        synth_map,
                                                        slitoffset_map,
                                                        debug=False)



            else: # if extended source
                from scipy.interpolate import UnivariateSpline
                if DO_AB:
                    delta = 0.01
                    profile_ = UnivariateSpline([0, 0.5-delta, 0.5+delta, 1],
                                                [1., 1., -1., -1.],
                                                k=1, s=0,
                                                bbox=[0, 1])
                else:
                    profile_ = UnivariateSpline([0, 1], [1., 1.],
                                                k=1, s=0,
                                                bbox=[0, 1])

                def profile(o, x, slitpos):
                    return profile_(slitpos)

                profile_map = ap.make_profile_map(ordermap,
                                                  slitpos_map,
                                                  profile)


                if self.frac_slit is not None:
                    frac1, frac2 = min(self.frac_slit), max(self.frac_slit)
                    slitpos_msk = (slitpos_map < frac1) | (slitpos_map > frac2)
                    profile_map[slitpos_msk] = np.nan

                # we need to update the variance map by rejecting
                # cosmic rays, but it is not clear how we do this
                # for extended source.
                #variance_map2 = variance_map

                # detect CRs

                if self.lacosmic_thresh > 0:

                    from ..libs.lacosmics import get_cr_mask

                    from ..libs.cosmics import cosmicsimage

                    # As our data is corrected for orderflat, it
                    # actually amplifies order boundary so that they
                    # can be more easily picked as CRs.  For now, we
                    # use a workaround by multiplying by the
                    # orderflat. But it would be better to improve it.

                    cosmic_input = data_minus/(variance_map**.5) *extractor.orderflat
                    lacosmic_thresh = self.lacosmic_thresh

                    cr_mask_p = get_cr_mask(cosmic_input,
                                            readnoise=lacosmic_thresh)

                    cr_mask_m = get_cr_mask(-cosmic_input,
                                            readnoise=lacosmic_thresh)

                    cr_mask = cr_mask_p | cr_mask_m

                else:
                    cr_mask = np.zeros(data_minus_flattened.shape,
                                       dtype=bool)

                #variance_map[cr_mask] = np.nan


                data_minus_flattened_orig = data_minus_flattened
                data_minus_flattened = data_minus_flattened_orig.copy()
                data_minus_flattened[cr_mask] = np.nan

                # extract spec

                shifted = extractor.get_shifted_all(ap,
                                                    profile_map,
                                                    variance_map,
                                                    data_minus_flattened,
                                                    slitoffset_map,
                                                    debug=self.debug_output)

                # for khjeong
                weight_thresh = None
                remove_negative = False

                _ = extractor.extract_spec_simple(ap, shifted)
                # _ = extractor.extract_spec_stellar(ap, shifted,
                #                                    weight_thresh,
                #                                    remove_negative)


                s_list, v_list = _

                # if self.debug_output:
                #     hdu_list = pyfits.HDUList()
                #     hdu_list.append(pyfits.PrimaryHDU(data=data_shft))
                #     hdu_list.append(pyfits.ImageHDU(data=variance_map_shft))
                #     hdu_list.append(pyfits.ImageHDU(data=profile_map_shft))
                #     hdu_list.append(pyfits.ImageHDU(data=ordermap_bpixed))
                #     #hdu_list.append(pyfits.ImageHDU(data=msk1_shft.astype("i")))
                #     #hdu_list.append(pyfits.ImageHDU(data=np.array(s_list)))
                #     hdu_list.writeto("test0.fits", clobber=True)





            if 1:
                # calculate S/N per resolution
                sn_list = []
                for wvl, s, v in zip(extractor.wvl_solutions,
                                     s_list, v_list):

                    dw = np.gradient(wvl)
                    pixel_per_res_element = (wvl/40000.)/dw
                    #print pixel_per_res_element[1024]
                    # len(pixel_per_res_element) = 2047. But we ignore it.
                    sn = (s/v**.5)*(pixel_per_res_element**.5)

                    sn_list.append(sn)


        from ..libs.products import PipelineImage as Image
        if self.debug_output:
            image_list = [Image([("EXTNAME", "DATA_CORRECTED")],
                                data_minus_flattened),
                          Image([("EXTNAME", "DATA_UNCORRECTED")],
                                data_minus_flattened_orig),
                          Image([("EXTNAME", "VARIANCE_MAP")],
                                variance_map),
                          Image([("EXTNAME", "CR_MASK")],
                                cr_mask)]


            shifted_image_list = [Image([("EXTNAME", "DATA_CORRECTED")],
                                        shifted["data"]),
                                  Image([("EXTNAME", "VARIANCE_MAP")],
                                        shifted["variance_map"]),
                                  ]

            if IF_POINT_SOURCE: # if point source
                image_list.extend([Image([("EXTNAME", "SIG_MAP")],
                                         sig_map),
                                   Image([("EXTNAME", "SYNTH_MAP")],
                                         synth_map),
                                   Image([("EXTNAME", "PROFILE_MAP")],
                                         profile_map),
                                   ])

            if self.subtract_interorder_background:
                image_list.append(Image([("EXTNAME", "INTERORDER_BG")],
                                        data_minus_sky))

            self.store_processed_inputs(igr_storage, mastername,
                                        image_list,
                                        shifted_image_list)



        self.store_1dspec(igr_storage,
                          extractor,
                          mastername,
                          v_list, sn_list, s_list)

        if not IF_POINT_SOURCE: # if point source
            cr_mask = None

        self.store_2dspec(igr_storage,
                          extractor,
                          mastername,
                          shifted["data"],
                          shifted["variance_map"],
                          ordermap_bpixed,
                          cr_mask=cr_mask,
                          conserve_flux=conserve_2d_flux,
                          height_2dspec=self.height_2dspec)


        if DO_STD:

            wvl = self.refine_wavelength_solution(extractor.wvl_solutions)

            a0v_flattened_data = self.get_a0v_flattened(extractor, ap,
                                                        s_list,
                                                        wvl)

            a0v_flattened = a0v_flattened_data[0][1]  #"flattened_spec"]

            self.store_a0v_results(igr_storage, extractor,
                                   mastername,
                                   a0v_flattened_data)


    def refine_wavelength_solution(self, wvl_solutions):
        return wvl_solutions


    def store_profile(self, mastername,
                      orders, slit_profile_list,
                      profile_x, profile_y):
        ## save profile
        igr_storage = self.igr_storage
        r = PipelineProducts("slit profile for point source")
        from ..libs.storage_descriptions import SLIT_PROFILE_JSON_DESC
        from ..libs.products import PipelineDict
        slit_profile_dict = PipelineDict(orders=orders,
                                         slit_profile_list=slit_profile_list,
                                         profile_x=profile_x,
                                         profile_y=profile_y)
        r.add(SLIT_PROFILE_JSON_DESC, slit_profile_dict)

        igr_storage.store(r,
                          mastername=mastername,
                          masterhdu=None,
                          basename_postfix=self.basename_postfix)

    def save_debug_output(self, shifted):
        """
        Save debug output.
        This method currently does not work and need to be fixed.
        """
        # hdu_list = pyfits.HDUList()
        # hdu_list.append(pyfits.PrimaryHDU(data=data_minus_flattened))
        # hdu_list.append(pyfits.ImageHDU(data=variance_map))
        # hdu_list.append(pyfits.ImageHDU(data=profile_map))
        # hdu_list.append(pyfits.ImageHDU(data=ordermap_bpixed))
        # #hdu_list.writeto("test_input.fits", clobber=True)

        if 1:
            return

        hdu_list = pyfits.HDUList()
        hdu_list.append(pyfits.PrimaryHDU(data=data_minus))
        hdu_list.append(pyfits.ImageHDU(data=extractor.orderflat))
        hdu_list.append(pyfits.ImageHDU(data=profile_map))
        hdu_list.append(pyfits.ImageHDU(data=ordermap_bpixed))
        #hdu_list.append(pyfits.ImageHDU(data=msk1_shft.astype("i")))
        #hdu_list.append(pyfits.ImageHDU(data=np.array(s_list)))
        hdu_list.writeto("test0.fits", clobber=True)




        # s_list, v_list = ap.extract_stellar(ordermap_bpixed,
        #                                     profile_map_shft,
        #                                     variance_map_shft,
        #                                     data_minus_flattened_shft,
        #                                     #slitoffset_map=slitoffset_map,
        #                                     slitoffset_map=None,
        #                                     remove_negative=True
        #                                     )


    def store_processed_inputs(self, igr_storage,
                               mastername,
                               image_list,
                               shifted_image_list):

        from ..libs.storage_descriptions import (COMBINED_IMAGE_DESC,
                                                 WVLCOR_IMAGE_DESC)
        from ..libs.products import PipelineImages

        r = PipelineProducts("1d specs")

        r.add(COMBINED_IMAGE_DESC, PipelineImages(image_list))

        igr_storage.store(r,
                          mastername=mastername,
                          masterhdu=None,
                          basename_postfix=self.basename_postfix)


        r = PipelineProducts("1d specs")

        r.add(WVLCOR_IMAGE_DESC, PipelineImages(shifted_image_list))

        igr_storage.store(r,
                          mastername=mastername,
                          masterhdu=None,
                          basename_postfix=self.basename_postfix)



    def get_wvl_header_data(self, igr_storage, extractor):
        from ..libs.storage_descriptions import SKY_WVLSOL_FITS_DESC
        fn = igr_storage.get_path(SKY_WVLSOL_FITS_DESC,
                                  extractor.basenames["wvlsol"])

        # fn = sky_path.get_secondary_path("wvlsol_v1.fits")
        f = pyfits.open(fn)

        if self.wavelength_increasing_order:
            from ..libs import iraf_helper
            header = iraf_helper.invert_order(f[0].header)
            convert_data = lambda d: d[::-1]
        else:
            header = f[0].header
            convert_data = lambda d: d

        return header, f[0].data, convert_data


    def store_1dspec(self, igr_storage,
                     extractor,
                     mastername,
                     v_list, sn_list, s_list):

        wvl_header, wvl_data, convert_data = \
                    self.get_wvl_header_data(igr_storage,
                                             extractor)


        from ..libs.load_fits import open_fits
        f_obj = open_fits(extractor.obj_filenames[0])
        f_obj[0].header.extend(wvl_header)

        # tgt_basename = extractor.pr.tgt_basename
        tgt_basename = mastername

        from ..libs.storage_descriptions import (SPEC_FITS_DESC,
                                               VARIANCE_FITS_DESC,
                                               SN_FITS_DESC)



        d = np.array(v_list)
        f_obj[0].data = convert_data(d.astype("float32"))
        fout = igr_storage.get_path(VARIANCE_FITS_DESC,
                                    tgt_basename, basename_postfix=self.basename_postfix)

        f_obj.writeto(fout, clobber=True)

        d = np.array(sn_list)
        f_obj[0].data = convert_data(d.astype("float32"))
        fout = igr_storage.get_path(SN_FITS_DESC,
                                    tgt_basename, basename_postfix=self.basename_postfix)

        f_obj.writeto(fout, clobber=True)

        d = np.array(s_list)
        f_obj[0].data = convert_data(d.astype("float32"))

        fout = igr_storage.get_path(SPEC_FITS_DESC,
                                    tgt_basename, basename_postfix=self.basename_postfix)

        hdu_wvl = pyfits.ImageHDU(data=convert_data(wvl_data),
                                  header=wvl_header)
        f_obj.append(hdu_wvl)

        f_obj.writeto(fout, clobber=True)


    def store_2dspec(self, igr_storage,
                     extractor,
                     mastername,
                     data_shft,
                     variance_map_shft,
                     ordermap_bpixed,
                     cr_mask=None,
                     conserve_flux=True,
                     height_2dspec=0):

        wvl_header, wvl_data, convert_data = \
                    self.get_wvl_header_data(igr_storage,
                                             extractor)


        from ..libs.load_fits import open_fits
        f_obj = open_fits(extractor.obj_filenames[0])
        f_obj[0].header.extend(wvl_header)

        # tgt_basename = extractor.pr.tgt_basename
        tgt_basename = mastername

        from ..libs.storage_descriptions import FLATCENTROID_SOL_JSON_DESC
        cent = igr_storage.load1(FLATCENTROID_SOL_JSON_DESC,
                                 extractor.basenames["flat_on"])

        #cent = json.load(open("calib/primary/20140525/FLAT_SDCK_20140525_0074.centroid_solutions.json"))
        _bottom_up_solutions = cent["bottom_up_solutions"]
        old_orders = extractor.get_old_orders()
        _o_s = dict(zip(old_orders, _bottom_up_solutions))
        new_bottom_up_solutions = [_o_s[o] for o in \
                                   extractor.orders_w_solutions]

        from ..libs.correct_distortion import get_flattened_2dspec

        d0_shft_list, msk_shft_list = \
                      get_flattened_2dspec(data_shft,
                                           ordermap_bpixed,
                                           new_bottom_up_solutions,
                                           conserve_flux=conserve_flux,
                                           height=height_2dspec)


        d = np.array(d0_shft_list) / np.array(msk_shft_list)
        f_obj[0].data = convert_data(d.astype("float32"))

        from ..libs.storage_descriptions import SPEC2D_FITS_DESC

        fout = igr_storage.get_path(SPEC2D_FITS_DESC,
                                    tgt_basename, basename_postfix=self.basename_postfix)

        hdu_wvl = pyfits.ImageHDU(data=convert_data(wvl_data),
                                  header=wvl_header)
        f_obj.append(hdu_wvl)

        f_obj.writeto(fout, clobber=True)

        #OUTPUT VAR2D, added by Kyle Kaplan Feb 25, 2015 to get variance map outputted as a datacube
        d0_shft_list, msk_shft_list = \
                      get_flattened_2dspec(variance_map_shft,
                                           ordermap_bpixed,
                                           new_bottom_up_solutions,
                                           conserve_flux=conserve_flux,
                                           height=height_2dspec)
        d = np.array(d0_shft_list) / np.array(msk_shft_list)
        f_obj[0].data = d.astype("float32")
        from ..libs.storage_descriptions import VAR2D_FITS_DESC
        fout = igr_storage.get_path(VAR2D_FITS_DESC,
                                    tgt_basename, basename_postfix=self.basename_postfix)
        f_obj.writeto(fout, clobber=True)


    def get_tel_interp1d_f(self, extractor, wvl_solutions):

        from ..libs.master_calib import get_master_calib_abspath
        #fn = get_master_calib_abspath("telluric/LBL_A15_s0_w050_R0060000_T.fits")
        #self.telluric = pyfits.open(fn)[1].data

        telfit_outname = "telluric/transmission-795.20-288.30-41.9-45.0-368.50-3.90-1.80-1.40.%s" % extractor.band
        telfit_outname_npy = telfit_outname+".npy"
        if 0:
            dd = np.genfromtxt(telfit_outname)
            np.save(open(telfit_outname_npy, "w"), dd[::10])

        from ..libs.a0v_flatten import TelluricTransmission
        _fn = get_master_calib_abspath(telfit_outname_npy)
        tel_trans = TelluricTransmission(_fn)

        wvl_solutions = np.array(wvl_solutions)

        w_min = wvl_solutions.min()*0.9
        w_max = wvl_solutions.max()*1.1
        def tel_interp1d_f(gw=None):
            return tel_trans.get_telluric_trans_interp1d(w_min, w_max, gw)

        return tel_interp1d_f

    def get_a0v_interp1d(self, extractor):

        from ..libs.a0v_spec import A0V
        a0v_interp1d = A0V.get_flux_interp1d(self.config)
        return a0v_interp1d


    def get_a0v_flattened(self, extractor, ap,
                          s_list, wvl):

        tel_interp1d_f = self.get_tel_interp1d_f(extractor, wvl)
        a0v_interp1d = self.get_a0v_interp1d(extractor)


        orderflat_response = extractor.orderflat_json["fitted_responses"]

        tgt_basename = extractor.pr.tgt_basename
        igr_path = extractor.igr_path
        figout = igr_path.get_section_filename_base("QA_PATH",
                                                    "flattened_"+tgt_basename) + ".pdf"

        from igrins.libs.a0v_flatten import get_a0v_flattened
        data_list = get_a0v_flattened(a0v_interp1d, tel_interp1d_f,
                                      wvl, s_list, orderflat_response,
                                      figout=figout)

        if self.fill_nan is not None:
            flattened_s = data_list[0][1]
            flattened_s[~np.isfinite(flattened_s)] = self.fill_nan

        return data_list


    def get_a0v_flattened_deprecated(self, igr_storage, extractor, ap,
                          s_list):

        from ..libs.storage_descriptions import ORDER_FLAT_JSON_DESC
        prod = igr_storage.load1(ORDER_FLAT_JSON_DESC,
                                 extractor.basenames["flat_on"])

        new_orders = prod["orders"]
        # fitted_response = orderflat_products["fitted_responses"]
        i1i2_list_ = prod["i1i2_list"]

        #order_indices = []
        i1i2_list = []

        for o in ap.orders:
            o_new_ind = np.searchsorted(new_orders, o)
            #order_indices.append(o_new_ind)
            i1i2_list.append(i1i2_list_[o_new_ind])


        from ..libs.a0v_spec import (A0VSpec, TelluricTransmission,
                                   get_a0v, get_flattend)
        a0v_spec = A0VSpec()
        tel_trans = TelluricTransmission()

        wvl_limits = []
        for wvl_ in extractor.wvl_solutions:
            wvl_limits.extend([wvl_[0], wvl_[-1]])

        dwvl = abs(wvl_[0] - wvl_[-1])*0.2 # padding

        wvl1 = min(wvl_limits) - dwvl
        wvl2 = max(wvl_limits) + dwvl

        a0v_wvl, a0v_tel_trans, a0v_tel_trans_masked = get_a0v(a0v_spec, wvl1, wvl2, tel_trans)

        a0v_flattened = get_flattend(a0v_spec,
                                     a0v_wvl, a0v_tel_trans_masked,
                                     extractor.wvl_solutions, s_list,
                                     i1i2_list=i1i2_list)


        return a0v_flattened

    def store_a0v_results(self, igr_storage, extractor,
                          mastername,
                          a0v_flattened_data):

        wvl_header, wvl_data, convert_data = \
                    self.get_wvl_header_data(igr_storage,
                                             extractor)


        from ..libs.load_fits import open_fits
        f_obj = open_fits(extractor.obj_filenames[0])
        f_obj[0].header.extend(wvl_header)

        from ..libs.products import PipelineImage as Image
        image_list = [Image([("EXTNAME", "SPEC_FLATTENED")],
                            convert_data(a0v_flattened_data[0][1]))]

        for ext_name, data in a0v_flattened_data[1:]:
            image_list.append(Image([("EXTNAME", ext_name.upper())],
                                    convert_data(data)))


        from ..libs.products import PipelineImages #Base
        from ..libs.storage_descriptions import SPEC_FITS_FLATTENED_DESC

        r = PipelineProducts("flattened 1d specs")
        r.add(SPEC_FITS_FLATTENED_DESC, PipelineImages(image_list))

        # mastername = extractor.obj_filenames[0]

        igr_storage.store(r,
                          mastername=mastername,
                          masterhdu=f_obj[0],
                          basename_postfix=self.basename_postfix)

        tgt_basename = extractor.pr.tgt_basename
        extractor.db["a0v"].update(extractor.band, tgt_basename)

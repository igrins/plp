def lazyprop(fn):
    attr_name = '_lazy_' + fn.__name__
    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazyprop


class OnedSpecHelper(object):
    def __init__(self, igr_storage, basename):
        self.basename = basename
        self.igr_storage = igr_storage

    @lazyprop
    def _spec_hdu_list(self):
        from libs.storage_descriptions import SPEC_FITS_DESC
        spec_hdu_list = self.igr_storage.load1(SPEC_FITS_DESC,
                                               self.basename,
                                               return_hdu_list=True)
        return spec_hdu_list

    @lazyprop
    def spec(self):
        spec = list(self._spec_hdu_list[0].data)
        return spec

    @lazyprop
    def um(self):
        um = list(self._spec_hdu_list[1].data)
        return um

    @lazyprop
    def sn(self):
        from libs.storage_descriptions import SN_FITS_DESC
        sn_ = self.igr_storage.load1(SN_FITS_DESC,
                                     self.basename)
        sn = list(sn_.data)
        return sn

    @lazyprop
    def flattened(self):

        from libs.storage_descriptions import SPEC_FITS_FLATTENED_DESC
        telluric_cor_ = self.igr_storage.load1(SPEC_FITS_FLATTENED_DESC,
                                               self.basename)

        #A0V_path = ProductPath(igr_path, A0V_basename)
        #fn = A0V_path.get_secondary_path("spec_flattened.fits")
        flattened = list(telluric_cor_.data)
        return flattened

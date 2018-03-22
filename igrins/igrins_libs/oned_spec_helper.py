def lazyprop(fn):
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazyprop


class OnedSpecHelper(object):
    def __init__(self, obsset, basename_postfix=""):
        self.obsset = obsset
        self.basename_postfix = basename_postfix

    @lazyprop
    def _spec_hdu_list(self):
        spec_hdu_list = self.obsset.load("SPEC_FITS",
                                         postfix=self.basename_postfix)
        # return_hdu_list=True,
        # prevent_split=True)
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
        sn_ = self.obsset.load("SN_FITS",
                               postfix=self.basename_postfix)
        # prevent_split=True)
        sn = list(sn_[0].data)
        return sn

    @lazyprop
    def flattened_hdu_list(self):
        spec_hdu_list = self.obsset.load("SPEC_FITS_FLATTENED",
                                         postfix=self.basename_postfix)
        # return_hdu_list=True,
        # prevent_split=True)
        return spec_hdu_list

    @lazyprop
    def flattened(self):

        telluric_cor_ = self.flattened_hdu_list
        # prevent_split=True)

        flattened = list(telluric_cor_[0].data)
        return flattened

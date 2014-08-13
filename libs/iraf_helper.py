def get_wat_spec(orders, wvl_sol):
    """
    WAT header
    wvl_sol should be a list of Chebyshev polynomials.
    """

    # specN = ap beam dtype w1 dw nw z aplow aphigh
    specN_tmpl = "{ap} {beam} {dtype} {w1} {dw} {nw} {z} {aplow} {aphigh}"
    function_i_tmpl = "{wt_i} {w0_i} {ftype_i} {parameters} {coefficients}"
    specN_list = []
    from itertools import izip, count
    for ap_num, o, wsol in izip(count(1), orders, wvl_sol):
        specN = "spec%d" % ap_num

        d = dict(ap=ap_num,
                 beam=o,
                 dtype=2,
                 w1=wsol(0),
                 dw=(wsol(2047)-wsol(0))/2048.,
                 nw=2048,
                 z=0.,
                 aplow=0.,
                 aphigh=1.
                 )
        specN_str = specN_tmpl.format(**d)

        param_d = dict(c_order=wsol.degree+1, # degree in iraf def starts from 1
                       pmin=wsol.domain[0],
                       pmax=wsol.domain[1])
        d = dict(wt_i=1.,
                 w0_i=0.,
                 ftype_i=1, # chebyshev(1), legendre(2), etc.
                 parameters="{c_order} {pmin} {pmax}".format(**param_d),
                 coefficients=" ".join(map(str, wsol.parameters)))

        function_i = function_i_tmpl.format(**d)

        s = '%s = "%s %s"' % (specN, specN_str, function_i)

        specN_list.append(s)
    return specN_list

if __name__ == "__main__":

    import numpy as np
    xxx = np.linspace(1, 2048, 100)
    yyy = xxx**4

    from astropy.modeling import models, fitting
    p_init = models.Chebyshev1D(domain=[xxx[0], xxx[-1]],
                                            degree=4)
    fit_p = fitting.LinearLSQFitter()
    p = fit_p(p_init, xxx, yyy)

    wat_list = get_wat_spec([111], [p])

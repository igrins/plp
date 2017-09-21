from __future__ import print_function

def polyfitr(x, y, N, s, fev=100, w=None, diag=False, clip='both', \
                 verbose=False, plotfit=False, plotall=False):
    """Matplotlib's polyfit with weights and sigma-clipping rejection.

    :DESCRIPTION:
      Do a best fit polynomial of order N of y to x.  Points whose fit
      residuals exeed s standard deviations are rejected and the fit is
      recalculated.  Return value is a vector of polynomial
      coefficients [pk ... p1 p0].

    :OPTIONS:
        w:   a set of weights for the data; uses CARSMath's weighted polynomial
             fitting routine instead of numpy's standard polyfit.

        fev:  number of function evaluations to call before stopping

        'diag'nostic flag:  Return the tuple (p, chisq, n_iter)

        clip: 'both' -- remove outliers +/- 's' sigma from fit
              'above' -- remove outliers 's' sigma above fit
              'below' -- remove outliers 's' sigma below fit

    :REQUIREMENTS:
       :doc:`CARSMath`

    :NOTES:
       Iterates so long as n_newrejections>0 AND n_iter<fev.


     """
    # 2008-10-01 13:01 IJC: Created & completed
    # 2009-10-01 10:23 IJC: 1 year later! Moved "import" statements within func.
    # 2009-10-22 14:01 IJC: Added 'clip' options for continuum fitting
    # 2009-12-08 15:35 IJC: Automatically clip all non-finite points
    # 2010-10-29 09:09 IJC: Moved pylab imports inside this function

    #from CARSMath import polyfitw
    from numpy import polyfit, polyval, isfinite, ones, array, std
    #from pylab import plot, legend, title

    xx = array(x, copy=True)
    yy = array(y, copy=True)
    noweights = (w==None)
    if noweights:
        ww = ones(xx.shape, float)
    else:
        ww = array(w, copy=True)

    ii = 0
    nrej = 1

    if noweights:
        goodind = isfinite(xx)*isfinite(yy)
    else:
        goodind = isfinite(xx)*isfinite(yy)*isfinite(ww)

    xx = xx[goodind]
    yy = yy[goodind]
    ww = ww[goodind]

    while (ii<fev and (nrej!=0)):
        if noweights:
            p = polyfit(xx,yy,N)
        else:
            p = polyfitw(xx,yy, ww, N)
            p = p[::-1]  # polyfitw uses reverse coefficient ordering
        residual = yy - polyval(p,xx)
        stdResidual = std(residual)
        if clip=='both':
            ind =  abs(residual) <= (s*stdResidual)
        elif clip=='above':
            ind = residual < s*stdResidual
        elif clip=='below':
            ind = residual > -s*stdResidual
        else:
            ind = ones(residual.shape, bool)
        xx = xx[ind]
        yy = yy[ind]
        if (not noweights):
            ww = ww[ind]
        ii = ii + 1
        nrej = len(residual) - len(xx)
        if plotall:
            plot(x,y, '.', xx,yy, 'x', x, polyval(p, x), '--')
            legend(['data', 'fit data', 'fit'])
            title('Iter. #' + str(ii) + ' -- Close all windows to continue....')

        if verbose:
            print(str(len(x)-len(xx)) + ' points rejected on iteration #' + str(ii))

    if (plotfit or plotall):
        plot(x,y, '.', xx,yy, 'x', x, polyval(p, x), '--')
        legend(['data', 'fit data', 'fit'])
        title('Close window to continue....')

    if diag:
        chisq = ( (residual)**2 / yy ).sum()
        p = (p, chisq, ii)

    return p

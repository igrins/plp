from . cosmics import cosmicsimage

try:
    import astroscrappy
except ImportError:
    astroscrappy = None

def get_cr_mask_astroscrappy(d, gain=2.5, readnoise=10.0,
                             sigclip=5, sigfrac = 0.3, objlim = 5.0):

    c = astroscrappy.detect_cosmics(d, gain=gain, readnoise=readnoise, 
                                    sigclip=sigclip, sigfrac=sigfrac,
                                    objlim=objlim, 
                                    cleantype='medmask', 
                                    psfmodel="gaussx")

    return c[0]

def get_cr_mask_cosmics(d, gain=2.5, readnoise=10.0,
                        sigclip=5, sigfrac = 0.3, objlim = 5.0):
    c = cosmicsimage(d, gain=gain, readnoise=readnoise, 
                     sigclip=sigclip, sigfrac = 0.3, objlim = 5.0)

    c.run(maxiter = 4)
    cr_mask_cosmics = c.getmask()

    return cr_mask_cosmics

if astroscrappy is None:
    get_cr_mask = get_cr_mask_cosmics
else:
    get_cr_mask = get_cr_mask_astroscrappy


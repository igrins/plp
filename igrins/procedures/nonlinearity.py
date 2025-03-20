import numpy as np
from scipy.interpolate import BSpline

def get_nonlinearity_corrector(obsset, thresh=30_000):
    j = obsset.load_ref_data(kind="NONLINEARITY_CORRECTION")

    _bspline_nl_correction = BSpline(j["t"], j["c"], j["k"])
    def bspline_nl_correction(d):
        return _bspline_nl_correction(np.ma.array(d, mask=d>thresh).filled(np.nan))

    return bspline_nl_correction

def main():
    import igrins

    utdate = "20240429"
    recipe_log = igrins.load_recipe_log(utdate)
    entry = recipe_log.subset(obstype="TAR").iloc[0]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, num=1, clear=True)

    for band in "HK":
        obsset = igrins.get_obsset(utdate, band, entry)

        bspline_nl_correction = get_nonlinearity_corrector(obsset)

        cnt = np.linspace(0, 32_000, 100)
        ax.plot(cnt, bspline_nl_correction(cnt)/cnt, label=band)

    ax.legend()
    ax.set_xlabel("Count [ADU]")
    ax.set_ylabel("Correction")

if __name__ == '__main__':
    main()

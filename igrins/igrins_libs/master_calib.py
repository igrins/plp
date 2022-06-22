import os
import numpy as np
import json

from six.moves import configparser as ConfigParser


def get_master_calib_abspath(fn):
    return os.path.join("master_calib", fn)


# def load_thar_ref_data(ref_date, band):
#     # load spec

#     igrins_orders = {}
#     igrins_orders["H"] = range(99, 122)
#     igrins_orders["K"] = range(72, 92)

#     ref_spec_file = "arc_spec_thar_%s_%s.json" % (band, ref_date)
#     ref_id_file = "thar_identified_%s_%s.json" % (band, ref_date)

#     s_list_ = json.load(open(get_master_calib_abspath(ref_spec_file)))
#     s_list_src = [np.array(s) for s in s_list_]

#     # reference line list : from previous run
#     ref_lines_list = json.load(open(get_master_calib_abspath(ref_id_file)))

#     r = dict(ref_date=ref_date,
#              band=band,
#              ref_spec_file=ref_spec_file,
#              ref_id_file=ref_id_file,
#              ref_lines_list=ref_lines_list,
#              ref_s_list=s_list_src,
#              orders=igrins_orders[band])

#     return r



# def load_sky_ref_data(config, band):
#     # json_name = "ref_ohlines_indices_%s.json" % (ref_utdate,)
#     # fn = get_master_calib_abspath(json_name)
#     # ref_ohline_indices_map = json.load(open(fn))

#     ref_ohline_indices_map = load_ref_data(config, band,
#                                            kind="OHLINES_INDICES_JSON")

#     ref_ohline_indices = ref_ohline_indices_map[band]

#     ref_ohline_indices = dict((int(k), v) for k, v \
#                               in ref_ohline_indices.items())

#     from .oh_lines import OHLines
#     fn = get_ref_data_path(config, band, kind="OHLINES_JSON")
#     #fn = get_master_calib_abspath("ohlines.dat")
#     ohlines = OHLines(fn)

#     # from fit_gaussian import fit_gaussian_simple


#     r = dict(#ref_date=ref_utdate,
#              band=band,
#              ohlines_db = ohlines,
#              ohline_indices=ref_ohline_indices,
#              )

#     return r


def json_loader(fn):
    import json
    return json.load(open(fn))

def fits_loader(fn):
    import astropy.io.fits as pyfits
    return pyfits.open(fn)

def npy_loader(fn):
    import numpy
    return numpy.load(fn)


ref_loader_dict = {".json":json_loader,
                   ".fits":fits_loader,
                   ".npy":npy_loader}

def get_ref_loader(fn):
    import os
    fn1, ext = os.path.splitext(fn)

    loader = ref_loader_dict[ext]
    return loader


def query_ref_value_from_section(config, band, section, kind,
                                 ref_utdate=None, default=None):
    # if ref_utdate is None:
    #     ref_utdate = config.get("MASTER_CAL", "REFDATE")
    # master_cal_dir = config.get("MASTER_CAL", "MASTER_CAL_DIR")
    try:
        v = config.get(section, kind,
                       BAND=band)
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        v = default
    return v

def query_ref_value(config, band, kind, ref_utdate=None):
    if ref_utdate is None:
        ref_utdate = config.get("MASTER_CAL", "REFDATE")
    master_cal_dir = config.get("MASTER_CAL", "MASTER_CAL_DIR")
    v = config.get("MASTER_CAL", kind,
                   MASTER_CAL_DIR=master_cal_dir,
                   REFDATE=ref_utdate,
                   BAND=band)
    return v


def query_ref_data_path(config, band, kind, ref_utdate=None):
    # if ref_utdate is None:
    #     ref_utdate = config.get("MASTER_CAL", "REFDATE")
    # master_cal_dir = config.get("MASTER_CAL", "MASTER_CAL_DIR")
    # fn0 = config.get("MASTER_CAL", kind,
    #                  MASTER_CAL_DIR=master_cal_dir,
    #                  REFDATE=ref_utdate,
    #                  BAND=band)
    fn0 = query_ref_value(config, band, kind, ref_utdate=ref_utdate)

    import os
    fn = os.path.join(config.master_cal_dir, fn0)
    return fn


def load_ref_data(config, band, kind, ref_utdate=None):
    fn = query_ref_data_path(config, band, kind, ref_utdate=ref_utdate)
    loader = get_ref_loader(fn)
    return loader(fn)

# def fetch_ref_data(config, band, kind, ref_utdate=None):
#     fn = get_ref_data_path(config, band, kind, ref_utdate=ref_utdate)
#     fn1, ext = os.path.splitext(fn)

#     loader = ref_loader_dict[ext]
#     return fn, loader(fn)

####




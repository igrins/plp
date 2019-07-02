from collections import OrderedDict
from astropy.table import Table
import pandas as pd

from astropy.table import Table
from astropy.io.fits.convenience import table_to_hdu


def get_raw_value_string(c):
    """This is not used for now."""
    c._parse_value()

    return c._valuestring.strip()


def get_cards_to_store(h):
    cc = [(c.keyword, c.value) for c in h.cards
          if c.keyword not in ["SIMPLE", "COMMENT", "FITSFILE"]]
    # cc = [(c.keyword, get_raw_value_string(c), ) for c in h.cards
    #       if c.keyword not in ["SIMPLE", "COMMENT", "FITSFILE"]]

    return OrderedDict(cc)

# def get_cards_to_store(obsdate, obsid, h):
#     cc = [(c.keyword, c.value) for c in h.cards
#           if c.keyword not in ["SIMPLE", "COMMENT", "FITSFILE"]]
#     # cc = [(c.keyword, get_raw_value_string(c), ) for c in h.cards
#     #       if c.keyword not in ["SIMPLE", "COMMENT", "FITSFILE"]]

#     return OrderedDict([("OBSDATE", obsdate), ("OBSID", obsid)] + cc)


def get_header_tables_as_df(hdu_list, aux_columns=None):
    ccc = [get_cards_to_store(hdu.header) for hdu in hdu_list]

    ccc1 = [pd.Series(od) for od in ccc]
    df = pd.DataFrame(ccc1)

    if aux_columns is not None:
        for k, a in aux_columns.items():
            df[k] = a

    return df


def get_header_tables_as_hdu(hdu_list, aux_columns=None):
    df = get_header_tables_as_df(hdu_list,
                                 aux_columns=aux_columns)
    tbl = Table.from_pandas(df)
    tbl_hdu = table_to_hdu(tbl)

    return tbl_hdu


def main():
    obsdate = "20170316"
    obsid_list = [122, 123, 124, 125]
    obsid_list_ = ["{:04d}".format(o) for o in obsid_list]
    rootdir = "/home/jjlee/work/igrins/PNe/k4-47/indata/{}/".format(obsdate)
    hdu_list = [pyfits.open(rootdir + 'SDCH_{}_{}.fits'.format(obsdate, obsid))[0]
                for obsid in obsid_list_]

    aux_columns = dict(OBSID=obsid_list, OBSDATE=[obsdate] * len(obsid_list))
    df = get_header_tables_as_df(hdu_list, aux_columns)
    tbl = Table.from_pandas(df)

    from astropy.io.fits.convenience import table_to_hdu
    t = table_to_hdu(tbl)
    # t = pyfits.BinTableHDU.from_columns([pyfits.Column(c.name,
    #                                                    format=c.format,
    #                                                    array=c.data) for c in tbl.itercols()])

    if False:
        str_len = dict((colname, max(len(a) for a in arr))
                       for colname, arr in df.iteritems())

        # for (colname, arr) in df.iteritems():
        #     fmt = "{}A".format(str_len[colname])
        #     col = pyfits.Column(colname, format=fmt,
        #                         array=list(arr))

        cols = [pyfits.Column(colname, format="{}A".format(str_len[colname]),
                              array=list(arr)) for colname, arr in df.iteritems()]


if __name__ == '__main__':
    main()

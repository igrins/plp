#!/usr/bin/env python3

import sys
from pathlib import Path
import argparse
import os
import glob
from astropy.io import fits
from astropy.coordinates import SkyCoord
# import pandas as pd
# from six import StringIO

#__version__ = "2024-Apr-27"  # HJKim, original version
__version__ = "2024-Apr-29" # FRAMETYPE by POFFSET0, QOFFSET0" updated.


def write_dtlog(utdate, datadir: Path):
    # utdate = args.ut_date[0]
    # dir = args.dir[0] if args.dir else f"./indata/{utdate}"
    # fn_out = os.path.join(dir, f"IGRINS_DT_Log_{utdate}-1_H.txt")

    fn_list = sorted(datadir.glob(f"SDCH_{utdate}_*.fits"))

    if len(fn_list) == 0:
        raise RuntimeError(
            f"No SDCH* file is found in {datadir} or No {datadir} is found.")

    headers_string = ("FILENAME,OBSTIME,GROUP1,GROUP2,OBJNAME,OBJTYPE,FRAMETYPE,"
                      "EXPTIME,ROTPA,RA,DEC,AM")
    headers = headers_string.split(",")
    formats = ",".join([f"{{{s}}}" for s in headers]) + "\n"

    lines = ["IGRINS Observation Log - H band\n"]
    lines.append(headers_string + "\n")

    for fi in fn_list:

        hdu = fits.open(fi)
        hd0 = hdu[0].header

        if hd0["OBSCLASS"] == "acq":
            continue

        filename = fi.name # split("/")[-1]
        obstime = hd0["UTSTART"].split("T")[-1][:11]
        objname = hd0["OBJECT"]

        match hd0["OBSTYPE"]:
            case "DARK":
                objtype = "DARK"
                frmtype = "-"
            case "FLAT":
                objtype = "FLAT"
                if "GCALLAMP" in hd0:
                    if (hd0["GCALLAMP"] == "QH") & (hd0["GCALSHUT"] == "CLOSED"):
                        frmtype = "ON"
                    elif (hd0["GCALLAMP"] == "IRhigh") & (hd0["GCALSHUT"] == "CLOSED"):
                        frmtype = "OFF"
                    else:
                        raise RuntimeError("Check 'GCALLAMP' and 'GCALSHUT' keywords.")
                else:
                    frmtype = "ON"
                    print(f"No GCAL keywords. Manually change FRAMETYPE in DT Log for {fi}.")

            case "OBJECT":
                if hd0["OBSCLASS"] == "partnerCal":
                    if hd0["OBJECT"] == "Blank sky":
                        objtype = "SKY"
                        frmtype = "-"
                    else:
                        objtype = "STD"
                elif hd0["OBSCLASS"] == "science":
                    objtype = "TAR"
                else:
                    objtype = "SVC"
                    frmtype = "-"
            case _:
                print(f"OBSCLASS is not matched for {fi}.")

        if (objtype == "STD") | (objtype == "TAR"):
            if "POFFSET0" in hd0:
                match (hd0["POFFSET0"], hd0["QOFFSET0"]):
                    case (0, -1.25): frmtype = "A"
                    case (0, 1.25): frmtype = "B"
                    case (0, 0): frmtype = "ON"
                    case (_,_): frmtype = "OFF"
            else:
                frmtype ="A"

        
        exptime = hd0["EXPTIME"]
        pa = f'{hd0["PA"]:.2f}'
        if hd0["OBSTYPE"] == "OBJECT":
            c = SkyCoord(hd0["RA"], hd0["DEC"], unit="deg")
            ra = f"{c.ra.to_string(unit='hour',sep=':', precision=3, pad=True)}"
            dec = f"{c.dec.to_string(sep=':', precision=2, alwayssign=True, pad=True)}"
        else:
            ra = "00:00:00.000"
            dec = "00:00:00.00"

        airmass = f'{hd0["AIRMASS"]:.2f}' if "AIRMASS" in hd0 else "1.0"

        v = dict(FILENAME=filename, OBSTIME=obstime,
                 GROUP1=1, GROUP2=1,
                 OBJNAME=objname, OBJTYPE=objtype,
                 FRAMETYPE=frmtype, EXPTIME=exptime,
                 ROTPA=pa, RA=ra, DEC=dec,AM=airmass)

        lines.append(formats.format(**v))

    fn_out = datadir / f"IGRINS_DT_Log_{utdate}-1_H.txt"
    open(fn_out, "w").writelines(lines)


def main():
    parser = argparse.ArgumentParser(
        # formatter_class=argparse.RawTextHelpFormatter,
        description="Make a DT Log from the headers of ./indata/ut_date/SDCH* files.",
        epilog="examples:\n" +
        "  make_dtlog.py 20240425\n\n" +
        "version:\n  " + __version__)
    parser.add_argument("ut_date", help="UT Date of Data")

    parser.add_argument("--datadir", type=str, help="data directory. Default is ./indata/{ut_date}",
                        default="./indata/{ut_date}")

    args = parser.parse_args(args=None if sys.argv[1:] else ["--help"])

    datadir = args.datadir.format(ut_date=args.ut_date)

    write_dtlog(args.ut_date, Path(datadir))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3

import sys
import argparse
import os
import glob
from astropy.io import fits
from pathlib import Path


#Match H and K band flats taken with different flat lamps so that they all have matching obsnums
def match_flats(path_outdir):
    # all_K_files = sorted(path_outdir.glob("SDCK.fits"))
    # all_H_files = sorted(path_outdir.glob("SDCH*.fits"))
    all_K_files = sorted(path_outdir.glob("*K.fits"))
    all_H_files = sorted(path_outdir.glob("*H.fits"))
    n_raw_files = len(all_K_files)
    flat_K_files = [] #Lists to store raw flats to match
    flat_H_files = []
    for i in range(n_raw_files):
        k_header = fits.getheader(all_K_files[i])
        h_header = fits.getheader(all_H_files[i])
        if k_header['OBSTYPE'] == 'FLAT' and k_header['GCALFILT'] == 'ND3.0' and k_header['GCALLAMP'] == 'IRhigh' and k_header['GCALSHUT'] == 'OPEN':
            flat_K_files.append(all_K_files[i]) #Found K-band flats
        if (h_header['OBSTYPE'] == 'FLAT' and h_header['GCALFILT'] == 'ND2.0' and h_header['GCALLAMP'] == 'QH' and h_header['GCALSHUT'] == 'CLOSED') or \
                         (h_header['OBSTYPE'] == 'FLAT' and h_header['GCALFILT'] == 'CLEAR' and h_header['GCALLAMP'] == 'IRhigh' and h_header['GCALSHUT'] == 'OPEN'): ##Feb 3-11 2025 or Feb 12-Mar 15 2025
            flat_H_files.append(all_H_files[i]) #Found H-band flats
    n_k_flats = len(flat_K_files)
    n_h_flats = len(flat_H_files)
    if n_k_flats == n_h_flats:
        # lowest_obsnum_h = int(flat_H_files[0]._str[-9:-5])
        # lowest_obsnum_k = int(flat_K_files[0]._str[-9:-5])
        lowest_obsnum_h = int(flat_H_files[0]._str[-11:-7])
        lowest_obsnum_k = int(flat_K_files[0]._str[-11:-7])
        if lowest_obsnum_h < lowest_obsnum_k: #If H band flats have the lowest obsnum, move K band flats to match them
            for i in range(n_k_flats):
                #destination = Path(flat_H_files[i]._str.replace('SDCH', 'SDCK')) #Get filename from H band flat, but change name to K band
                destination = Path(flat_H_files[i]._str.replace('_H', '_K')) #Get filename from H band flat, but change name to K band
                flat_K_files[i].replace(destination) #Move file
                # file_to_delete = Path(flat_K_files[i]._str.replace('SDCK', 'SDCH')) #Delete extra unneeded flat files
                file_to_delete = Path(flat_K_files[i]._str.replace('_K', '_H')) #Delete extra unneeded flat files
                file_to_delete.unlink() #Delete file
        #else: #Else if K band flats have the lowest obsnum, move H band flats to match them
        elif lowest_obsnum_k < lowest_obsnum_h:
            for i in range(n_k_flats):
                #destination = Path(flat_K_files[i]._str.replace('SDCK', 'SDCH')) #Get filename from K band flat, but change name to H band
                destination = Path(flat_K_files[i]._str.replace('_K', '_H')) #Get filename from K band flat, but change name to H band
                flat_H_files[i].replace(destination) #Move file        
                # file_to_delete = Path(flat_H_files[i]._str.replace('SDCH', 'SDCK')) #Delete extra unneeded flat files
                file_to_delete = Path(flat_H_files[i]._str.replace('_H', '_K')) #Delete extra unneeded flat files
                file_to_delete.unlink() #Delete file
        else:
            print('It appears the H and K band FLAT ONs are not part of distinct sets of FLAT ONs for each band.  Will not modify FLAT ONs  Use the argument --disable-match-flats to turn flat matching off.')
    else:
        print('It appears the H and K band FLAT ONs are not part of distinct sets of FLAT ONs for each band.  Will not modify FLAT ONs  Use the argument --disable-match-flats to turn flat matching off.')


def unbundle(indir: Path, utdate: str, outdir: Path):
    fn_list = sorted(indir.glob(f"N{utdate}*.fits*"))

    if len(fn_list) == 0:
        print("no matching files are found")
        return

    outdir.mkdir(parents=True, exist_ok=True)

    for fi_ in fn_list:
        fi = fi_.name
        print(fi)
        # obsdate = fi[-18:-10]
        # obsid = fi[-9:-5]
        obsdate = fi[1:9]
        obsid = fi[10:14]

        hdu = fits.open(indir / fi)

        hd0 = hdu[0].header.copy()
        if hd0['OBSCLASS'] != 'acq':
            del hd0["EXTEND"]
            hd0.set("NAXIS1",0,after="NAXIS")
            hd0.set("NAXIS2",0,after="NAXIS1")

            band = ["H","K"]

            for i in range(2):
                hd = hd0.copy()
                hd.update(hdu[i+1].header[1:])
                im = hdu[i+1].data

                #filename = f"SDC{band[i]}_{obsdate}_{obsid}.fits"
                filename = f"N{obsdate}S{obsid}_{band[i]}.fits"
                hd["ORIGNAME"] = filename
                fits.PrimaryHDU(header=hd,data=im).writeto(
                    outdir / filename, overwrite=True)


def main():
    descriptions = """Given the ut_date, read the MEFs files of that date ('N{ut_date}*.fits')
and extract H & K spectra extensions into directory of ./indata/{ut_date} with names like
N{ut_date}*_H.fits, N{ut_date}*_K.fits.
"""
    parser = argparse.ArgumentParser(
        # formatter_class=argparse.RawTextHelpFormatter,
        description=descriptions,
        epilog="examples:\n" +
        "  mef_extract.py 20240425\n\n")

    parser.add_argument("ut_date", help="UT Date of Data")
    parser.add_argument("--mefdir", help="Directory containing MEF files.",
                        default=".")
    parser.add_argument("--outdir", type=str, help="Ouput data directory. Default is ./indata/{ut_date}",
                        default="./indata/{ut_date}")
    parser.add_argument("--disable-match-flats", help="Disable matching obsnums for seperately taken H and K band flat ON calibrations.",
                        action='store_true')
    # main(parser.parse_args(args=None if sys.argv[1:] else ["--help"]))

    args = parser.parse_args(args=None if sys.argv[1:] else ["--help"])

    outdir = args.outdir.format(ut_date=args.ut_date)

    unbundle(Path(args.mefdir), args.ut_date, Path(outdir))

    breakpoint()
    if not args.disable_match_flats:
        match_flats(Path(outdir))
    else:
        print('Argument --disable-match-flats set.  Do not try to match H and K band flat ONs or modify them.')


if __name__ == "__main__":
    main()

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11080095.svg)](https://doi.org/10.5281/zenodo.11080095)

# IGRINS Pipeline Package

The IGRINS PLP is designed to produce decent quality of processed
data of all the observing data without (or minimum) human interaction. It was also
designed to be adaptable for a real time processing during the
observing run.

The key concept of this pipeline is having a "recipe" to process a
certain data group. These approach is utilized by many other
observatories/instruments. In most cases, the recipe needs to be set
and recorded in the header (or log) when the observation is
executed. Unfortunately, this is not properly done. Therefore, to run
the pipeline, you should have an input file describing which recipe
should be used to which data sets.

IGRINS pipeline package is currently in active development. Version 3 is the latest version and is recommended for use for reducing data from IGRINS on the McDonald 2.7m, LDT/DCT, and Gemini-South telescopes.  

Version 2 will still work for those who need it.  Version 1, that was originally developed by Prof. Soojong Pak's team at Kyung Hee University (KHU), is deprecated and is not recommended to use.  Versions 2-3 were developed by the pipeline team at KASI (led by Dr. Jae-Joon Lee) in close collaboration with KHU's team.  Additional development and testing for v3 has been carried out by Kyle Kaplan and Erica Sawczynec at the University of Texas at Austin Department of Astronomy.

# IGRINS-2 Pipeline

The `igrins2` branch should be used for those who want to reduce IGRINS-2 data.

## Downloads

- https://github.com/igrins/plp/releases

## Documentation

- https://github.com/igrins/plp/wiki/

## Data Access and Archives

IGRINS data are made publicly available through the The Raw & Reduced IGRINS Spectral Archive (RRISA).

- RRISA: https://igrinscontact.github.io

IGRINS-2 data are made available through the Gemini Archive.

- Gemini Archive: https://archive.gemini.edu


## Publications

Many of the updates for v3 of the IGRINS PLP are described in the RRISA paper.
- https://ui.adsabs.harvard.edu/abs/2025PASP..137c4505S

The version 1 pipeline is described in the following publication.
- http://adsabs.harvard.edu/abs/2014AdSpR..53.1647S


[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.18579.svg)](http://dx.doi.org/10.5281/zenodo.18579)

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

IGRINS pipeline package is currently in active development. Version 3 is the latest version and is recommended for use for reducing data from the McDonald 2.7m, LDT/DCT, and Gemini-South telescopes.  Version 2 will still work for those who need it.  Version 1, that was originally developed by Prof. Soojong Pak's team at Kyung Hee University (KHU), is deprecated and is not recommended to use.  Versions 2-3 were developed by the pipeline team at KASI (led by Dr. Jae-Joon Lee) in close collaboration with KHU's team.  Additional development and testing for v3 has been carried out by Kyle Kaplan and Erica Sawczynec at the University of Texas at Austin Department of Astronomy.



## Downloads

- https://github.com/igrins/plp/releases

## Documentation

- https://github.com/igrins/plp/wiki/

## The Raw & Reduced IGRINS Spectral Archive (RRISA)
IGRINS data is made publically availiable through the The Raw & Reduced IGRINS Spectral Archive (RRISA).  The current version of RRISA (v1) uses the IGRINS PLP v2 for it's data reductions.  The raw data are also availiable for RRISA for those who want to perform their own data reduction with the IGRINGS PLP.

- RRISA v1: https://igrinscontact.github.io


## Publications

The version 1 pipeline is described in the following publication.
- http://adsabs.harvard.edu/abs/2014AdSpR..53.1647S


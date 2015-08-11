[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.18579.svg)](http://dx.doi.org/10.5281/zenodo.18579)

# IGRINS Pipeline Package

IGRINS pipeline package is currently in active development. Version 1, that was originally developed by Prof. Soojong Pak's team at Kyung Hee University (KHU), is deprecated and is not recommended to use. While the version 2 of the pipeline is still in active development, it is encouraged that you try this version.

- https://github.com/igrins/plp/releases

## Version 2

While the original pipeline by KHU's team works to some extent, there
are things that need to be fixed/improved/extended. A new version of
pipeline (version 2) is being actively developed by the pipeline team
at KASI (led by Dr. Jae-Joon Lee) in close collaboration with KHU's
team.

The new pipeline is designed to produce decent quality of processed
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

The page below briefly describes how to run the pipeline version 2.

 - https://github.com/igrins/plp/wiki/How-to-run-pipeline


## Version 1

This is the original pipeline that has developed by Prof. Soojong
Pak's team at Kyung Hee University (KHU). We recommend users to use the version 2 instead of this.

The version 1 pipeline is described in the following publication.

- http://adsabs.harvard.edu/abs/2014AdSpR..53.1647S

More information about this version can be found from the link below.

- http://irlab.khu.ac.kr/~igrins/



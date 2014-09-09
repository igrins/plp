# IGRINS Pipeline Package

IGRINS pipeline package is currently in active development. 

Currently there are two versions.

## Version 1

This is the original pipeline that has developed by Prof. Soojong
Pak's team at Kyung Hee University (KHU).

The pipeline is described in the following publication.

- http://adsabs.harvard.edu/abs/2014AdSpR..53.1647S

More information about this version can be found from the link below.

- http://irlab.khu.ac.kr/~igrins/


## Version 2

While the original pipeline by KHU's team works to some extent, there
are things that need to be fixed/improved/extended. A new version of
pipeline (version 2) is being actively developed by the pipeline team
at KASI (led by Dr. Jae-Joon Lee) in close collaboration with KHU's
team.

The new pipeline is designed to produce decent quality of processed
data of all the observing data without human interaction. It was also
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


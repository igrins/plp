from setuptools import setup, find_packages

setup(name='igrins',
      scripts=['igr_pipe.py',
               ],
      package_dir={'igrins': 'igrins'},
      packages=find_packages(),
      )

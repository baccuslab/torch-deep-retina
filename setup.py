from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import shutil
import atexit

import matplotlib

def install_mplstyle():
    stylefile = "deepretina.mplstyle"

    mpl_stylelib_dir = os.path.join(matplotlib.get_configdir() ,"stylelib")
    if not os.path.exists(mpl_stylelib_dir):
        os.makedirs(mpl_stylelib_dir)

    print("Installing style into", mpl_stylelib_dir)
    shutil.copy(
        os.path.join(os.path.dirname(__file__), stylefile),
        os.path.join(mpl_stylelib_dir, stylefile))

class PostInstallMoveFile(install):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        atexit.register(install_mplstyle)

setup(name='torchdeepretina',
      packages=find_packages(),
      version="0.1.0",
      description='Neural network models of the retina',
      author='Niru Maheshwaranathan, Lane McIntosh, Josh Melander, Julia Wang, Satchel Grant',
      author_email='grantsrb@stanford.edu',
      url='https://github.com/baccuslab/torch-deep-retina.git',
      install_requires=[i.strip() for i in open("requirements.txt").readlines()],
      long_description='''
          The torchdeepretina package contains methods for learning neural network models of the retina.
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X :: Ubuntu',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      )

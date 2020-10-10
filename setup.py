from setuptools import setup, find_packages
import torchdeepretina

setup(name='torchdeepretina',
      version=torchdeepretina.__version__,
      description='Neural network models of the retina',
      author='Niru Maheshwaranathan, Lane McIntosh, Josh Melander, Julia Wang, Satchel Grant, Xuehao Ding',
      author_email='xhding@stanford.edu',
      url='https://github.com/baccuslab/torch-deep-retina.git',
      install_requires=[i.strip() for i in open("requirements.txt").readlines()],
      long_description='''
          The torchdeepretina package contains methods for learning neural network models of the retina.
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X :: Ubuntu',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      packages=find_packages(),
      )

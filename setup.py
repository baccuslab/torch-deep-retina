from setuptools import setup, find_packages

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

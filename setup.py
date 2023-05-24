import setuptools
import sys

sys.path[0:0] = ['wfmoments']
from package_metadata import *

setuptools.setup(
    name=NAME,
    version=VERSION,
    description='',
    author=AUTHOR,
    author_email=EMAIL,
    packages=['wfmoments'],
    package_dir={'wfmoments': 'wfmoments'},
    install_requires=['numpy>=1.20.0',
                      'scipy>=1.7.3'],
)

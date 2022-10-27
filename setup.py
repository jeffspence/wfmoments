from wfmoments import VERSION
import setuptools


setuptools.setup(
    name='wfmoments',
    version=VERSION,
    description='',
    author='Jeffrey P. Spence',
    author_email='jspence@stanford.edu',
    packages=['wfmoments'],
    package_dir={'wfmoments': 'wfmoments'},
    install_requires=['numpy>=1.20.0',
                      'scipy>=1.7.3'],
)

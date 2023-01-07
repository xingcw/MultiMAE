import setuptools

__version__ = '0.0.1'

pkgs = setuptools.find_packages()

setuptools.setup(
    name='multimae',
    version=__version__,
    packages=pkgs
)
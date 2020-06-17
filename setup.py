import runpy
from setuptools import setup, find_packages

__version__ = runpy.run_path("batchlib/__version__.py")["__version__"]

setup(
    name="batchlib",
    packages=find_packages(exclude=["test"]),
    version=__version__,
    url="https://github.com/hci-unihd/batchlib.git",
    author="Constantin Pape, Roman Remme, Adrian Wolny, Steffen Wolf, Lorenzo Cerrone"
)

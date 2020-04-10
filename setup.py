from setuptools import setup, find_packages

# TODO add all the authors
# minimal setup script for the mmpb package
setup(
    name="batchlib",
    packages=find_packages(exclude=["test"]),
    version='0.1.0',
    url="https://github.com/hci-unihd/batchlib.git"
    # author="Constantin Pape",
)

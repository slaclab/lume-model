from setuptools import setup, find_packages
from os import path, environ
import versioneer

cur_dir = path.abspath(path.dirname(__file__))

# parse requirements
with open(path.join(cur_dir, "requirements.txt"), "r") as f:
    requirements = f.read().split()

# set up additional dev requirements
dev_requirements = []
with open(path.join(cur_dir, "dev-requirements.txt"), "r") as f:
    dev_requirements = f.read().split()


setup(
    name="lume-model",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Jacqueline Garrahan",
    author_email="jgarra@slac.stanford.edu",
    license="Apache License, Version 2.0",
    packages=find_packages(),
    install_requires=requirements,
    # set up development requirements
    extras_require={"dev": dev_requirements, "test": ["pytest"]},
    url="https://github.com/slaclab/lume-model",
    include_package_data=True,
    python_requires=">=3.7",
)

from setuptools import setup, find_packages, SetuptoolsDeprecationWarning
import pkg_resources
import warnings
warnings.filterwarnings('ignore', category=SetuptoolsDeprecationWarning)

with open("README.md", "r") as fh:
    long_description = fh.read()

from __init__ import __version__
setup(
    name = "seed",
    version = __version__,
    author = "Magolor",
    author_email = "magolorcz@gmail.com",
    description = "SEED",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url = "https://github.com/Magolor/",
    project_urls={
        "Author":"https://github.com/Magolor/",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    package_dir={"":"src"},
    packages=find_packages(where="src"),
    package_data={
        'seed': ['resources/**/*.json'],
    },
    python_requires=">=3.9",
)

from pathlib import Path
from setuptools import find_packages, setup

from src.amr_mbart import __version__


extras = {"style": ["flake8", "isort", "black", "pygments"]}

setup(
    name="amr_mbart",
    version=__version__,
    description="",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    keywords="nlp amr parsing semantic-parsing multilingual bart mbart",
    package_dir={"": "src"},
    packages=find_packages("src"),
    url="https://github.com/BramVanroy/multilingual-text-to-amr",
    author="Bram Vanroy",
    author_email="bramvanroy@hotmail.com",
    license="cc-by-nc-sa-4.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Text Processing",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    project_urls={
        "Bug Reports": "https://github.com/BramVanroy/multilingual-text-to-amr/issues",
        "Source": "https://github.com/BramVanroy/multilingual-text-to-amr",
    },
    python_requires=">=3.8",
    install_requires=[
        "transformers",
        "evaluate",
        "scikit-learn",
        "datasets",
        "penman>=1.2.2"
        "dataclasses;python_version<'3.7'"
    ],
    extras_require=extras,
)

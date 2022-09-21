from pathlib import Path
from setuptools import find_packages, setup

from src.amr_bart import __version__


extras = {"style": ["flake8", "isort", "black"]}

setup(
    name="amr_bart",
    version=__version__,
    description="",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    keywords="nlp amr parsing semantic-parsing bart",
    package_dir={"": "src"},
    packages=find_packages("src"),
    url="https://github.com/BramVanroy/multilingual-text-to-amr",
    author="Bram Vanroy",
    author_email="bramvanroy@hotmail.com",
    license="GPLv3",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Text Processing",
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
        "penman>=1.2.2",
        "networkx",
        "tqdm",
        "torch"
    ],
    extras_require=extras,
)

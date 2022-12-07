from pathlib import Path
from setuptools import find_packages, setup

from src.mbart_amr import __version__


extras = {"style": ["flake8", "isort", "black"]}

setup(
    name="mbart_amr",
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
        "evaluate",
        "ftfy",
        "networkx",
        "penman>=1.2.2",
        "sacrebleu",
        "scikit-learn",
        "sentencepiece",
        "smatch",
        "torch",
        "tqdm",
        "transformers",
    ],
    extras_require=extras,
    entry_points={
        "console_scripts": [
            "run-mbart-amr = mbart_amr.run_mbart_amr:main",
        ],
    }
)

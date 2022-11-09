# Format source code automatically
style:
	black --line-length 119 --target-version py38 src/mbart_amr
	isort src/mbart_amr

# Control quality
quality:
	black --check --line-length 119 --target-version py38 src/mbart_amr
	isort --check-only src/mbart_amr
	flake8 src/mbart_amr --exclude __pycache__,__init__.py

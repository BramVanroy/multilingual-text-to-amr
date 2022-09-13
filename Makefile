# Format source code automatically
style:
	black --line-length 119 --target-version py37 src/amr_mbart examples
	isort src/amr_mbart examples

# Control quality
quality:
	black --check --line-length 119 --target-version py37 src/amr_mbart examples
	isort --check-only src/amr_mbart examples
	flake8 src/amr_mbart examples --exclude __pycache__,__init__.py

# Run tests
test:
	pytest tests

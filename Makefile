# Format source code automatically
style:
	black --line-length 119 --target-version py37 src/amr_bartt
	isort src/amr_mbart

# Control quality
quality:
	black --check --line-length 119 --target-version py37 src/amr_bartt
	isort --check-only src/amr_bartt
	flake8 src/amr_mbart --exclude __pycache__,__init__.py

# Run tests
test:
	pytest tests

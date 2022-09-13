# Format source code automatically
style:
	black --line-length 119 --target-version py37 src/amr_bart
	isort src/amr_bart

# Control quality
quality:
	black --check --line-length 119 --target-version py37 src/amr_bart
	isort --check-only src/amr_bart
	flake8 src/amr_bart --exclude __pycache__,__init__.py

# Run tests
test:
	pytest tests

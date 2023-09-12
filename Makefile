style:
	black --line-length 119 --target-version py311 src/multi_amr tests/
	isort src/multi_amr

quality:
	black --check --line-length 119 --target-version py311 src/multi_amr
	isort --check-only src/multi_amr
	flake8 src/multi_amr --exclude __pycache__,__init__.py

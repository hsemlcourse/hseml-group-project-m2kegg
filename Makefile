.PHONY: lint test run

lint:
	flake8 src/ tests/

test:
	pytest tests/

run:
	python src/modeling.py

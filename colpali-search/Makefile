.PHONY: check clean install runner
.DEFAULT_GOAL:=runner

install: pyproject.toml
	poetry install

clean:
	rm -rf `find . -type d -name __pycache__`
	rm -rf .ruff_cache .pytest_cache

check:
	poetry run ruff check searchagent/

runner: install check clean
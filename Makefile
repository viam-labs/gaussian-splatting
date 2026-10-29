MODULE_FILES := meta.json README.md run.sh setup.sh pyproject.toml uv.lock .python-version src

setup:
	uv sync --frozen --extra build

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest -q

integration:
	uv run pytest -q -m integration

module.tar.gz: $(MODULE_FILES)
	tar czf $@ $(MODULE_FILES)

module: lint typecheck test module.tar.gz

clean:
	rm -f module.tar.gz

.PHONY: setup lint format typecheck test integration module clean

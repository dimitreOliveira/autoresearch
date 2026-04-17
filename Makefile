build:
	uv sync --frozen

lint:
	uv run codespell
	uv run ruff check --select I --fix
	uv run ruff check --fix
	uv run ruff format
	uv run ty check --ignore unresolved-import
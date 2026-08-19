.PHONY: install run debug clean lint lint-strict build


run:
	@uv run python3 -m src

debug:
	@uv run python3 -m pdb -m src

clean:
	@rm -rf data/output 
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	
install:
	@uv sync

lint:
	@uv run mypy .
	@uv run flake8 src
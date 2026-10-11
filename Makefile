.PHONY: install lint format test build science medical chat-data clean-data mix compare train serve data

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests

test:
	pytest -q

build:
	docker compose --profile jobs build
	docker compose build serve

science:
	docker compose run --rm science

medical:
	docker compose run --rm medical

chat-data:
	docker compose run --rm chat-data

clean-data:
	docker compose run --rm clean

mix:
	docker compose run --rm mix

compare:
	docker compose run --rm compare

train:
	docker compose run --rm train

serve:
	docker compose up serve

data: science medical chat-data clean-data mix

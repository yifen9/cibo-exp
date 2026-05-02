set shell := ["bash", "-lc"]

default:
    just --list

init:
    just venv && \
    just sync

venv:
    test -d .venv || uv venv

sync:
    uv sync --all-packages

sync-lock:
    uv sync --locked --all-packages

up:
    uv lock --upgrade

add PKG:
    uv add {{PKG}}

add-dev PKG:
    uv add --dev {{PKG}}

rm PKG:
    uv remove {{PKG}}

rm-dev PKG:
    uv remove --dev {{PKG}}

fmt:
    just sync && \
    uv run ruff format .

fmt-check:
    uv run ruff format --check .

lint:
    uv run ruff check . --fix

lint-check:
    uv run ruff check .

test:
    uv run pytest

ci:
    just venv && \
    just sync-lock && \
    just fmt-check && \
    just lint-check # && \
    # just test

docs-build:
	uv run pdoc --math -o site cibo_exp

docs-serve:
    uv run pdoc --math -p 8080 cibo_exp

s-t-exp-01:
    uv run python scripts/tests/exp_01.py

s-t-exp-02:
    uv run python scripts/tests/exp_02.py

s-t-exp-03:
    uv run python scripts/tests/exp_03.py

s-t-exp-04:
    uv run python scripts/tests/exp_04.py

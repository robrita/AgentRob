PY=python
UV=uv

.PHONY: help setup sync run lint test

help:
	@echo "Targets: setup, sync, run, lint"

setup:
	$(UV) sync

sync:
	$(UV) sync

run:
	$(UV) run uvicorn app.main:app --host 0.0.0.0 --port 8000
# 	$(UV) run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

lint:
	$(UV) run ruff check .

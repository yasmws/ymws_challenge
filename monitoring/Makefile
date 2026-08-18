HOST := 0.0.0.0
PORT := 8001
PYTHON ?= conda run --no-capture-output -n trillia python

.PHONY: help run-server run-notebook install

help:
	@echo "Available targets:"
	@echo "  make run-server   Start the FastAPI server with Uvicorn"
	@echo "  make run-notebook  Open the notebook used to test the API"
	@echo "  make install       Install the project dependencies"

run-server:
	$(PYTHON) -m uvicorn app.main:app --host $(HOST) --port $(PORT) --reload

run-notebook:
	$(PYTHON) -m jupyter notebook caller.ipynb

install:
	pip install -r requirements.txt && pip install -r app/requirements.txt

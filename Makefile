PYTHON ?= python3
VENV ?= .venv
PYTHON_BIN := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
INSTALL_STAMP := $(VENV)/.installed

.PHONY: all re clean fclean run_analyzer test lint

all: $(INSTALL_STAMP)

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)

$(INSTALL_STAMP): $(VENV)/bin/python pyproject.toml
	$(PYTHON_BIN) -m pip install --upgrade pip
	$(PYTHON_BIN) -m pip install -e .
	touch $(INSTALL_STAMP)

run_analyzer: $(INSTALL_STAMP)
	$(PYTHON_BIN) -m my_torch_analyzer

test: $(INSTALL_STAMP)
	$(PYTHON_BIN) -m pytest tests || (status=$$?; if [ $$status -eq 5 ]; then echo "No tests collected."; else exit $$status; fi)

lint: $(INSTALL_STAMP)
	$(PYTHON_BIN) -m compileall my_torch my_torch_analyzer

clean:
	rm -rf build dist .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.egg-info" -type d -prune -exec rm -rf {} +
	rm -f $(INSTALL_STAMP)

fclean: clean
	rm -rf $(VENV)

re: fclean all

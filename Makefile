PYTHON ?= python3
VENV ?= .venv
PYTHON_BIN := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
INSTALL_STAMP := $(VENV)/.installed
RUFF := $(PYTHON_BIN) -m ruff

.PHONY: all re clean fclean run_analyzer test lint

all: $(INSTALL_STAMP)

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV) || $(PYTHON) -m venv --without-pip $(VENV)
	$(VENV)/bin/python -m pip --version >/dev/null 2>&1 || { \
		echo "Pip not found, installing..."; \
		curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py; \
		$(VENV)/bin/python get-pip.py; \
		rm get-pip.py; \
	}

$(INSTALL_STAMP): $(VENV)/bin/python pyproject.toml
	$(PYTHON_BIN) -m pip install --upgrade pip
	$(PYTHON_BIN) -m pip install -e .[dev]
	touch $(INSTALL_STAMP)

run_analyzer: $(INSTALL_STAMP)
	$(PYTHON_BIN) -m my_torch_analyzer

test: $(INSTALL_STAMP)
	$(PYTHON_BIN) -m pytest tests

lint: $(INSTALL_STAMP)
	$(PIP) show ruff >/dev/null 2>&1 || $(PIP) install ruff
	$(PYTHON_BIN) -m compileall my_torch my_torch_analyzer
	$(RUFF) check my_torch my_torch_analyzer

clean:
	rm -rf build dist .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.egg-info" -type d -prune -exec rm -rf {} +
	rm -f $(INSTALL_STAMP)

fclean: clean
	rm -rf $(VENV)

re: fclean all

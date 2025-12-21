PYTHON ?= python3
BIN := my_torch_analyzer

.PHONY: all re clean fclean run_analyzer test lint compile

all: $(BIN)

$(BIN): Makefile
	@printf '%s\n' \
		'#!/usr/bin/env python3' \
		'from __future__ import annotations' \
		'' \
		'import os' \
		'import runpy' \
		'import sys' \
		'' \
		'ROOT = os.path.dirname(os.path.abspath(__file__))' \
		'if ROOT not in sys.path:' \
		'    sys.path.insert(0, ROOT)' \
		'' \
		'runpy.run_module("my_torch_analyzer_pkg", run_name="__main__")' \
		> $(BIN)
	@chmod +x $(BIN)

compile:
	$(PYTHON) -m compileall my_torch my_torch_analyzer_pkg

run_analyzer: $(BIN)
	./$(BIN)

test:
	$(PYTHON) -m pytest tests

lint:
	$(PYTHON) -m compileall my_torch my_torch_analyzer_pkg

clean:
	rm -rf build dist .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.egg-info" -type d -prune -exec rm -rf {} +

fclean: clean
	rm -f $(BIN)

re: fclean all

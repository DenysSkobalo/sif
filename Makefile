PYTHON := python3
QUERY ?= apple-logo.jpg

QUERIES_DIR := data/queries
LOG_DIR := logs

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make run QUERY=<image>    - run single query"
	@echo "  make clean                - clean + logs + cache"
	@echo "  make rerun QUERY=<image>  - clean + run single query"
	@echo "  make list                 - list of quires"

.PHONY: run
run:
	$(PYTHON) -B main.py --query $(QUERY)

.PHONY: clean
clean:
	@echo "Cleaning cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Removing logs..."
	rm -rf $(LOG_DIR)


.PHONY: rerun
rerun: clean
	$(PYTHON) -B main.py --query $(QUERY)

.PHONY: list
list: 
	ls -all data/queries

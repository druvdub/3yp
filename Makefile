#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = 3yp
PYTHON_VERSION = 3.12.7
PYTHON_INTERPRETER = python
VENV_DIR = .venv

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	@echo ">>> Installing Python Dependencies"
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.egg-info" -delete
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 3yp
	isort --check --diff --profile black 3yp
	black --check --config pyproject.toml 3yp

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml 3yp




## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	@if [ -d "$(VENV_DIR)" ]; then \
		echo ">>> Deleting existing virtual environment"; \
		rm -rf $(VENV_DIR); \
	fi
	@if command -v virtualenv > /dev/null; then \
		echo ">>> Creating virtual environment using virtualenv"; \
		virtualenv -p $(PYTHON_INTERPRETER) $(VENV_DIR); \
	else \
		echo ">>> virtualenv is not installed. Falling back to venv"; \
		$(PYTHON_INTERPRETER) -m venv $(VENV_DIR); \
	fi
	@echo ""
	@echo ">>> New virtual environment created. Activate with:"
	@echo ">>>  source $(VENV_DIR)/bin/activate (MacOS/Linux)"
	@echo ">>>  source $(VENV_DIR)\Scripts\activate (Windows)"
	
# TODO: Make makefile compatible with Windows native shell



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) 3yp/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

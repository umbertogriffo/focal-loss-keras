.PHONY: check clean setup-env export-env test

all: check setup-env test

check:
	which pip3
	which python3

clean:
	rm -rf .pyenv/
	rm -rf .pytest_cache/

setup-env:
	virtualenv .pyenv; \
	. .pyenv/bin/activate; \
	pip3 install -r requirements.txt; \

export-env:
	. .pyenv/bin/activate; \
	pip3 freeze > requirements.txt

test:
	. .pyenv/bin/activate && cd src/tblx_pdm/tests; \
	pytest -s -vv;

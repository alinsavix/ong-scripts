PYINSTALLER_BIN ?= uv run --extra build pyinstaller
SPEC ?= obsstats.spec

.PHONY: install
install:
	pip install -e .

.PHONY: install-build
install-build:
	pip install -e .[build]

.PHONY: dist
dist:
	$(PYINSTALLER_BIN) $(SPEC)

.PHONY: clean
clean:
	rm -rf dist build

# helpful for debugging
print-%: ;@echo $*=$($*)

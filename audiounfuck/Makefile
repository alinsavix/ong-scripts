PYINSTALLER_BIN ?= pyinstaller
SPEC ?= audiounfuck.spec

.PHONY: dist
dist:
	$(PYINSTALLER_BIN) $(SPEC)

.PHONY: clean
clean:
	rm -rf dist build

# helpful for debugging
print-%: ;@echo $*=$($*)

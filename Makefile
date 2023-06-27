.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  build
	@echo "  clean

.PHONY: build
build:
	cmake -B build -G Ninja
	cmake --build build

.PHONY : clean
clean:
	find . -type d -name '__pycache__' -exec rm -r {} \+
	rm -rf dist
	rm -rf build
	rm -rf BuildTests BuildTidy BuildGBench
	rm -rf .coverage coverage_html_report/
	rm -rf tmp
	rm -rf *.dat
	rm -rf pennylane_lightning/lightning_qubit_ops*

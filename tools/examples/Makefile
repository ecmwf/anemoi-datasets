YAML := $(wildcard *.yaml)

TARGETS := $(YAML:.yaml=.zarr)

all: $(TARGETS)

%.zarr: %.yaml
	anemoi-datasets create $< $@ --overwrite


clean::
	rm -fr *.zarr *.sync *.statistics

.SUFFIXES: .zarr .yaml

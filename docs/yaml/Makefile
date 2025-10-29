YAML := $(wildcard building*.yaml)

TARGETS := $(YAML:.yaml=.txt)

all: $(TARGETS)

%.zarr: %.yaml
	anemoi-datasets create $< $@ --overwrite

%.txt: %.zarr
	ln -sf $< dataset.zarr
	anemoi-datasets inspect dataset.zarr > $@
	rm -f dataset.zarr


clean::
	rm -fr *.zarr

.SUFFIXES: .zarr .yaml .txt
.PRECIOUS: %.zarr

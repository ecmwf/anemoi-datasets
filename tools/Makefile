TARGETS=dataset1.zarr dataset2.zarr

all: $(TARGETS)

%.zarr: %.yaml
	anemoi-datasets create $< $@ --overwrite


.SUFFIXES: .zarr .yaml

TARGETS=building1.txt

all: $(TARGETS)

%.zarr: %.yaml
	anemoi-datasets create $< $@ --overwrite

%.txt: %.zarr
	ln -sf $< dataset.zarr
	anemoi-datasets inspect dataset.zarr > $@
	rm -f dataset.zarr



.SUFFIXES: .zarr .yaml .txt

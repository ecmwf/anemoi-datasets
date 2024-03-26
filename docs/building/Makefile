TARGETS=building1.txt

all: $(TARGETS)

%.zarr: %.yaml
	anemoi-datasets create $< $@ --overwrite

%.txt: %.zarr
	anemoi-datasets inspect $< > $@ 



.SUFFIXES: .zarr .yaml .txt

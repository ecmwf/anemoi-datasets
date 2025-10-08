#!/bin/bash
set -eux
NAME=${1:-join}


# This script is not used in the tests
# it is only a tool to develop

anemoi-datasets init $NAME.yaml $NAME.zarr --overwrite
anemoi-datasets load $NAME.zarr --part 1/2
anemoi-datasets load $NAME.zarr --part 2/2

anemoi-datasets finalise $NAME.zarr

anemoi-datasets patch $NAME.zarr

anemoi-datasets init-additions $NAME.zarr --delta 12h
anemoi-datasets load-additions $NAME.zarr --part 1/2 --delta 12h
anemoi-datasets load-additions $NAME.zarr --part 2/2 --delta 12h
anemoi-datasets finalise-additions $NAME.zarr --delta 12h

anemoi-datasets cleanup $NAME.zarr

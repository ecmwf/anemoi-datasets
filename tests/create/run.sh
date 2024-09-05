#!/bin/bash
set -eux
NAME=join

anemoi-datasets create-step init $NAME.yaml $NAME.zarr --overwrite
anemoi-datasets create-step load $NAME.zarr --part 1/2
anemoi-datasets create-step load $NAME.zarr --part 2/2

anemoi-datasets create-step statistics $NAME.zarr
anemoi-datasets create-step size $NAME.zarr
# anemoi-datasets create-step finalise $NAME.zarr

anemoi-datasets create-step patch $NAME.zarr

anemoi-datasets create-step init-additions $NAME.zarr
anemoi-datasets create-step run-additions $NAME.zarr --part 1/2
anemoi-datasets create-step run-additions $NAME.zarr --part 2/2
anemoi-datasets create-step finalise-additions $NAME.zarr
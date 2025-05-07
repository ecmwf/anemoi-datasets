# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

## [0.5.23](https://github.com/ecmwf/anemoi-datasets/compare/0.5.22...0.5.23) (2025-05-07)


### Features

* support accumulations that resets regularly (e.g. every 24h) ([#314](https://github.com/ecmwf/anemoi-datasets/issues/314)) ([0cd772b](https://github.com/ecmwf/anemoi-datasets/commit/0cd772b168ea6a3eb2113a9b3de0ddd9e964a6de))

## [0.5.22](https://github.com/ecmwf/anemoi-datasets/compare/0.5.21...0.5.22) (2025-05-05)


### Features

* add command line tool to check naming conventions ([#287](https://github.com/ecmwf/anemoi-datasets/issues/287)) ([38a81e2](https://github.com/ecmwf/anemoi-datasets/commit/38a81e2256e9eb043594698b1adb3bf5bd9a68ed))
* copy datasets from http(s) sources ([#307](https://github.com/ecmwf/anemoi-datasets/issues/307)) ([cedd9db](https://github.com/ecmwf/anemoi-datasets/commit/cedd9dbfca0ee2add889d86a3f37990d508b2af3))
* downloading zip files if needed ([#288](https://github.com/ecmwf/anemoi-datasets/issues/288)) ([3024099](https://github.com/ecmwf/anemoi-datasets/commit/30240999a380b6d9087e5203d356f6770361ffbb))
* optionally search for zarr files at various location if not found ([#281](https://github.com/ecmwf/anemoi-datasets/issues/281)) ([8bd174c](https://github.com/ecmwf/anemoi-datasets/commit/8bd174ca4457ee57fc111df91e02f6b515d84c0c))
* reset accumulations ([#300](https://github.com/ecmwf/anemoi-datasets/issues/300)) ([b12666a](https://github.com/ecmwf/anemoi-datasets/commit/b12666aab59af389b591d098e3a3b1ee9b9a3bcb))
* work on check ([#311](https://github.com/ecmwf/anemoi-datasets/issues/311)) ([c6a8db9](https://github.com/ecmwf/anemoi-datasets/commit/c6a8db9ccecf821cfce216907c83a7f60dd21c0d))


### Bug Fixes

* bug in complement ([#296](https://github.com/ecmwf/anemoi-datasets/issues/296)) ([4e57f7c](https://github.com/ecmwf/anemoi-datasets/commit/4e57f7c4c1a8eaee3300065248963e1382819b32))
* bugs affecting creation of datasets from xarray-zarr ([#299](https://github.com/ecmwf/anemoi-datasets/issues/299)) ([09fcb51](https://github.com/ecmwf/anemoi-datasets/commit/09fcb51e8d1934024441e60d1f01367c742feea8))
* fix copy ([#289](https://github.com/ecmwf/anemoi-datasets/issues/289)) ([a79759f](https://github.com/ecmwf/anemoi-datasets/commit/a79759fb7ca0d019f03768b56cc7e807fa21f3d4))
* Proper indent for parallel additions ([#303](https://github.com/ecmwf/anemoi-datasets/issues/303)) ([0643fb8](https://github.com/ecmwf/anemoi-datasets/commit/0643fb80b0b83b15c7a01f140788665dd0d2076a))


### Documentation

* update some docstrings ([#283](https://github.com/ecmwf/anemoi-datasets/issues/283)) ([d3fe801](https://github.com/ecmwf/anemoi-datasets/commit/d3fe801abac7da932d14f03a660e9469683f91a3))

## [0.5.21](https://github.com/ecmwf/anemoi-datasets/compare/0.5.20...0.5.21) (2025-04-08)


### Features

* more tests ([#277](https://github.com/ecmwf/anemoi-datasets/issues/277)) ([0ea7d46](https://github.com/ecmwf/anemoi-datasets/commit/0ea7d46a61aaa4381f8eb75b2623033bd404da5b))


### Bug Fixes

* grib-index command default ([#275](https://github.com/ecmwf/anemoi-datasets/issues/275)) ([04a37d8](https://github.com/ecmwf/anemoi-datasets/commit/04a37d8a046dc6b314e9f2376c69f1824cbfa43d))
* pin version of numcodecs ([#285](https://github.com/ecmwf/anemoi-datasets/issues/285)) ([d86f317](https://github.com/ecmwf/anemoi-datasets/commit/d86f317981a312438a19323d5d67e1b31f80cf23))
* update version of transform ([#284](https://github.com/ecmwf/anemoi-datasets/issues/284)) ([e72cdde](https://github.com/ecmwf/anemoi-datasets/commit/e72cddedd8160a27a9e63e2581f4fcc1a91cd8b6))


### Documentation

* some cleanup ([#276](https://github.com/ecmwf/anemoi-datasets/issues/276)) ([496b82a](https://github.com/ecmwf/anemoi-datasets/commit/496b82a5e10119c1fd27a58a85d5964fcad3c7bb))
* time interpolation example ([#274](https://github.com/ecmwf/anemoi-datasets/issues/274)) ([9d19bd6](https://github.com/ecmwf/anemoi-datasets/commit/9d19bd6bc3c8e0c25718a05d670357936ec99a8b))
* update documentation ([#271](https://github.com/ecmwf/anemoi-datasets/issues/271)) ([85ea386](https://github.com/ecmwf/anemoi-datasets/commit/85ea38690e7e8fab4dbac5c5b4d10eafdabde766))

## [0.5.20](https://github.com/ecmwf/anemoi-datasets/compare/0.5.19...0.5.20) (2025-03-31)


### Features

* interpolate nearest spatial ([#260](https://github.com/ecmwf/anemoi-datasets/issues/260)) ([e6c9af4](https://github.com/ecmwf/anemoi-datasets/commit/e6c9af48dca7292940a8b10f2804e9d456b6bccc))
* new data sources ([#258](https://github.com/ecmwf/anemoi-datasets/issues/258)) ([708c816](https://github.com/ecmwf/anemoi-datasets/commit/708c816b80ae30781f42442e0e9d8d70fed2371c))
* save opened anemoi_dataset ([#259](https://github.com/ecmwf/anemoi-datasets/issues/259)) ([4759dd9](https://github.com/ecmwf/anemoi-datasets/commit/4759dd9da67ece246e691eb6be4637ef6bc0b157))


### Bug Fixes

* Remove hardcoded indices and get pressure levels from keys directly ([#257](https://github.com/ecmwf/anemoi-datasets/issues/257)) ([777fbac](https://github.com/ecmwf/anemoi-datasets/commit/777fbac9fb09afd6e77ee13f8442ccbb3efb73f8))


### Documentation

* Add API Docs ([#255](https://github.com/ecmwf/anemoi-datasets/issues/255)) ([be13424](https://github.com/ecmwf/anemoi-datasets/commit/be1342400552f520b64f4f0fdb29d39ee51d81fe))

## [0.5.19](https://github.com/ecmwf/anemoi-datasets/compare/0.5.18...0.5.19) (2025-03-27)


### Features

* 24h accumulations for era5 ([#266](https://github.com/ecmwf/anemoi-datasets/issues/266)) ([33f7919](https://github.com/ecmwf/anemoi-datasets/commit/33f791961649de287f7900fb5c7b340ebc6fb32c))
* add environment variables in recipe ([#228](https://github.com/ecmwf/anemoi-datasets/issues/228)) ([6d898aa](https://github.com/ecmwf/anemoi-datasets/commit/6d898aa681b052bb3444614f643ab14eaf6fdc62))


### Bug Fixes

* broken references in recipes ([#269](https://github.com/ecmwf/anemoi-datasets/issues/269)) ([bbdf339](https://github.com/ecmwf/anemoi-datasets/commit/bbdf3393dfd07755fbc424079e0a9072d823362b))
* fix typo ([#268](https://github.com/ecmwf/anemoi-datasets/issues/268)) ([a12f58f](https://github.com/ecmwf/anemoi-datasets/commit/a12f58fb7e379b75baf8063a02f3938ff29303ba))
* swap meshgrid dimension ordering in xarray grid creation ([#249](https://github.com/ecmwf/anemoi-datasets/issues/249)) ([938f3c9](https://github.com/ecmwf/anemoi-datasets/commit/938f3c926e6d8083db5e00a515301f79bac5637d))


### Documentation

* add reference to anemoi contributing guidelines  ([#265](https://github.com/ecmwf/anemoi-datasets/issues/265)) ([7322e8a](https://github.com/ecmwf/anemoi-datasets/commit/7322e8a04b631e15c0d424f4cf73e6ee8b7bc199))

## [0.5.18](https://github.com/ecmwf/anemoi-datasets/compare/0.5.17...0.5.18) (2025-03-25)


### Features

* better error message ([#252](https://github.com/ecmwf/anemoi-datasets/issues/252)) ([e74cbe9](https://github.com/ecmwf/anemoi-datasets/commit/e74cbe9a6eac8a15c65c477f24106c97b4ce1b54))


### Bug Fixes

* modify execute function signature ([#253](https://github.com/ecmwf/anemoi-datasets/issues/253)) ([e4ad1a4](https://github.com/ecmwf/anemoi-datasets/commit/e4ad1a4244387853529d112996071f720d673b1a))

## [0.5.17](https://github.com/ecmwf/anemoi-datasets/compare/0.5.16...0.5.17) (2025-03-24)


### Features

* add coordinate standard_name for rotated pole grid ([#192](https://github.com/ecmwf/anemoi-datasets/issues/192)) ([5464347](https://github.com/ecmwf/anemoi-datasets/commit/5464347322b235e391e17c543d8031dd1b9a5ebf))
* better logging for group of dates ([#235](https://github.com/ecmwf/anemoi-datasets/issues/235)) ([d7fc616](https://github.com/ecmwf/anemoi-datasets/commit/d7fc61680e424f0dc87ab052ce972718f8e34379))
* fix to_index ([#225](https://github.com/ecmwf/anemoi-datasets/issues/225)) ([baefd70](https://github.com/ecmwf/anemoi-datasets/commit/baefd70150da1e4f08bdda483ebc7a268bf2abdf))
* plugin support ([#241](https://github.com/ecmwf/anemoi-datasets/issues/241)) ([376ef1c](https://github.com/ecmwf/anemoi-datasets/commit/376ef1c21a16dcee0d88abb82c96aef5ad63494d))
* support sub-hourly steps ([#188](https://github.com/ecmwf/anemoi-datasets/issues/188)) ([7f219e9](https://github.com/ecmwf/anemoi-datasets/commit/7f219e9e41befd732557e124e3f828fd2436c58d))
* update test breaking due to noaa external data change ([#204](https://github.com/ecmwf/anemoi-datasets/issues/204)) ([4b99ea6](https://github.com/ecmwf/anemoi-datasets/commit/4b99ea63ba30a8d6d5ddf5ac3ab01573f0e77802))


### Bug Fixes

* allow xarrays with single value dimensions ([#230](https://github.com/ecmwf/anemoi-datasets/issues/230)) ([ee9fbf8](https://github.com/ecmwf/anemoi-datasets/commit/ee9fbf89eadd9a00cbe6d176fe9a9530e05c9ea4))
* create forcings based on xarray-zarr template (again) ([#244](https://github.com/ecmwf/anemoi-datasets/issues/244)) ([84cb312](https://github.com/ecmwf/anemoi-datasets/commit/84cb3129663223ba5e21446a769b7b7490c36374))
* fix for bug [#237](https://github.com/ecmwf/anemoi-datasets/issues/237) ([#247](https://github.com/ecmwf/anemoi-datasets/issues/247)) ([de3cab8](https://github.com/ecmwf/anemoi-datasets/commit/de3cab83ac6b18606aeb94ae4ca14beb052c8c7b))
* Use set in computing statistics, faster and use less memory ([#209](https://github.com/ecmwf/anemoi-datasets/issues/209)) ([e93dbc1](https://github.com/ecmwf/anemoi-datasets/commit/e93dbc11759d2d8992b7e466eeade0883dd29f83))


### Documentation

* add animation ([#208](https://github.com/ecmwf/anemoi-datasets/issues/208)) ([2af2fd6](https://github.com/ecmwf/anemoi-datasets/commit/2af2fd6850279670214d9f2b2b83bddb18ebed45))
* Docathon 2025 ([#234](https://github.com/ecmwf/anemoi-datasets/issues/234)) ([fb68b95](https://github.com/ecmwf/anemoi-datasets/commit/fb68b959a666899d09d4c2cbc069b7f805df84c4))
* fix readthedocs ([#223](https://github.com/ecmwf/anemoi-datasets/issues/223)) ([ce1b44e](https://github.com/ecmwf/anemoi-datasets/commit/ce1b44e72742b6a80dcc03c0a47129eb4620ad04))
* update doc with eccc-fstd, cdsapi and regrid ([#201](https://github.com/ecmwf/anemoi-datasets/issues/201)) ([57a53fc](https://github.com/ecmwf/anemoi-datasets/commit/57a53fcfde19f00262dbc3418e2b6208c56f080e))
* update project name ([#246](https://github.com/ecmwf/anemoi-datasets/issues/246)) ([5ddd4d1](https://github.com/ecmwf/anemoi-datasets/commit/5ddd4d1a3d89f4b98ed2e9fe95944dfcf697e194))
* use new logo ([#211](https://github.com/ecmwf/anemoi-datasets/issues/211)) ([76ecf15](https://github.com/ecmwf/anemoi-datasets/commit/76ecf15bfcb6635e85845e5e51336a7991053f16))

## 0.5.16 (2025-02-04)

<!-- Release notes generated using configuration in .github/release.yml at main -->

## What's Changed
### Other Changes 🔗
* chore: synced file(s) with ecmwf-actions/reusable-workflows by @DeployDuck in https://github.com/ecmwf/anemoi-datasets/pull/186

## New Contributors
* @DeployDuck made their first contribution in https://github.com/ecmwf/anemoi-datasets/pull/186

**Full Changelog**: https://github.com/ecmwf/anemoi-datasets/compare/0.5.15...0.5.16

## [Unreleased](https://github.com/ecmwf/anemoi-datasets/compare/0.5.8...HEAD)

## Changed

- Fix metadata serialization handling of numpy.integer (#140)
- Fix negative variance for constant variables (#148)
- Fix cutout slicing of grid dimension (#145)
- Use cKDTree instead of KDTree
- Implement 'complement' feature
- Add ability to patch xarrays (#160)
- Add support of ECCC file formats (fstd)
- Add `use_cdsapi_dataset` option to mars and accumulations

### Added

- Call filters from anemoi-transform
- Make test optional when adls is not installed Pull request #110
- Add wz_to_w, orog_to_z, and sum filters (#149)

## [0.5.8](https://github.com/ecmwf/anemoi-datasets/compare/0.5.7...0.5.8) - 2024-10-26

### Changed

- Bugfix in `auto_adjust`
- Fixed precommit CI errors
- Improve tests
- More verbosity

### Added

- Add anemoi-transform link to documentation
- Various bug fixes
- Control compatibility check in xy/zip
- Add `merge` feature
- Add support for storing `supporting_arrays` in checkpoint files
- Allow naming of datasets components
- Contributors file (#105)

### Changed

- Upload with ssh (experimental)
- Remove upstream dependencies from downstream-ci workflow (temporary) (#83)
- ci: pin python versions to 3.9 ... 3.12 for checks (#93)
- Fix `__version__` import in init

## [0.5.7](https://github.com/ecmwf/anemoi-datasets/compare/0.5.6...0.5.7) - 2024-10-09

### Changed

- Add support to fill missing dates

## [Allow for unknown CF coordinates](https://github.com/ecmwf/anemoi-datasets/compare/0.5.5...0.5.6) - 2024-10-04

### Changed

- Add `variables_metadata` entry in the dataset metadata
- Update documentation

### Changed

- Add `variables_metadata` entry in the dataset metadata

### Changed

- Add `variables_metadata` entry in the dataset metadata

## [0.5.5](https://github.com/ecmwf/anemoi-datasets/compare/0.5.4...0.5.5) - 2024-10-04

### Changed

- Allow for unknown coordinates when parsing CF input

## [Add support for (semi-)constant fields](https://github.com/ecmwf/anemoi-datasets/compare/0.5.1...0.5.2) - 2024-10-03

### Changed

- Fix failing zarr lat/lon tests

## [Bug fixes](https://github.com/ecmwf/anemoi-datasets/compare/0.5.0...0.5.1) - 2024-10-01

### Added

- Adding the user recipe in the dataset PR #59.
- Add `repeated_dates` action in create.

### Changed

- Bug fix in create/rename

## [0.5.0 - Incremental builds and Rescaling](https://github.com/ecmwf/anemoi-datasets/compare/0.4.5...0.5.0) - 2024-09-25

### Added

- New `rescale` keyword in `open_dataset` to change units of variables #36
- Add support for constant fields when creating datasets
- Simplify imports

### Changed

- Added incremental building of datasets
- Add missing dependency for documentation building
- Fix failing test due to previous merge
- Bug fix when creating dataset from zarr
- Bug fix with area selection in cutout operation
- add paths-ignore to ci workflow
- call provenance less often

### Removed

- pytest for notebooks

## [0.4.5](https://github.com/ecmwf/anemoi-datasets/compare/0.4.4...0.4.5)

### Added

- New `interpolate_frequency` keyword in `open_dataset`
- CI workflow to update the changelog on release
- adds the reusable cd pypi workflow
- merge strategy for changelog in .gitattributes #25
- adds ci hpc config (#43)

### Changed

- update CI to reusable workflows for PRs and releases
- Support sub-hourly datasets.
- Change negative variance detection to make it less restrictive
- Fix cutout bug that left some global grid points in the lam part
- Fix bug in computing missing dates in cutout option

### Removed

## [0.4.4](https://github.com/ecmwf/anemoi-datasets/compare/0.4.0...0.4.4) Bug fixes

- Bug fix in accumulations() that did not work with webmars
- Bug fix when using hindcasts input (missing dates on leap years)

## [0.4.0](https://github.com/ecmwf/anemoi-datasets/compare/0.3.0...0.4.0) Minor Release

### Added

- earthkit-data replaces climetlab

### Removed

- climetlab

## [0.3.0](https://github.com/ecmwf/anemoi-datasets/compare/0.2.0...0.3.0) Minor Release

### Added

- hindcast source

### Changed

- updated documentation

## [0.2.0](https://github.com/ecmwf/anemoi-datasets/compare/0.1.0...0.2.0) Minor Release

### Added

- statistics tendencies

### Removed

- CubesFilter

## [0.1.0](https://github.com/ecmwf/anemoi-models/releases/tag/0.1.0) Initial Release

### Added

- Documentation
- Initial code release for anemoi-datasets: create datasets for data-driven weather-models
- open datasets
- combine datasets

## Git Diffs:

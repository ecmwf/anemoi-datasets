# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

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

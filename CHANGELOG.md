# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

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

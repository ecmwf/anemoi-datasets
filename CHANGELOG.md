# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!

## [0.5.1](https://github.com/ecmwf/anemoi-utils/compare/0.5.0...0.5.1) (2026-03-17)


### Features

* Fix humanise corner cases for dates with timezone ([#274](https://github.com/ecmwf/anemoi-utils/issues/274)) ([0a82016](https://github.com/ecmwf/anemoi-utils/commit/0a820165107881a2347705a4e4fba1b2cdb71bad))
* **mlflow auth:** Add method to get user info from access token ([#273](https://github.com/ecmwf/anemoi-utils/issues/273)) ([0e77c63](https://github.com/ecmwf/anemoi-utils/commit/0e77c638901a55608e271e0d3a38b80161721ee0))


### Bug Fixes

* Adjust due to deprecations and add typing ([#271](https://github.com/ecmwf/anemoi-utils/issues/271)) ([b52788e](https://github.com/ecmwf/anemoi-utils/commit/b52788eff7c642cfeea17de179fb981341bc3594))
* First-time alias lookup for lazy factories ([#278](https://github.com/ecmwf/anemoi-utils/issues/278)) ([682f4e8](https://github.com/ecmwf/anemoi-utils/commit/682f4e8c1684ff3a3482d1c529fc31c516b17920))

## [0.5.0](https://github.com/ecmwf/anemoi-utils/compare/0.4.43...0.5.0) (2026-02-09)


### ⚠ BREAKING CHANGES

* **utils:** store vcs info in metadata ([#268](https://github.com/ecmwf/anemoi-utils/issues/268))
* Update pyproject remove python 3.10 ([#259](https://github.com/ecmwf/anemoi-utils/issues/259))

### Features

* Update pyproject remove python 3.10 ([#259](https://github.com/ecmwf/anemoi-utils/issues/259)) ([93a1c61](https://github.com/ecmwf/anemoi-utils/commit/93a1c61ac0ce0d99f2138440b5d14152d8dca437))
* **utils:** Store vcs info in metadata ([#268](https://github.com/ecmwf/anemoi-utils/issues/268)) ([b28a62b](https://github.com/ecmwf/anemoi-utils/commit/b28a62b3a0da6382449362132cdc000efc39ce5d))


### Bug Fixes

* Add Filelock to cache ([#262](https://github.com/ecmwf/anemoi-utils/issues/262)) ([20664f0](https://github.com/ecmwf/anemoi-utils/commit/20664f0818985d7da6af934f32b62bf389990023))
* Add provenance to tests deps ([#265](https://github.com/ecmwf/anemoi-utils/issues/265)) ([942936b](https://github.com/ecmwf/anemoi-utils/commit/942936b035db3606c51c7099e5983ee510e8ec6d))

## [0.4.43](https://github.com/ecmwf/anemoi-utils/compare/0.4.42...0.4.43) (2026-01-21)


### Features

* Deprecate `ai-models.json` in favour of `anemoi.json` ([#247](https://github.com/ecmwf/anemoi-utils/issues/247)) ([b5b1437](https://github.com/ecmwf/anemoi-utils/commit/b5b14375831f68bf7877350470e2ea62084a25e9))


### Bug Fixes

* Review provenance information ([#250](https://github.com/ecmwf/anemoi-utils/issues/250)) ([0c23efe](https://github.com/ecmwf/anemoi-utils/commit/0c23efe1811bd8cebb30d03b44d0c863c35bf583))

## [0.4.42](https://github.com/ecmwf/anemoi-utils/compare/0.4.41...0.4.42) (2026-01-09)


### Features

* Bugfix ([#254](https://github.com/ecmwf/anemoi-utils/issues/254)) ([aa35fc6](https://github.com/ecmwf/anemoi-utils/commit/aa35fc636883c6f08e18ab17601fdea38529a295))

## [0.4.41](https://github.com/ecmwf/anemoi-utils/compare/0.4.40...0.4.41) (2026-01-08)


### Features

* Fix argument type for transfer tool ([#253](https://github.com/ecmwf/anemoi-utils/issues/253)) ([7d020d1](https://github.com/ecmwf/anemoi-utils/commit/7d020d1e6d9791f8a34394515e59a90b5c83ca17))


### Bug Fixes

* Further improve typing on registry ([#249](https://github.com/ecmwf/anemoi-utils/issues/249)) ([97c8874](https://github.com/ecmwf/anemoi-utils/commit/97c887421097e34fcdad7d661d8561efca841085))
* Save_metadata to support both multiple datasets with many arrays and old set up with no arrays ([#239](https://github.com/ecmwf/anemoi-utils/issues/239)) ([b22444f](https://github.com/ecmwf/anemoi-utils/commit/b22444f37a127aea950ae2777a880db1e5911820))

## [0.4.40](https://github.com/ecmwf/anemoi-utils/compare/0.4.39...0.4.40) (2025-12-02)


### Bug Fixes

* Don't skip tests marked as [@skip](https://github.com/skip)_if_offline when running CIs ([#244](https://github.com/ecmwf/anemoi-utils/issues/244)) ([9b89d33](https://github.com/ecmwf/anemoi-utils/commit/9b89d33fda792aaf81106f75bb233623becf3c2c))
* Missing bool check of resume in download ([#242](https://github.com/ecmwf/anemoi-utils/issues/242)) ([a3a7ed2](https://github.com/ecmwf/anemoi-utils/commit/a3a7ed2b7ed0a3dc384e2e4e55ec5a95fbe20974))

## [0.4.39](https://github.com/ecmwf/anemoi-utils/compare/0.4.38...0.4.39) (2025-11-17)


### Features

* Resetting of s3 options ([#233](https://github.com/ecmwf/anemoi-utils/issues/233)) ([6081072](https://github.com/ecmwf/anemoi-utils/commit/60810729204d94bd5c75ba84ca33f8f4259fba06))


### Bug Fixes

* Improve typing on registry methods ([#232](https://github.com/ecmwf/anemoi-utils/issues/232)) ([66e4ec6](https://github.com/ecmwf/anemoi-utils/commit/66e4ec6cba626c18232fbf9151ea4827cdc59a07))
* Reraise exceptions in CLI-mode if run in debugger ([#225](https://github.com/ecmwf/anemoi-utils/issues/225)) ([33c0f8e](https://github.com/ecmwf/anemoi-utils/commit/33c0f8e125d6acc88fa907727cdf97f5a7bce858))
* Update README to reflect project maturity status ([#234](https://github.com/ecmwf/anemoi-utils/issues/234)) ([ab769c4](https://github.com/ecmwf/anemoi-utils/commit/ab769c473f44e00b8a1091feb813d9ec364232f0))

## [0.4.38](https://github.com/ecmwf/anemoi-utils/compare/0.4.37...0.4.38) (2025-10-22)


### Features

* **testing:** Add download timeout ([#230](https://github.com/ecmwf/anemoi-utils/issues/230)) ([721d114](https://github.com/ecmwf/anemoi-utils/commit/721d114f2702985d9fbabf68f384f7ccedb7cfb3))
* **testing:** Sane test data download retries ([#227](https://github.com/ecmwf/anemoi-utils/issues/227)) ([1e08996](https://github.com/ecmwf/anemoi-utils/commit/1e089962b14f05f4aa56eb66a884bfe50ce60dcc))


### Bug Fixes

* Fix frequency_to_string outputing 108000s ([#216](https://github.com/ecmwf/anemoi-utils/issues/216)) ([5806a0c](https://github.com/ecmwf/anemoi-utils/commit/5806a0c996235fb3a19d34eeb25813280c5c989f))
* Support dicts of supporting_arrays ([#229](https://github.com/ecmwf/anemoi-utils/issues/229)) ([9badbad](https://github.com/ecmwf/anemoi-utils/commit/9badbad360609d254717e0d32c6171beb903eb21))

## [0.4.37](https://github.com/ecmwf/anemoi-utils/compare/0.4.36...0.4.37) (2025-09-30)


### Features

* **mlflow auth:** Support for multiple servers ([#217](https://github.com/ecmwf/anemoi-utils/issues/217)) ([8ccfb1a](https://github.com/ecmwf/anemoi-utils/commit/8ccfb1ab063cccfec5852c386580036286b097c6))


### Bug Fixes

* Update s3 chunk size to 10 MB ([#220](https://github.com/ecmwf/anemoi-utils/issues/220)) ([aa20fa8](https://github.com/ecmwf/anemoi-utils/commit/aa20fa8b0b572fb6fa510b2f28c2b8b8a2f76d7c))
* Use `yaml` and `json` flag in metadata get command ([#222](https://github.com/ecmwf/anemoi-utils/issues/222)) ([6af46c4](https://github.com/ecmwf/anemoi-utils/commit/6af46c4e715fc55aca374d2112976aa7d1bac589))

## [0.4.36](https://github.com/ecmwf/anemoi-utils/compare/0.4.35...0.4.36) (2025-09-22)


### Features

* Add aliases to registry ([#219](https://github.com/ecmwf/anemoi-utils/issues/219)) ([37267b5](https://github.com/ecmwf/anemoi-utils/commit/37267b548556a796a01b43abb908011eeec85454))
* Debug imports ([#182](https://github.com/ecmwf/anemoi-utils/issues/182)) ([1eaa615](https://github.com/ecmwf/anemoi-utils/commit/1eaa61540dc9ac3d5fe82f2c91b7fc98c8bb10af))
* NoAuth for AML mlflow Logging ([#200](https://github.com/ecmwf/anemoi-utils/issues/200)) ([732182e](https://github.com/ecmwf/anemoi-utils/commit/732182ea5d255ba69ea2ed0a23b307d6f64aaf84))
* Rich logging ([#209](https://github.com/ecmwf/anemoi-utils/issues/209)) ([3c762a5](https://github.com/ecmwf/anemoi-utils/commit/3c762a593ba2dc734becc54b92984d6dc62967ac))
* Speedup checkpoint editing - remove compression ([#218](https://github.com/ecmwf/anemoi-utils/issues/218)) ([b49120f](https://github.com/ecmwf/anemoi-utils/commit/b49120f763b0b6ee10c805bab2aa7b973047f963))
* Use obstore to access s3 buckets ([#210](https://github.com/ecmwf/anemoi-utils/issues/210)) ([da380be](https://github.com/ecmwf/anemoi-utils/commit/da380be71d78274d72bd0a3859ef00b1c80e9469))


### Bug Fixes

* Add missing s3 function used by datasets ([#212](https://github.com/ecmwf/anemoi-utils/issues/212)) ([30589e8](https://github.com/ecmwf/anemoi-utils/commit/30589e891fbdb1cff205f0350c63e93a725c7242))

## [0.4.35](https://github.com/ecmwf/anemoi-utils/compare/0.4.34...0.4.35) (2025-08-12)


### Bug Fixes

* Config override ([#204](https://github.com/ecmwf/anemoi-utils/issues/204)) ([cdeef1e](https://github.com/ecmwf/anemoi-utils/commit/cdeef1ef95ecd3696fefc751f8a97e90fe357329))

## [0.4.34](https://github.com/ecmwf/anemoi-utils/compare/0.4.33...0.4.34) (2025-08-11)


### Bug Fixes

* Typo ([#201](https://github.com/ecmwf/anemoi-utils/issues/201)) ([7c98725](https://github.com/ecmwf/anemoi-utils/commit/7c987258b8c3ccfc159175d6d8f5bf460f308499))

## [0.4.33](https://github.com/ecmwf/anemoi-utils/compare/0.4.32...0.4.33) (2025-08-07)


### Features

* **config:** Expand environment var recognition for the use of config override ([#197](https://github.com/ecmwf/anemoi-utils/issues/197)) ([9bd9170](https://github.com/ecmwf/anemoi-utils/commit/9bd9170478857cf129fa6b8042bc22d1a3940156))

## [0.4.32](https://github.com/ecmwf/anemoi-utils/compare/0.4.31...0.4.32) (2025-08-05)


### Features

* Improve types of testing ([#186](https://github.com/ecmwf/anemoi-utils/issues/186)) ([7bc7cbd](https://github.com/ecmwf/anemoi-utils/commit/7bc7cbdc1f3452e131b12067684f39dea067eed6))

## [0.4.31](https://github.com/ecmwf/anemoi-utils/compare/0.4.30...0.4.31) (2025-08-04)


### Bug Fixes

* Remove too many warnings ([#193](https://github.com/ecmwf/anemoi-utils/issues/193)) ([df6862b](https://github.com/ecmwf/anemoi-utils/commit/df6862bf829e67651ccc97cbaac9f38096ad4d34))

## [0.4.30](https://github.com/ecmwf/anemoi-utils/compare/0.4.29...0.4.30) (2025-07-31)


### Bug Fixes

* Refactor code for casting dotdicts and apply this in getitem and setitem methods ([#169](https://github.com/ecmwf/anemoi-utils/issues/169)) ([e91aecf](https://github.com/ecmwf/anemoi-utils/commit/e91aecf6699a0daaed6f79e92b4ebc57cd4abe36))

## [0.4.29](https://github.com/ecmwf/anemoi-utils/compare/0.4.28...0.4.29) (2025-07-22)


### Features

* Better support for negative timedeltas ([#180](https://github.com/ecmwf/anemoi-utils/issues/180)) ([3f8041a](https://github.com/ecmwf/anemoi-utils/commit/3f8041a46b525b6fcbe6171cd8a8a40ec30b2c1f))
* **deps:** Use mlflow-skinny instead of mlflow ([#184](https://github.com/ecmwf/anemoi-utils/issues/184)) ([82e5c30](https://github.com/ecmwf/anemoi-utils/commit/82e5c3053962cd8e1e8f6a1ea9e8f92492e497b4))
* Protect mlflow token file ([#183](https://github.com/ecmwf/anemoi-utils/issues/183)) ([fdf0fc8](https://github.com/ecmwf/anemoi-utils/commit/fdf0fc84ee3e8076928f6c888374cd3aa008023b))
* **sanitise:** Sanitation level ([#175](https://github.com/ecmwf/anemoi-utils/issues/175)) ([8d85d8f](https://github.com/ecmwf/anemoi-utils/commit/8d85d8fd889bf72b8066cc021d4d7b329a360848))
* Support negative timedelta ([#178](https://github.com/ecmwf/anemoi-utils/issues/178)) ([546f6ec](https://github.com/ecmwf/anemoi-utils/commit/546f6ec76534cd39094957ce3b57b34f14f7a000))


### Bug Fixes

* Clean utils ([#185](https://github.com/ecmwf/anemoi-utils/issues/185)) ([de3c7a4](https://github.com/ecmwf/anemoi-utils/commit/de3c7a47f14c258997942564717c480caa124ee6))

## [0.4.28](https://github.com/ecmwf/anemoi-utils/compare/0.4.27...0.4.28) (2025-07-03)


### Features

* Migrate mlflow utils from anemoi-training ([#174](https://github.com/ecmwf/anemoi-utils/issues/174)) ([0b7767b](https://github.com/ecmwf/anemoi-utils/commit/0b7767bc23486b140ad7423e3c5c7d5857cef71c))


### Bug Fixes

* Treat mlflow as an optional dependency ([#177](https://github.com/ecmwf/anemoi-utils/issues/177)) ([feb1088](https://github.com/ecmwf/anemoi-utils/commit/feb1088169a29f42032bf26d5c43f9817557bafc))

## [0.4.27](https://github.com/ecmwf/anemoi-utils/compare/0.4.26...0.4.27) (2025-06-27)


### Features

* Split s3 config from s3 client code ([#170](https://github.com/ecmwf/anemoi-utils/issues/170)) ([56dacb1](https://github.com/ecmwf/anemoi-utils/commit/56dacb19efa0979acd72edb72a95f058b69d612a))

## [0.4.26](https://github.com/ecmwf/anemoi-utils/compare/0.4.25...0.4.26) (2025-06-25)


### Features

* Fixtures for temp dir handling for test data ([#166](https://github.com/ecmwf/anemoi-utils/issues/166)) ([2b9677f](https://github.com/ecmwf/anemoi-utils/commit/2b9677fffc5eba84876f974001b87b73c7e542af))
* Move anemoi-inference metadata command to this package, add metadata removal options ([#167](https://github.com/ecmwf/anemoi-utils/issues/167)) ([cabb989](https://github.com/ecmwf/anemoi-utils/commit/cabb989bdd4154a0476acf48e1ac44099c91c6db))

## [0.4.25](https://github.com/ecmwf/anemoi-utils/compare/0.4.24...0.4.25) (2025-06-24)


### Features

* Add a CLI to transfer data ([#164](https://github.com/ecmwf/anemoi-utils/issues/164)) ([3a845ca](https://github.com/ecmwf/anemoi-utils/commit/3a845ca0c31d115e6b3d0496d862a3eaee5fb236))
* Add function to test cli ([#168](https://github.com/ecmwf/anemoi-utils/issues/168)) ([9ac9b06](https://github.com/ecmwf/anemoi-utils/commit/9ac9b06b8fd0a62cad33ea5de6a6b482f0a13656))

## [0.4.24](https://github.com/ecmwf/anemoi-utils/compare/0.4.23...0.4.24) (2025-06-06)


### Features

* Add s3.object_exists() function ([#157](https://github.com/ecmwf/anemoi-utils/issues/157)) ([d898811](https://github.com/ecmwf/anemoi-utils/commit/d8988116320265dc6dfe467c57e0b6f29f76a2c1))
* Allow wildcard in config for matching s3 buckets to end points ([#160](https://github.com/ecmwf/anemoi-utils/issues/160)) ([ab20da7](https://github.com/ecmwf/anemoi-utils/commit/ab20da7e9497435a7183705b02dcbb7317d2700b))

## [0.4.23](https://github.com/ecmwf/anemoi-utils/compare/0.4.22...0.4.23) (2025-05-20)


### Bug Fixes

* fix list_folder on s3 ([#154](https://github.com/ecmwf/anemoi-utils/issues/154)) ([3ceb42c](https://github.com/ecmwf/anemoi-utils/commit/3ceb42c5185290d4c12e3fe90c3c331e3d8c7a5f))
* Remove the requirment to have git installed ([#149](https://github.com/ecmwf/anemoi-utils/issues/149)) ([88846e8](https://github.com/ecmwf/anemoi-utils/commit/88846e80be2927050a879ff953a78aecf39c3ac5))
* Use urllib to make _offline() aware of HTTP(s) proxies. ([#150](https://github.com/ecmwf/anemoi-utils/issues/150)) ([5c4d06f](https://github.com/ecmwf/anemoi-utils/commit/5c4d06f931590cc360eb4ffeeb8753a5d3d72bcb))

## [0.4.22](https://github.com/ecmwf/anemoi-utils/compare/0.4.21...0.4.22) (2025-04-10)


### Bug Fixes

* do not write to existing dir ([#148](https://github.com/ecmwf/anemoi-utils/issues/148)) ([38c6db6](https://github.com/ecmwf/anemoi-utils/commit/38c6db62c113e093d11c49b0fc398587ee89946c))
* remove archive file after unpacking ([#145](https://github.com/ecmwf/anemoi-utils/issues/145)) ([790e2a3](https://github.com/ecmwf/anemoi-utils/commit/790e2a3370db3d5c275f95b920926d5a01f894a7))

## [0.4.21](https://github.com/ecmwf/anemoi-utils/compare/0.4.20...0.4.21) (2025-04-07)


### Features

* allow temporary settings ([#143](https://github.com/ecmwf/anemoi-utils/issues/143)) ([38cefb5](https://github.com/ecmwf/anemoi-utils/commit/38cefb5c4ebd4e496d2c332e1d1d8b86d551615c))


### Bug Fixes

* pydantic schemas ([#141](https://github.com/ecmwf/anemoi-utils/issues/141)) ([c30f804](https://github.com/ecmwf/anemoi-utils/commit/c30f804012a4200eee69f5fb0708d4af760cb5f7))

## [0.4.20](https://github.com/ecmwf/anemoi-utils/compare/0.4.19...0.4.20) (2025-04-04)


### Features

* better message in testing ([#138](https://github.com/ecmwf/anemoi-utils/issues/138)) ([44f1638](https://github.com/ecmwf/anemoi-utils/commit/44f1638d64439af1e66f37f5e01f0cb4a384e175))

## [0.4.19](https://github.com/ecmwf/anemoi-utils/compare/0.4.18...0.4.19) (2025-04-04)


### Features

* more testing support functions ([#136](https://github.com/ecmwf/anemoi-utils/issues/136)) ([5687b87](https://github.com/ecmwf/anemoi-utils/commit/5687b87ed17748412340d00f0724249f59b4e3f2))


### Documentation

* add api ([#133](https://github.com/ecmwf/anemoi-utils/issues/133)) ([16af518](https://github.com/ecmwf/anemoi-utils/commit/16af5184eafbfc29cc3f0217a35675f2aa32847e))

## [0.4.18](https://github.com/ecmwf/anemoi-utils/compare/0.4.17...0.4.18) (2025-03-31)


### Features

* add matching rules ([#132](https://github.com/ecmwf/anemoi-utils/issues/132)) ([2382980](https://github.com/ecmwf/anemoi-utils/commit/2382980f4f53909a73fa0a5c8cfab108625f3c55))

## [0.4.17](https://github.com/ecmwf/anemoi-utils/compare/0.4.16...0.4.17) (2025-03-27)


### Features

* add generic env variables to override anemoi user config ([#128](https://github.com/ecmwf/anemoi-utils/issues/128)) ([fdc7248](https://github.com/ecmwf/anemoi-utils/commit/fdc72485616a0c092356a9ffa4cdca838a0c1a9d))


### Bug Fixes

* Iterate over copy of sys.modules. ([#127](https://github.com/ecmwf/anemoi-utils/issues/127)) ([7b0e7d0](https://github.com/ecmwf/anemoi-utils/commit/7b0e7d08264f7eb4c92fdcd744ff8c46eac82fb7))
* plugin name on error ([#120](https://github.com/ecmwf/anemoi-utils/issues/120)) ([a747f63](https://github.com/ecmwf/anemoi-utils/commit/a747f63d74bf1b108d913694915df59ffc4640c1))


### Documentation

* add links to GitHub  ([#123](https://github.com/ecmwf/anemoi-utils/issues/123)) ([cfe1ea2](https://github.com/ecmwf/anemoi-utils/commit/cfe1ea281e03a56b9a02108b6787c6c05b9518b0))
* Docathon ([#121](https://github.com/ecmwf/anemoi-utils/issues/121)) ([e1c9292](https://github.com/ecmwf/anemoi-utils/commit/e1c9292d65b1ffc8c9ce8eed41c7ffbe81f865a3))
* fix comment ([#125](https://github.com/ecmwf/anemoi-utils/issues/125)) ([ad3ed12](https://github.com/ecmwf/anemoi-utils/commit/ad3ed126f9a507dde7ce19064f1d32dae2cee6a3))

## [0.4.16](https://github.com/ecmwf/anemoi-utils/compare/0.4.15...0.4.16) (2025-03-22)


### Bug Fixes

* support plugin errors ([#118](https://github.com/ecmwf/anemoi-utils/issues/118)) ([1f0bb30](https://github.com/ecmwf/anemoi-utils/commit/1f0bb30d4d9441e6883c060e35fe4410f0c91833))

## [0.4.15](https://github.com/ecmwf/anemoi-utils/compare/0.4.14...0.4.15) (2025-03-21)


### Features

* accept hyphens in factory names ([#116](https://github.com/ecmwf/anemoi-utils/issues/116)) ([ada96e9](https://github.com/ecmwf/anemoi-utils/commit/ada96e911b592ff9d95d3a93fff5a6aa21cdebbe))

## [0.4.14](https://github.com/ecmwf/anemoi-utils/compare/0.4.13...0.4.14) (2025-03-21)


### Bug Fixes

* plugin support ([#110](https://github.com/ecmwf/anemoi-utils/issues/110)) ([329395a](https://github.com/ecmwf/anemoi-utils/commit/329395a5870cbf59bacb39cb5afea6b91c465b07))

## [0.4.13](https://github.com/ecmwf/anemoi-utils/compare/0.4.12...0.4.13) (2025-03-14)


### Features

* add robust requests ([#112](https://github.com/ecmwf/anemoi-utils/issues/112)) ([5d87227](https://github.com/ecmwf/anemoi-utils/commit/5d87227e6f0b39f087f8a34f238806a2f73480f1))
* bugfix ([#100](https://github.com/ecmwf/anemoi-utils/issues/100)) ([c016cb4](https://github.com/ecmwf/anemoi-utils/commit/c016cb46c23b6a0575d9d843b06fd6b9f71b9f27))
* keep yaml formating in error messages ([#108](https://github.com/ecmwf/anemoi-utils/issues/108)) ([3bd6682](https://github.com/ecmwf/anemoi-utils/commit/3bd66828cf19d8e3d7d3fbed27533161b6285828))
* re-add default values in transfer function ([#101](https://github.com/ecmwf/anemoi-utils/issues/101)) ([6462205](https://github.com/ecmwf/anemoi-utils/commit/6462205ee25fa35a71af047b1fbb04bd3c4ca2c4))


### Bug Fixes

* add optional dependency. boto3 &lt;= 1.36 ([#105](https://github.com/ecmwf/anemoi-utils/issues/105)) ([c8c8393](https://github.com/ecmwf/anemoi-utils/commit/c8c8393ab1e886289541d3aa47a614afe5cd379b))


### Documentation

* update logo ([#96](https://github.com/ecmwf/anemoi-utils/issues/96)) ([c297127](https://github.com/ecmwf/anemoi-utils/commit/c297127e066c92023ca065b3e7d36ac4ab62527e))

## 0.4.12 (2025-01-30)

<!-- Release notes generated using configuration in .github/release.yml at main -->

## What's Changed
### Other Changes 🔗
* feat: better support for timedelta larger than 24h by @floriankrb in https://github.com/ecmwf/anemoi-utils/pull/81
* feat(requests): read input from stdin by @gmertes in https://github.com/ecmwf/anemoi-utils/pull/82
* chore: synced file(s) with ecmwf-actions/reusable-workflows by @DeployDuck in https://github.com/ecmwf/anemoi-utils/pull/80

## New Contributors
* @DeployDuck made their first contribution in https://github.com/ecmwf/anemoi-utils/pull/80

**Full Changelog**: https://github.com/ecmwf/anemoi-utils/compare/0.4.11...0.4.12

## 0.4.11 (2025-01-17)

<!-- Release notes generated using configuration in .github/release.yml at develop -->

## What's Changed
### Other Changes
* Feature request: Add option to read configuration from stdin by @mpartio in https://github.com/ecmwf/anemoi-utils/pull/59
* feat(plots): Add quick map plot for debugging by @b8raoult in https://github.com/ecmwf/anemoi-utils/pull/69
* feat: added-anemoi-utils-grids-and-tests by @floriankrb in https://github.com/ecmwf/anemoi-utils/pull/74
* feat(plot): added plotting options by @NRaoult in https://github.com/ecmwf/anemoi-utils/pull/72
* ci(release): Simplify Release Workflow to Minimum by @JesperDramsch in https://github.com/ecmwf/anemoi-utils/pull/78
* feat: adding-tools-for-grids by @floriankrb in https://github.com/ecmwf/anemoi-utils/pull/76

## New Contributors
* @mpartio made their first contribution in https://github.com/ecmwf/anemoi-utils/pull/59
* @NRaoult made their first contribution in https://github.com/ecmwf/anemoi-utils/pull/72

**Full Changelog**: https://github.com/ecmwf/anemoi-utils/compare/0.4.10...0.4.11

## [0.4.5](https://github.com/ecmwf/anemoi-utils/compare/0.4.4...0.4.5) - 2024-11-06

### What's Changed

* upload with ssh by @floriankrb in https://github.com/ecmwf/anemoi-utils/pull/25
* feat: Add aliases decorator by @HCookie in https://github.com/ecmwf/anemoi-utils/pull/40

**Full Changelog**: https://github.com/ecmwf/anemoi-utils/compare/0.4.4...0.4.5

## [0.4.4](https://github.com/ecmwf/anemoi-utils/compare/0.4.3...0.4.4) - 2024-11-01

## [0.4.3](https://github.com/ecmwf/anemoi-utils/compare/0.4.1...0.4.3) - 2024-10-26

## [0.4.2](https://github.com/ecmwf/anemoi-utils/compare/0.4.1...0.4.2) - 2024-10-25

### Added

- Add supporting_arrays to checkpoints
- Add factories registry
- Optional renaming of subcommands via `command` attribute [#34](https://github.com/ecmwf/anemoi-utils/pull/34)
- `skip_on_hpc` pytest marker for tests that should not be run on HPC [36](https://github.com/ecmwf/anemoi-utils/pull/36)

## [0.4.1](https://github.com/ecmwf/anemoi-utils/compare/0.4.0...0.4.1) - 2024-10-23

## Fixed

- Fix `__version__` import in init

### Changed

- Fix: resolve mounted filesystems in provenance
- Fix pre-commit regex
- ci: extend python versions [#23] (https://github.com/ecmwf/anemoi-utils/pull/23)
- Update copyright notice

## [0.4.0](https://github.com/ecmwf/anemoi-utils/compare/0.3.18...0.4.0) - 2024-10-11

### Added

- Add anemoi-transform link to documentation
- Add CONTRIBUTORS.md (#33)

## [0.3.17](https://github.com/ecmwf/anemoi-utils/compare/0.3.13...0.3.17) - 2024-10-01

### Added

- Codeowners file
- Pygrep precommit hooks
- Docsig precommit hooks
- Changelog merge strategy- Codeowners file
- Create dependency on wcwidth. MIT licence.
- Add distribution name dictionary to provenance [#15](https://github.com/ecmwf/anemoi-utils/pull/15) & [#19](https://github.com/ecmwf/anemoi-utils/pull/19)
- Add anonymize() function.
- Add transfer to ssh:// target (experimental)
- Deprecated 'anemoi.utils.s3'

### Changed

- downstream-ci should only runs for changes in src and tests
- bugfixes for CI
- python3.9 support

### Removed

## [0.3.0] - Initial Release, utility functions

### Added

- Command line interface utility

## [0.2.0] - Initial Release, utility functions

### Changed

- updated documentation

## [0.1.0] - Initial Release, utility functions

### Added

- Documentation
- Initial implementation for a series of utility functions for used by the rest of the Anemoi packages

<!-- Add Git Diffs for Links above -->
https://github.com/ecmwf/anemoi-utils/compare/0.2.0...0.3.0
https://github.com/ecmwf/anemoi-utils/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/ecmwf/anemoi-utils/releases/tag/0.1.0

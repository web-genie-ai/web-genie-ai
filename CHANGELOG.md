# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-24
### Added
- Initial release of the project.
- Features include Image2Html.

## [1.0.1] - 2025-01-25
### Fixed
- Fixed an issue with sending session results to the stats collector.
- Fixed an issue with the high level matching score because of re-initializing CUDA in forked processes.

## [1.0.2] - 2025-01-25
### Fixed
- Fixed an issue that happens in selecting winners when the total scores are all 0.

## [1.0.3] - 2025-01-26
### Fixed
- Fixed an issue with saving the validator state, specifically the session number when the validator last set weights.

## [1.0.4] - 2025-01-27
### Changed
- Changed the wandb run name to include the version.

## [1.0.5] - 2025-01-27
### Fixed
- Fixed an issue with dictionary's size change during iteration.

## [1.0.6] - 2025-01-29
### Added
- Added an organic task.

## [1.0.7] - 2025-01-29
### Changed
- Set weights based on the total scores.

## [1.0.8] - 2025-01-29
### Changed
- Set weights to zero in the new session.

## [1.0.9] - 2025-01-30
### Changed
- Set weights based on the winner-take-all strategy.

## [1.0.10] - 2025-01-31
### Changed
- Query miners without splitting the query window.

## [1.0.11] - 2025-01-31
### Changed
- Kill the process on the port before starting the axon and the lighthouse server.
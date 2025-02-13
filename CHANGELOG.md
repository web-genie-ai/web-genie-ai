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

## [1.0.12] - 2025-02-01
### Changed
- Install sudo if it is not installed.

## [1.0.13] - 2025-02-01
### Added
- Added a balanced competition type.
### Changed
- Do not raise an exception if the error occurs while killing the process on the port.

## [1.0.14] - 2025-02-02
### Changed
- Update logging information.

## [1.1.0] - 2025-02-03
### Added
- Added a new column to the total scores table to show the average score.
### Changed
- Only reward the winner of current session.
- Reward the winners of previous sessions with a tiny weight to prevent deregistration.
- Changed the scoring mechanism to exclude the previous winner.

## [1.1.1] - 2025-02-06
### Changed
- Changed the scoring mechanism to reward all miners.

## [1.1.2] - 2025-02-07
### Fixed
- Fixed an issue with clearing the scores.

## [1.1.3] - 2025-02-07
### Changed
- Only search for official websites.

## [1.1.4] - 2025-02-10
### Added
- Save the results of the previous sessions.

## [1.1.5] - 2025-02-12
### Changed
- Switched to winner-take-all strategy.

## [1.1.6] - 2025-02-13
### Changed
- Resolved an issue with duckduckgo search returning different results for the same query.

## [1.1.7] - 2025-02-13
### Fixed
- Fixed an issue of infinite loop in synthensize a task

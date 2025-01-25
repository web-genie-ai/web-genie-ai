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

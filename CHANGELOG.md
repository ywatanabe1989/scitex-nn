# Changelog

All notable changes to `scitex-nn` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.13]

- fix(deps): repoint scitex_gen imports to the public API (`from scitex_gen import to_even/to_odd`) after scitex-gen's `_numeric` reorg; the old private `scitex_gen._to_even`/`._to_odd` paths broke against scitex-gen 0.1.10
- deps: raise `scitex-gen` floor to `>=0.1.10` (where `_numeric` + public exports landed)
- test(gate): drop stale `scitex_gen._numeric._*` cross-package gate entries (PS-140)

## [0.1.12]

- de-mock: replace all mocks with real collaborators in all test suites
- test-quality: fix TQ001/002/003/007 violations across full test tree
- cleanup: remove embedded source block from test__Hilbert.py
- cleanup: remove dead matplotlib.pyplot import from _Filters.py
- ci: disable codecov PR comments, make docs commit-back non-fatal

## [0.1.9]

- Initial CHANGELOG entry — see git log for prior history.

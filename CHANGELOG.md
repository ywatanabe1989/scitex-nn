# Changelog

All notable changes to `scitex-nn` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

- refactor: migrate `to_even` / `to_odd` from `scitex_gen` to the new
  `scitex_math` package. All four call sites (`_Filters.py`,
  `_Wavelet.py`, `_vendor_dsp_utils/_differential_bandpass_filters.py`,
  `_vendor_dsp_utils/filter.py`) and the matching test now do
  `from scitex_math import to_even, to_odd`. `scitex-gen` remains a
  runtime dep because `_PAC.py` still imports `DimHandler` from it.
  Adds `scitex-math>=0.1.0` to `[project] dependencies`.

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

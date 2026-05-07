# scitex-nn handoff — closed

**Date**: 2026-05-07. **Hand-off from**: Claude Opus 4.7 (1M ctx).
**Status**: All examples rewritten as `.ipynb` (simple → complex) per
the scitex-dsp pattern. 16 narrative notebooks cover every public
class. Audit fully clean (no `skip_rules` workarounds). Original
HANDOFF goals + the notebook-rewrite request both complete.

## What landed (final state)

### Examples (16 .ipynb, simple → complex)

| # | Notebook | Topic |
|---|---|---|
| 01 | `01_axiswise_dropout.ipynb` | `AxiswiseDropout` — drop slices along an axis |
| 02 | `02_channel_aug.ipynb` | `DropoutChannels` / `SwapChannels` / `ChannelGainChanger` |
| 03 | `03_gaussian_filter.ipynb` | `GaussianFilter` smoothing at three sigmas |
| 04 | `04_filter_bank.ipynb` | Low/High/Band/BandStop frequency response |
| 05 | `05_psd.ipynb` | `PSD` vs `scipy.signal.welch` |
| 06 | `06_freq_gain_changer.ipynb` | `FreqGainChanger` random per-band gain |
| 07 | `07_hilbert.ipynb` | `Hilbert` vs `scipy.signal.hilbert` |
| 08 | `08_spectrogram.ipynb` | `Spectrogram` STFT |
| 09 | `09_wavelet.ipynb` | `Wavelet` Morlet CWT |
| 10 | `10_modulation_index.ipynb` | `ModulationIndex` (Tort 2010) |
| 11 | `11_pac.ipynb` | `PAC` end-to-end comodulogram |
| 12 | `12_differentiable_bandpass.ipynb` | `DifferentiableBandPassFilter` |
| 13 | `13_spatial_attention.ipynb` | `SpatialAttention` |
| 14 | `14_resnet1d.ipynb` | `ResNet1D` tiny train loop |
| 15 | `15_mnet1000.ipynb` | `MNet1000` forward/backward + grad norms |
| 16 | `16_bnet.ipynb` | `BNet_v1` 2-modality |

Cell outputs baked in via `jupyter nbconvert --execute --inplace`;
GitHub renders inline figures without cloning.
`examples/00_run_all.sh` re-runs the whole gallery.

### Test mirrors (16)
`tests/examples/test_<NN>_<name>.py` per notebook — runs
`jupyter nbconvert --execute` on a tmp copy and asserts exit 0.
Skipped when `jupyter` is missing. Same pattern as scitex-dsp.

### Source bug fixes
1. **`Spectrogram.forward`** — referenced `scitex.dsp.ensure_3d` without
   importing `scitex`. Switched to vendored `_ensure_3d`.
2. **`Wavelet.forward`** — imported `scitex_dsp._ensure_3d`, but
   `scitex_dsp` is not a runtime dep of scitex-nn (CI failure on
   Python 3.12). Same fix.
3. **`GaussianFilter.init_kernels`** — `torch.tensor(sourceTensor)`
   raised UserWarning. Replaced with `kernels.detach().clone()`.
4. **`DifferentiableBandPassFilter` (vendor)** — hard-imported
   `torchaudio.prototype.functional.sinc_impulse_response`, which was
   removed in torchaudio 2.x. Added a Hann-windowed sinc fallback
   matching the original signature.
5. **`_GaussianFilter.py`** removed — orphan duplicate of the
   `GaussianFilter` class that lives in `_Filters.py`.

### Structural refactor
- `src/scitex_nn/_aug/` subpackage groups the five augmentation
  modules: `_AxiswiseDropout`, `_DropoutChannels`, `_SwapChannels`,
  `_ChannelGainChanger`, `_FreqGainChanger`. Public API unchanged
  (`scitex_nn.AxiswiseDropout` etc. still resolve).
- `tests/scitex_nn/_aug/` mirror.
- `tests/develop/test_audit.py` no longer skips `PS108b`/`PS121` —
  audit is fully green.

### CI fixes
- `pyproject.toml [docs]` extra now matches
  `docs/sphinx/requirements.txt` (`sphinx-copybutton`,
  `sphinx-autodoc-typehints` were missing).
- `tests/integration/test_cross_package_imports.py`: dropped the
  removed `scitex_dsp._ensure_3d` from the import gate.
- README gained mandatory `## Demo` and `## Architecture` sections
  (PS141, PS142).

## Remaining open quality items (optional follow-ups)

- **`_BNet_Res.py`** chains `ResNetBasicBlock(int(n_chs / 2**k))`
  for k=1-4 but never downsamples the channel dim — likely broken
  for any real input; needs an integration test.
- **`_PAC.py` `generate_surrogates`** pins `device = "cuda"` even when
  the outer module is on CPU. With `n_perm=None` (default) the path
  is skipped.
- Further structural grouping (`_transforms/`, `_coupling/`, `_arch/`)
  would clean the source root further; the current 14 flat files is
  below the threshold (15) so this is not required.

## Cross-references

- Reference layout (notebook-only): `~/proj/scitex-dsp/examples/`
- Examples skill: `~/.claude/skills/scitex/general/02_package_05_project-structure-examples.md`
- Hilbert fix memory: `~/.claude/projects/-home-ywatanabe-proj-scitex-dev/memory/scitex_nn_hilbert_fix_2026-05-07.md`

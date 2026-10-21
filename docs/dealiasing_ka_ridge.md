# Ridge-based dealiasing of Ka-band Doppler spectra (`--dealias_method ridge`)

This method dealiases Doppler spectra of a vertically pointing Ka-band cloud radar
(Metek MIRA, Nyquist velocity ±10.66 m/s) in strongly convective situations, where
hydrometeor and air velocities exceed the Nyquist interval and spectra are folded,
sometimes several times and sometimes over their full width. It works **per spectral
bin**: every bin of every spectrum gets an integer fold count *n*, so that its true
velocity is `v = v_measured + n · 2 v_Nyq`. A co-located X-band radar with a much larger
Nyquist velocity (≈ 80 m/s, practically unaliased) is used as a reference.

It was developed and tuned on one case: **2025-06-04, 18:00–18:10 UTC** (KAMACS Ka-band
and XMACS X-band, Munich). It is not a general-purpose dealiasing method; see
[Limitations](#limitations).

Code: `src/spectrawiz/dealias_ridge.py` (algorithm), `src/spectrawiz/process_data.py`
(integration). The algorithm is a 1:1 port of the debug pipeline
`debug/scripts_foldfix2/ka_ridge1_topdown.py` + `ka_ridge1_final_spectrogram.py` with
default settings; the port reproduces that pipeline bin for bin (see [Validation](#validation)).

---

## How to run

```bash
PYTHONPATH=src python -m spectrawiz.process_data \
  --date 20250604 --raw_path /project/meteo/data/kamacs/mom/2025/06/04/ \
  --output_dir /scratch/<user>/radar_processed/ --hourly --include_moments --include_ldr \
  --file_pattern '20250604_18*.znc' \
  --dealias_method ridge \
  --dealias_ref_pattern '/project/meteo/data/xmacs/mom/2025/06/04/20250604_18*.znc' \
  --start_time 18:00 --end_time 18:10 --dealias_n_jobs 4 --overwrite
```

| Option | Meaning |
|---|---|
| `--dealias_method ridge` | use this method |
| `--dealias_ref_pattern` | glob for the **raw** X-band spectra files (Metek `.znc`). They are cut to the processing window ±60 s and processed with the Metek backend (`sZeCo`). |
| `--start_time`, `--end_time` | restrict hourly processing to a time-of-day window (`HH:MM` or `HH:MM:SS`). The output file name gets the window as suffix, e.g. `20250604_18_metek_hourly_proc_180000-181000.nc`. |
| `--dealias_n_jobs` | number of parallel processes (one time step per task) |

Run time: about 15 s per time step and process (10 min of data ≈ 145 time steps ≈ 10 min
with 4 processes). Memory ≈ 1–2 GB per process.

From Python:

```python
from spectrawiz.dealias_ridge import dealias_ka_ridge, RidgeDealiasParams
res = dealias_ka_ridge(ds_ka_processed.assign_attrs(vnyq=10.6615), ds_x_processed, n_jobs=4)
```

`dealias_ka_ridge_profile(...)` dealiases a single profile and returns all intermediate
results (ridges, folds, shifts).

---

## Output

The dealiased file contains the spectra (`sZeCo`, `sZeCx`, `sLDR`) on an **extended
velocity axis** that covers all folds occurring in the file (in steps of one Nyquist
interval). The per-bin folds (`n_folds_Co`) are used by the writer to place each bin and
are not written themselves; the same folds are applied to all spectral variables.

| Variable | Dims | Meaning |
|---|---|---|
| `sZeCo`, `sZeCx`, `sLDR` | time, range, Vel (extended) | spectra placed at their dealiased velocity |
| `dealias_in_window` | time, range, Vel (extended) | 1 = bin inside the ridge window (the part the method assigned to a population), 0 = bin outside the window, kept at its raw fold (n = 0, plus the whole-row shift). **Nothing is removed**; use this mask to drop bins if wanted. |
| `MDV_Co`, `WIDTH_Co` (and `_Cx`) | time, range | mean Doppler velocity and spectral width recomputed from the dealiased spectrum, **deliberately from all bins** (inside and outside the window). Where they deviate from `dealias_ridge1` or become noisy (e.g. weak echo below ~2 km with stray bins outside the window), the dealiasing is less trustworthy. |
| `ZeCo`, `ZeCx`, `LDR` | time, range | unchanged (independent of the fold) |
| `dealias_ridge1`, `dealias_ridge2` | time, range | dealiased velocity of ridge 1 (dominant population) and ridge 2 (second population, NaN where none) |
| `dealias_fold_shift` | time, range | whole-row shift applied to each gate (local jump correction + global X-band fold) |
| `dealias_global_shift` | time | global fold chosen against X-band |
| `dealias_status` | time | 0 ok, 1 no sustained signal run, 2 no clear start gate, 3 no X-band spectrum within 10 s. For status ≠ 0 all folds are 0 (spectra left as measured). |
| `dealias_xband_offset` | time | time offset to the X-band spectrum used [s] |

The parameter set is stored in the global attribute `dealias_ridge_params`.

---

## Algorithm

All steps work in **periodic space**: each Ka spectrum (512 bins, one Nyquist interval of
21.32 m/s) is tiled into 5 copies side by side (−53 to +53 m/s). A population that wraps
around ±v_Nyq is then simply a continuous feature that crosses a copy boundary. Signal =
noise-subtracted `sZeCo > 0`; bins without signal score −40 dB.

Parameters are in `RidgeDealiasParams`; defaults in brackets.

### 1. X-band reference ridge
In the X-band spectrum (3–9 km) a single dynamic-programming (DP) ridge is traced:
the path through the spectrogram that maximises the summed power minus a penalty for
velocity steps between gates (max 3 m/s per gate, 2 dB per m/s). It follows the
**dominant peak**, not the mean velocity (VELg), which differs from the peak in
multimodal spectra. The ridge is interpolated to the Ka range gates. The X time step
closest to the Ka time step is used (here ≤ 2 s apart).

### 2. Start point of Ka ridge 1
The top of the topmost sustained signal run (≥ 5 gates, below 10 km) whose peak exceeds
−35 dB. At that gate the start bin is the signal bin (in any copy) closest to the X-band
ridge; without X-band there, the strongest bin of the middle copy.

### 3. Ka ridge 1 (dominant population)
Top-down DP through the tiled Ka spectrogram from the start bin (same step limit and
penalty as for X-band). A **soft bias** of 1 dB per m/s distance to the X-band ridge is
subtracted from the reward, so that a comparably strong but wrong population (often a
neighbouring copy) does not win. The bias does not force Ka onto X-band.

### 4. Edges of ridge 1
For each gate, the contiguous run of signal bins containing ridge 1's bin; its two ends
are the left and right edge of the population.

### 5. Ridge 2 (second population)
- Peaks in each spectrum with a Savitzky–Golay-smoothed peak finder (prominence 3 dB,
  window 2 × 0.4 m/s). The second-strongest peak counts if it is ≥ 1 m/s from ridge 1 and
  at most 30 dB weaker.
- Gaps of ≤ 3 gates are bridged; only runs of ≥ 7 gates are kept.
- Within each run, ridge 2 is a DP restricted to ±3 m/s around the second peak (middle copy).
- **Edge-between check:** ridge 2 is kept only where ridge 1's edge lies strictly between
  ridge 1 and ridge 2 (i.e. there is a gap without signal between the populations), with
  gaps ≤ 3 gates bridged and runs ≥ 3 gates.

### 6. Cuts between the populations
Where ridge 2 exists, the spectral minima found by the peak finder are the candidate
cuts. For each ridge the nearest minimum on its own **outer** side is taken (≥ 1 m/s from
the ridge, ≥ 2 dB below the main peak, not at the window edge). Missing cuts are filled
from the ridge's own edge if it agrees with the gate below within 2 m/s, else from the
gate below. Short runs (≤ 6 gates) of cuts that disagree with both neighbours (by > 2 m/s)
while the neighbours agree with each other are replaced by linear interpolation.

### 7. Fold per bin
- Gates with ridge 2: ridge 1 gets the window from its cut to the midpoint between the
  ridges, ridge 2 from the midpoint to its cut.
- Other gates: the window between ridge 1's edges.
- Each bin is assigned the copy (fold) in which it falls inside the window — per bin, so
  a population that straddles ±v_Nyq is handled correctly.
- Bins outside the window keep fold 0 and are flagged (`dealias_in_window = 0`).
- Gates without edge information: all bins at ridge 1's fold.

### 8. Local jump correction
For each gate, the lowest and highest dealiased velocity of the shown spectrum, using
only bins ≥ 3 dB above the noise. Walking down from the start gate: if **one** of these
edges jumps by a whole number of Nyquist intervals (2 v_Nyq ± 1 m/s) relative to the gate
above, the ridge has switched to a neighbouring copy; that gate and all gates below it are
shifted back by that number of folds (whole rows). If both edges jump by different
amounts, the gate is skipped.

### 9. Global fold
The whole (locally corrected) profile is shifted by the integer number of folds
(−2 … +2) that minimises the RMS difference between ridge 1 and the X-band ridge. This
fixes the absolute fold; the local structure is not changed.

Steps 8 and 9 are applied **after** all detection, as whole-row shifts of the finished
spectrogram, so they never change which bins belong together.

---

## Validation

Case 2025-06-04 18:00–18:10 UTC, 145 Ka time steps:

- 132 time steps dealiased, 13 skipped (no clear start gate; status 2).
- Local jump corrections in 27 time steps (34 jumps, most at ≈ 3.0 km at the bottom of the
  main echo); a global shift ≠ 0 in 1 time step (18:02:35).
- Reference plots (final reconstruction and top-down diagnostics for every time step):
  `debug/all_time_steps_foldfix6_rerun/`.
- Port check: `dealias_ka_ridge_profile` was compared with the debug pipeline on identical
  inputs for all 145 time steps — fold per bin, window mask and shifts are identical.
  (One float32 vs float64 difference was found and fixed: in signal-free gaps several DP
  paths tie exactly and rounding decides between them; the reward is now float64 as in the
  pipeline.)
- End-to-end run through `process_data` (command as in [How to run](#how-to-run)):
  145 time steps in 2 min 48 s with 4 processes, peak memory 5.1 GB, output 85 MB. Status,
  local jumps (height and direction) and global shift of every time step are identical to
  the reference summary (`debug/all_time_steps_foldfix6_rerun/summary.txt`). Plots read
  straight from the output file: `debug/ridge_backend_check/`
  (script `plot_from_file.py`).

Judged by eye, the dealiased spectrograms are continuous and plausible for most time
steps; the remaining errors are described below. Agreement with X-band is used only as a
plausibility check: the two radars do not sample exactly the same volume and their time
steps are up to a few seconds apart, which in a convective cell changes the spectra.

---

## Limitations

- **Spectra filling a whole fold.** Where the Ka spectrum is wider than the Nyquist
  interval (e.g. 18:02:02 and 18:02:06 at ≈ 4.85 km), the window is cut wherever the
  weakest bins happen to be. Ridge 1 drifts gradually through continuous signal into the
  next copy, and the edges jump by a non-integer amount (≈ 13–16 m/s); neither a local
  correction nor a whole-fold shift can make the spectrogram continuous there.
- **Weak echo below the main echo.** Jump corrections triggered by a single edge in the
  weak, patchy lower echo (≈ 2–3 km) are sometimes spurious.
- **Second population only with a gap.** Ridge 2 requires a gap without signal between
  the populations (edge-between check). Touching populations are treated as one.
- **Ridge 2 copy.** Ridge 2 is always placed in the middle copy; where ridge 1 sits in
  another copy this can put ridge 2 a fold off (e.g. 18:02:23 near 3.5–3.85 km).
- **X-band dependence.** Start point, bias and absolute fold come from X-band. Without
  X-band within 10 s, a time step is not dealiased (status 3).
- **Tuned on one case and one radar.** Parameters (thresholds, smoothing, 3–9 km X-band
  range, 10 km start region) were chosen for this case; the X-band DP assumes v_Nyq = 80 m/s.

---

## Alternatives that were tried and rejected

| Variant | Why rejected |
|---|---|
| Jump detection with "≥ v_Nyq" and merging of left/right edge jumps (`all_time_steps_foldfix8_merge3`) | many more corrections (86 vs 27 time steps, 19 vs 1 global shifts), worse in most time steps |
| Ridge 2 in the copy nearest ridge 1 | fixed 18:02:23, broke 18:00:40 (populations > half a fold apart) |
| 5 dB instead of 3 dB for the jump-detector edges | false jump at 18:02:35 |
| Weaker or no X-band bias | ridge 1 loses the population (e.g. 18:01:54) |
| sLDR-based ridge, cut or ridge 2 (`RIDGE1_MODE`, `CUT_SOURCE`, `RIDGE2_SOURCE` in the debug script) | sLDR also fills the whole fold where Ze does; requires cross-channel SNR, so low-LDR populations disappear |
| RADALIA IterBU (Tolentino et al., 2026) | bottom-up start propagates the fold error of the aliased weak lower echo; circular mean unstable for full-fold or touching multimodal spectra (`debug/iterbu/`) |
| RADALIA IntB (single cut at gap or minimum, first guess from X-band or previous profile) | works for one population folded onto itself, but the single cut jumps between populations in bimodal regions (stripes, `debug/intb/`) |

---

## References

- Tolentino, M., et al.: RADALIA: A new tool for dealiasing Doppler cloud radar
  observations, EGUsphere preprint, https://doi.org/10.5194/egusphere-2026-5139, 2026.
- Peak finding follows the approach of PEAKO/peakTree (Savitzky–Golay smoothing,
  prominence-based peaks, minima between adjacent peaks); see `_find_spectral_peaks` in
  `processing_common.py`.

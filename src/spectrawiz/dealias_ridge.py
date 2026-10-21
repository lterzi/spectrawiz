"""
Ridge-based dealiasing of Ka-band (Metek MIRA) Doppler spectra, anchored to a
co-located wide-Nyquist X-band radar.

Port of the debug pipeline validated on 2025-06-04 18:00-18:10 UTC
(debug/scripts_foldfix2/ka_ridge1_topdown.py + ka_ridge1_final_spectrogram.py, default
settings, results in debug/all_time_steps_foldfix6_rerun). See
docs/dealiasing_ka_ridge.md for the full description of every step.

Per time step:
  1. Tile the spectrum over N_EXT periodic copies (extended velocity axis).
  2. X-band ridge: dynamic-programming (DP) ridge through the X-band spectrum (3-9 km),
     interpolated to the Ka range gates.
  3. Ka ridge 1: top-down DP through the tiled Ka spectrum, starting at the top of the
     echo at the signal bin nearest the X-band ridge, softly biased toward the X-band ridge.
  4. Edges of ridge 1: the contiguous run of signal containing ridge 1's bin.
  5. Ridge 2: per-gate bimodality (peak finder), bridged and persistence-filtered, tracked
     by a local DP; kept only where ridge 1's edge lies between ridge 1 and ridge 2.
  6. Cuts between the populations (spectral minima), with fallbacks and bridging.
  7. Fold per bin: ridge-2 gates are split at the cuts, other gates take ridge 1's
     window; bins outside the window keep fold 0 (flagged in `in_window`).
  8. Local jump correction: where an edge of the shown spectrum (bins >= 3 dB above
     noise) jumps by a whole Nyquist interval (+-1 m/s), shift that gate and all gates
     below it back.
  9. Global fold: one integer shift for the whole profile that best matches X-band.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

from .processing_common import _ridge_dp, _bool_runs, _find_spectral_peaks

# dealias_status values
STATUS_OK = 0
STATUS_NO_SIGNAL_RUN = 1   # no sustained signal run within the region of interest
STATUS_NO_START_GATE = 2   # topmost run has no gate with a clear peak above the floor
STATUS_NO_XBAND = 3        # no X-band spectrum close enough in time


@dataclass(frozen=True)
class RidgeDealiasParams:
    n_ext: int = 5                     # periodic copies of the Nyquist interval
    floor_db: float = -40.0            # reward for bins without signal
    span_ms: float = 0.4               # Savitzky-Golay span for the peak finder (doubled below)
    savgol_factor: int = 2
    max_jump_ms: float = 3.0           # DP: max velocity step between adjacent gates
    jump_penalty_per_ms: float = 2.0   # DP: penalty per m/s of step
    prom_db: float = 3.0               # peak finder prominence
    polyorder: int = 3
    min_sep_ms: float = 1.0            # ridge 2: min separation of the two peaks
    max_sep_drop_db: float = 30.0      # ridge 2: max power difference of the two peaks
    min_run_gates: int = 7             # ridge 2: persistence
    local_window_ms: float = 3.0       # ridge 2: half-width of the local DP window
    min_peak_sep_ms: float = 1.0       # peak filter / cut candidates: min distance to ridge
    min_peak_drop_db: float = 2.0      # cut candidates: min dip below the top peak
    max_bridge_gates: int = 3          # bridging of short gaps (bimodality, edge-between mask)
    min_ridge2_run_gates: int = 3      # edge-between mask: min run length
    fallback_agree_ms: float = 2.0     # cut fallback / bad-run bridging tolerance
    max_bad_run: int = 6               # cut bad-run bridging: max run length
    roi_max_range_m: float = 10000.0   # start gate search region
    min_start_run_gates: int = 5
    min_start_margin_db: float = 5.0   # start gate peak must exceed floor_db by this
    xband_bias_db_per_ms: float = 1.0  # soft bias of ridge 1 toward the X-band ridge
    xband_range_m: tuple = (3000.0, 9000.0)
    xband_vnyq_ms: float = 80.0        # X-band Nyquist velocity used for its DP bin width
    edge_snr_db: float = 3.0           # jump detector: bins counted for the edges
    edge_jump_tol_ms: float = 1.0      # jump detector: tolerance around a whole interval


def _xband_ridge_on_ka(w_x, x_vel, x_range, ka_range, p: RidgeDealiasParams):
    """X-band DP ridge (dominant peak, not VELg) interpolated to the Ka range gates."""
    w_x = np.nan_to_num(w_x, nan=0.0)
    x_r_idx = np.where((x_range >= p.xband_range_m[0]) & (x_range <= p.xband_range_m[1]))[0]
    db_x = np.where(w_x[x_r_idx] > 0, 10.0 * np.log10(np.maximum(w_x[x_r_idx], 1e-30)), p.floor_db)
    db_x = np.maximum(db_x, p.floor_db)
    x_bin_width = (2 * p.xband_vnyq_ms) / x_vel.size
    x_max_jump_bins = max(1, int(round(p.max_jump_ms / x_bin_width)))
    x_path = _ridge_dp(db_x, x_max_jump_bins, p.jump_penalty_per_ms * x_bin_width)
    return np.interp(ka_range, x_range[x_r_idx], x_vel[x_path], left=np.nan, right=np.nan)


def dealias_ka_ridge_profile(
    w: np.ndarray,
    noise: np.ndarray,
    vel: np.ndarray,
    vnyq: float,
    ka_range: np.ndarray,
    xridge_on_ka: np.ndarray,
    params: RidgeDealiasParams | None = None,
) -> dict:
    """
    Dealias one Ka profile.

    Parameters
    ----------
    w            : (n_range, nfft) noise-subtracted co-channel spectrum (sZeCo), linear
    noise        : (n_range,) noise level per gate (NoisePowCo), same units as w
    vel          : (nfft,) ascending Doppler velocity axis of one Nyquist interval
    vnyq         : Nyquist velocity [m s-1]
    ka_range     : (n_range,) range [m]
    xridge_on_ka : (n_range,) X-band ridge velocity at the Ka gates (NaN where unknown)

    Returns a dict with
      n_folds   (n_range, nfft) int8  final fold per bin (incl. local + global shift)
      in_window (n_range, nfft) bool  bin lies inside the ridge window (shown in the plots)
      ridge1, ridge2 (n_range,)       dealiased ridge velocities [m s-1]
      fold_shift (n_range,) int8      local + global shift applied to each gate
      global_shift int, status int
    """
    p = params or RidgeDealiasParams()
    n_range, nfft = w.shape
    interval = 2.0 * vnyq
    bin_width = interval / nfft
    savgol_window = int(round(p.span_ms / bin_width)) * p.savgol_factor
    if savgol_window % 2 == 0:
        savgol_window += 1
    max_jump_bins = max(1, int(round(p.max_jump_ms / bin_width)))
    penalty_per_bin = p.jump_penalty_per_ms * bin_width
    local_window_bins = max(1, int(round(p.local_window_ms / bin_width)))

    center_k = p.n_ext // 2
    ext_vel = np.concatenate([vel + k * interval for k in range(-center_k, center_k + 1)])
    ext_len = ext_vel.size
    center_offset = center_k * nfft

    out = dict(
        n_folds=np.zeros((n_range, nfft), dtype=np.int8),
        in_window=np.zeros((n_range, nfft), dtype=bool),
        ridge1=np.full(n_range, np.nan),
        ridge2=np.full(n_range, np.nan),
        fold_shift=np.zeros(n_range, dtype=np.int8),
        global_shift=0,
        status=STATUS_OK,
    )

    # 1. periodic tiling
    above = (w > 0) & ~np.isnan(w)
    has_signal = above.any(axis=1)
    db_row = np.maximum(np.where(above, 10.0 * np.log10(np.maximum(np.nan_to_num(w), 1e-30)), p.floor_db), p.floor_db)
    # float64 on purpose: in signal-free gaps several DP paths tie exactly, and float32
    # rounding of the X-band bias term (~1e-6 dB) is enough to change which one wins
    ext_db = np.tile(db_row.astype(np.float64), (1, p.n_ext))
    ext_above = np.tile(above, (1, p.n_ext))

    # 3. start gate: top of the topmost sustained signal run with a clear peak
    min_start_power_db = p.floor_db + p.min_start_margin_db
    in_roi = has_signal & (ka_range <= p.roi_max_range_m)
    runs = [(s, e) for (s, e) in _bool_runs(in_roi) if (e - s + 1) >= p.min_start_run_gates]
    if not runs:
        out["status"] = STATUS_NO_SIGNAL_RUN
        return out
    top_s, top_e = max(runs, key=lambda se: se[1])
    start_ir = None
    for ir in range(top_e, top_s - 1, -1):
        if ext_db[ir, center_offset:center_offset + nfft].max() >= min_start_power_db:
            start_ir = ir
            break
    if start_ir is None:
        out["status"] = STATUS_NO_START_GATE
        return out

    # start bin: signal bin (any copy) nearest the X-band ridge, else strongest peak
    xv_start = xridge_on_ka[start_ir]
    signal_mask = ext_above[start_ir] & (ext_db[start_ir] >= min_start_power_db)
    if np.isfinite(xv_start) and signal_mask.any():
        cand = np.where(signal_mask)[0]
        start_bin = int(cand[np.argmin(np.abs(ext_vel[cand] - xv_start))])
    else:
        start_bin = center_offset + int(np.argmax(ext_db[start_ir, center_offset:center_offset + nfft]))

    # ridge 1: top-down DP with soft X-band bias, start bin locked
    gate_order = np.arange(start_ir, -1, -1)
    db_sub = ext_db[gate_order].copy()
    for k, ir in enumerate(gate_order):
        if np.isfinite(xridge_on_ka[ir]):
            db_sub[k, :] -= p.xband_bias_db_per_ms * np.abs(ext_vel - xridge_on_ka[ir])
    db_sub[0, :] = -np.inf
    db_sub[0, start_bin] = ext_db[start_ir, start_bin]
    path1 = np.full(n_range, -1, dtype=int)
    path1[gate_order] = _ridge_dp(db_sub, max_jump_bins, penalty_per_bin)
    walked = np.zeros(n_range, dtype=bool)
    walked[gate_order] = True
    vel1 = np.where(walked, ext_vel[np.clip(path1, 0, ext_len - 1)], np.nan)

    # 4. edges of ridge 1 (contiguous signal run containing its bin)
    def _run_edges(ir, b):
        if not ext_above[ir, b]:
            return np.nan, np.nan
        run = next((r for r in _bool_runs(ext_above[ir]) if r[0] <= b <= r[1]), None)
        return (np.nan, np.nan) if run is None else (ext_vel[run[0]], ext_vel[run[1]])

    edge_l1 = np.full(n_range, np.nan)
    edge_r1 = np.full(n_range, np.nan)
    for ir in np.where(walked)[0]:
        edge_l1[ir], edge_r1[ir] = _run_edges(ir, path1[ir])

    # 5. ridge 2
    def _nearest_periodic_vel(raw_vel, target):
        cands = raw_vel + np.arange(-2, 3) * interval
        return cands[np.argmin(np.abs(cands - target))]

    def _filter_real_peaks(peaks, ref_vel):
        if len(peaks) < 2:
            return peaks
        ps = sorted(peaks, key=lambda q: -q["power_db"])
        kept = [ps[0]]
        for q in ps[1:]:
            if abs(_nearest_periodic_vel(vel[q["peak_bin"]], ref_vel) - ref_vel) >= p.min_peak_sep_ms:
                kept.append(q)
        return kept

    def _bridge_mask(mask):
        out_m = mask.copy()
        for (s, e) in _bool_runs(~mask):
            if (e - s + 1) <= p.max_bridge_gates and s > 0 and e < mask.size - 1 and mask[s - 1] and mask[e + 1]:
                out_m[s:e + 1] = True
        return out_m

    bimodal = np.zeros(n_range, dtype=bool)
    peak2_raw = np.full(n_range, np.nan)
    for ir in range(n_range):
        if not has_signal[ir] or not np.isfinite(vel1[ir]):
            continue
        peaks = _find_spectral_peaks(w[ir], p.prom_db, savgol_window, p.polyorder, 0.0)
        if len(peaks) < 2:
            continue
        ps = _filter_real_peaks(peaks, vel1[ir])
        if len(ps) < 2:
            continue
        q1, q2 = ps[0], ps[1]
        if (abs(vel[q1["peak_bin"]] - vel[q2["peak_bin"]]) >= p.min_sep_ms
                and q1["power_db"] - q2["power_db"] <= p.max_sep_drop_db):
            bimodal[ir] = True
            peak2_raw[ir] = vel[q2["peak_bin"]]
    bimodal_b = bimodal.copy()
    for (s, e) in _bool_runs(~bimodal):
        n_gap = e - s + 1
        if n_gap <= p.max_bridge_gates and s > 0 and e < n_range - 1 and bimodal[s - 1] and bimodal[e + 1]:
            bimodal_b[s:e + 1] = True
            for k, ir in enumerate(range(s, e + 1), start=1):
                peak2_raw[ir] = peak2_raw[s - 1] + k / (n_gap + 1) * (peak2_raw[e + 1] - peak2_raw[s - 1])
    long_runs2 = [(s, e) for (s, e) in _bool_runs(bimodal_b) if (e - s + 1) >= p.min_run_gates]

    path2 = np.full(n_range, -1, dtype=int)
    ridge2_valid = np.zeros(n_range, dtype=bool)
    for (s, e) in long_runs2:
        sub = ext_db[s:e + 1]
        restricted = np.full_like(sub, p.floor_db)
        for k, ir in enumerate(range(s, e + 1)):
            cb = center_offset + int(np.argmin(np.abs(vel - peak2_raw[ir])))
            lo, hi = max(0, cb - local_window_bins), min(ext_len, cb + local_window_bins + 1)
            restricted[k, lo:hi] = sub[k, lo:hi]
        path2[s:e + 1] = _ridge_dp(restricted, max_jump_bins, penalty_per_bin)
        ridge2_valid[s:e + 1] = True
    vel2_full = np.where(ridge2_valid, ext_vel[np.clip(path2, 0, ext_len - 1)], np.nan)

    # edge-between mask: ridge 1's edge strictly between ridge 1 and ridge 2
    between = np.zeros(n_range, dtype=bool)
    for ir in np.where(ridge2_valid)[0]:
        v1, v2 = vel1[ir], vel2_full[ir]
        if not (np.isfinite(v1) and np.isfinite(v2)):
            continue
        edge = edge_r1[ir] if v2 > v1 else edge_l1[ir]
        if np.isfinite(edge) and min(v1, v2) < edge < max(v1, v2):
            between[ir] = True
    final_runs = [(s, e) for (s, e) in _bool_runs(_bridge_mask(between)) if (e - s + 1) >= p.min_ridge2_run_gates]
    keep = np.zeros(n_range, dtype=bool)
    for (s, e) in final_runs:
        keep[s:e + 1] = True
    ridge2_valid &= keep
    vel2 = np.where(ridge2_valid, vel2_full, np.nan)

    edge_l2 = np.full(n_range, np.nan)
    edge_r2 = np.full(n_range, np.nan)
    for ir in np.where(ridge2_valid)[0]:
        edge_l2[ir], edge_r2[ir] = _run_edges(ir, path2[ir])

    # 6. cuts: nearest real minimum on each ridge's outer side
    cut1 = np.full(n_range, np.nan)
    cut2 = np.full(n_range, np.nan)
    for ir in np.where(ridge2_valid)[0]:
        peaks = _find_spectral_peaks(w[ir], p.prom_db, savgol_window, p.polyorder, 0.0)
        if len(peaks) < 2:
            continue
        peaks = _filter_real_peaks(peaks, vel1[ir])
        if len(peaks) < 2:
            continue
        boundary_bins = sorted(set(b for q in peaks for b in (q["lo"], q["hi"]) if b != 0 and b != nfft - 1))
        lo_v, hi_v = min(vel1[ir], vel2[ir]), max(vel1[ir], vel2[ir])
        p1_db = peaks[0]["power_db"]

        def _bin_db(b):
            x = w[ir][b]
            return 10.0 * np.log10(x) if (x > 0 and not np.isnan(x)) else p.floor_db

        c1 = [v for b in boundary_bins for v in [_nearest_periodic_vel(vel[b], vel1[ir])]
              if (v <= lo_v if vel1[ir] <= vel2[ir] else v >= hi_v)
              and abs(v - vel1[ir]) >= p.min_peak_sep_ms and (p1_db - _bin_db(b)) >= p.min_peak_drop_db]
        c2 = [v for b in boundary_bins for v in [_nearest_periodic_vel(vel[b], vel2[ir])]
              if (v <= lo_v if vel2[ir] <= vel1[ir] else v >= hi_v)
              and abs(v - vel2[ir]) >= p.min_peak_sep_ms and (p1_db - _bin_db(b)) >= p.min_peak_drop_db]
        if c1:
            cut1[ir] = min(c1, key=lambda v: abs(v - vel1[ir]))
        if c2:
            cut2[ir] = min(c2, key=lambda v: abs(v - vel2[ir]))

    # fallback: own edge if it agrees with the gate below, else the gate below's cut
    for ir in np.where(ridge2_valid)[0]:
        for cut, e_cand in ((cut1, edge_l1[ir] if vel1[ir] <= vel2[ir] else edge_r1[ir]),
                            (cut2, edge_l2[ir] if vel2[ir] <= vel1[ir] else edge_r2[ir])):
            if np.isfinite(cut[ir]):
                continue
            ref = cut[ir - 1] if ir > 0 else np.nan
            if np.isfinite(e_cand) and np.isfinite(ref) and abs(e_cand - ref) <= p.fallback_agree_ms:
                cut[ir] = e_cand
            elif np.isfinite(ref):
                cut[ir] = ref

    # isolated bad-run bridging
    def _bridge_bad_runs(arr):
        n = arr.size
        for s in range(1, n):
            if not np.isfinite(arr[s - 1]):
                continue
            for L in range(1, p.max_bad_run + 1):
                e = s + L
                if e >= n or not np.isfinite(arr[e]):
                    continue
                run = arr[s:e]
                if not np.isfinite(run).all():
                    break
                before, after = arr[s - 1], arr[e]
                if abs(before - after) > p.fallback_agree_ms:
                    continue
                if (np.all(np.abs(run - before) > p.fallback_agree_ms)
                        and np.all(np.abs(run - after) > p.fallback_agree_ms)):
                    for k, g in enumerate(range(s, e), start=1):
                        arr[g] = before + k / (L + 1) * (after - before)
                    break

    _bridge_bad_runs(cut1)
    _bridge_bad_runs(cut2)

    # 7. fold per bin (before corrections)
    fold1 = (path1 // nfft) - center_k
    n_folds = np.zeros((n_range, nfft), dtype=np.int8)
    in_window = np.zeros((n_range, nfft), dtype=bool)
    for ir in range(n_range):
        if not has_signal[ir] or not walked[ir]:
            continue
        raw_above = above[ir]
        k_for_bin = np.full(nfft, 99, dtype=int)
        if ridge2_valid[ir] and np.isfinite(cut1[ir]) and np.isfinite(cut2[ir]):
            mid = 0.5 * (vel1[ir] + vel2[ir])
            lo1, hi1 = sorted([cut1[ir], mid])
            lo2, hi2 = sorted([mid, cut2[ir]])
            for k in range(-2, 3):
                v_k = vel + k * interval
                r1 = (v_k >= lo1) & (v_k <= hi1) & (k_for_bin == 99)
                k_for_bin[r1] = k
                r2 = (v_k >= lo2) & (v_k <= hi2) & (k_for_bin == 99)
                k_for_bin[r2] = k
        elif np.isfinite(edge_l1[ir]) and np.isfinite(edge_r1[ir]):
            lo_e, hi_e = min(edge_l1[ir], edge_r1[ir]), max(edge_l1[ir], edge_r1[ir])
            for k in range(-2, 3):
                v_k = vel + k * interval
                inw = (v_k >= lo_e) & (v_k <= hi_e) & (k_for_bin == 99)
                k_for_bin[inw] = k
        else:
            n_folds[ir, :] = fold1[ir]
            in_window[ir, :] = raw_above
            continue
        valid = (k_for_bin != 99) & raw_above
        n_folds[ir, valid] = k_for_bin[valid]
        in_window[ir, :] = valid

    # 8. local jump correction on the edges of the shown spectrum (>= edge_snr_db above noise)
    edge_min_w = (10 ** (p.edge_snr_db / 10.0) - 1.0) * np.asarray(noise, dtype=float).reshape(-1)
    sel_l = np.full(n_range, np.nan)
    sel_r = np.full(n_range, np.nan)
    for ir in range(n_range):
        strong = in_window[ir] & (w[ir] >= edge_min_w[ir])
        if strong.any():
            v_shown = vel[strong] + n_folds[ir, strong] * interval
            sel_l[ir], sel_r[ir] = v_shown.min(), v_shown.max()

    local_shift = np.zeros(n_range, dtype=int)
    cum = 0
    for i in range(1, gate_order.size):
        ir_prev, ir = gate_order[i - 1], gate_order[i]
        if (np.isfinite(sel_l[ir]) and np.isfinite(sel_r[ir])
                and np.isfinite(sel_l[ir_prev]) and np.isfinite(sel_r[ir_prev])):
            d_l, d_r = sel_l[ir] - sel_l[ir_prev], sel_r[ir] - sel_r[ir_prev]
            k_l, k_r = round(d_l / interval), round(d_r / interval)
            l_ok = k_l != 0 and abs(d_l - k_l * interval) <= p.edge_jump_tol_ms
            r_ok = k_r != 0 and abs(d_r - k_r * interval) <= p.edge_jump_tol_ms
            k_jump = 0
            if l_ok and r_ok:
                k_jump = k_l if k_l == k_r else 0  # disagreeing edges: skipped
            elif l_ok:
                k_jump = k_l
            elif r_ok:
                k_jump = k_r
            cum -= k_jump
        local_shift[ir] = cum

    # 9. global fold vs X-band ridge
    vel1_local = vel1 + local_shift * interval
    best_k, best_rmse = 0, np.inf
    for k in range(-center_k, center_k + 1):
        shifted = vel1_local + k * interval
        m = np.isfinite(shifted) & np.isfinite(xridge_on_ka)
        if not m.any():
            continue
        rmse = np.sqrt(np.mean((shifted[m] - xridge_on_ka[m]) ** 2))
        if rmse < best_rmse:
            best_rmse, best_k = rmse, k

    fold_shift = np.where(walked, local_shift + best_k, 0)
    out.update(
        n_folds=(n_folds + fold_shift[:, None]).astype(np.int8),
        in_window=in_window,
        ridge1=vel1 + fold_shift * interval,
        ridge2=vel2 + fold_shift * interval,
        fold_shift=fold_shift.astype(np.int8),
        global_shift=int(best_k),
    )
    return out


def _profile_worker(args):
    return dealias_ka_ridge_profile(*args)


def dealias_ka_ridge(
    ds_ka: xr.Dataset,
    ds_x: xr.Dataset,
    *,
    params: RidgeDealiasParams | None = None,
    max_xband_offset: str = "10s",
    n_jobs: int = 1,
    channel: str = "Co",
) -> xr.Dataset:
    """
    Dealias every time step of a processed Ka dataset.

    ds_ka : processed Ka spectra (sZeCo, NoisePowCo, Vel ascending, range, time) plus
            the NyquistVelocity in attrs['vnyq'] or derived from the Vel axis
    ds_x  : processed X-band spectra (sZeCo, Vel, range, time); the X time step nearest to
            each Ka time step is used, if within max_xband_offset

    Returns a Dataset with
      n_folds_<channel> (time, range, Vel) int8 -- read by the dealias writer
      dealias_in_window (time, range, Vel) int8 -- 1 where the bin is inside the ridge window
      dealias_ridge1, dealias_ridge2 (time, range) [m s-1], dealias_fold_shift (time, range)
      dealias_global_shift, dealias_status, dealias_xband_offset (time)
    """
    p = params or RidgeDealiasParams()
    vel = ds_ka["Vel"].values
    if not np.all(np.diff(vel) > 0):
        raise ValueError("dealias_ka_ridge expects an ascending Vel axis")
    nfft = vel.size
    vnyq = float(ds_ka.attrs.get("vnyq", 0.5 * (vel[-1] - vel[0] + (vel[1] - vel[0]))))
    ka_range = ds_ka["range"].values
    x_vel, x_range, x_times = ds_x["Vel"].values, ds_x["range"].values, ds_x["time"].values
    tol = pd.Timedelta(max_xband_offset)

    jobs, x_offsets, no_x = [], [], []
    for it, t in enumerate(ds_ka["time"].values):
        dt = np.abs((x_times - t).astype("timedelta64[ms]").astype(float))
        xi = int(np.argmin(dt))
        x_offsets.append(dt[xi] / 1000.0)
        if pd.Timedelta(milliseconds=dt[xi]) > tol:
            xridge = np.full(ka_range.size, np.nan)
            no_x.append(it)
        else:
            xridge = _xband_ridge_on_ka(ds_x["sZeCo"].isel(time=xi).values, x_vel, x_range, ka_range, p)
        w = ds_ka[f"sZe{channel}"].isel(time=it).values
        noise = ds_ka[f"NoisePow{channel}"].isel(time=it).values
        jobs.append((w, noise, vel, vnyq, ka_range, xridge, p))

    if n_jobs > 1:
        with ProcessPoolExecutor(n_jobs) as ex:
            results = list(ex.map(_profile_worker, jobs, chunksize=1))
    else:
        results = [_profile_worker(j) for j in jobs]

    status = np.array([r["status"] for r in results], dtype=np.int8)
    status[no_x] = np.where(status[no_x] == STATUS_OK, STATUS_NO_XBAND, status[no_x])
    dims3 = ("time", "range", "Vel")
    out = xr.Dataset(
        {
            f"n_folds_{channel}": (dims3, np.stack([r["n_folds"] for r in results])),
            "dealias_in_window": (dims3, np.stack([r["in_window"] for r in results]).astype(np.int8)),
            "dealias_ridge1": (("time", "range"), np.stack([r["ridge1"] for r in results])),
            "dealias_ridge2": (("time", "range"), np.stack([r["ridge2"] for r in results])),
            "dealias_fold_shift": (("time", "range"), np.stack([r["fold_shift"] for r in results])),
            "dealias_global_shift": ("time", np.array([r["global_shift"] for r in results], dtype=np.int8)),
            "dealias_status": ("time", status),
            "dealias_xband_offset": ("time", np.array(x_offsets)),
        },
        coords={"time": ds_ka["time"], "range": ds_ka["range"], "Vel": ds_ka["Vel"]},
    )
    out[f"n_folds_{channel}"].attrs = {"long_name": f"Doppler aliasing fold count per bin ({channel} channel)", "units": "1"}
    out["dealias_in_window"].attrs = {
        "long_name": "Bin lies inside the ridge window (1) or outside it, kept at its raw fold (0)", "units": "1"}
    out["dealias_ridge1"].attrs = {"long_name": "Dealiased velocity of ridge 1 (dominant population)", "units": "m s-1"}
    out["dealias_ridge2"].attrs = {"long_name": "Dealiased velocity of ridge 2 (second population)", "units": "m s-1"}
    out["dealias_fold_shift"].attrs = {"long_name": "Whole-row fold shift (local jump correction + global X-band fold)", "units": "1"}
    out["dealias_global_shift"].attrs = {"long_name": "Global fold shift chosen against the X-band ridge", "units": "1"}
    out["dealias_status"].attrs = {
        "long_name": "Ridge dealiasing status",
        "flag_values": "0 1 2 3",
        "flag_meanings": "ok no_sustained_signal_run no_clear_start_gate no_xband_within_tolerance",
    }
    out["dealias_xband_offset"].attrs = {"long_name": "Time offset to the X-band spectrum used", "units": "s"}
    out.attrs["dealias_ridge_params"] = str(asdict(p))
    return out


def dealiased_moments(spec: xr.DataArray, n_folds: np.ndarray, vnyq: float, vel_dim: str = "Vel"):
    """Mean Doppler velocity and spectral width from a spectrum with per-bin folds."""
    vel = spec[vel_dim].values
    s = np.where(np.isfinite(spec.values) & (spec.values > 0), spec.values, 0.0)
    v = vel + n_folds.astype(float) * 2.0 * vnyq
    p0 = s.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = (s * v).sum(axis=-1) / p0
        var = (s * (v - m1[..., None]) ** 2).sum(axis=-1) / p0
    m1 = np.where(p0 > 0, m1, np.nan)
    width = np.where(p0 > 0, np.sqrt(np.clip(var, 0, None)), np.nan)
    dims = tuple(d for d in spec.dims if d != vel_dim)
    coords = {k: c for k, c in spec.coords.items() if vel_dim not in c.dims}
    return xr.DataArray(m1, dims=dims, coords=coords), xr.DataArray(width, dims=dims, coords=coords)

from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr

def ensure_datetime_time(ds: xr.Dataset, time_var: str = "time") -> xr.Dataset:
    """
    Force time coordinate to datetime64 using CF decoding.
    Reads existing units attribute to determine epoch.
    If units attribute is missing, assumes seconds since 1970-01-01 00:00:00 UTC.
    """
    if time_var not in ds.coords:
        return ds

    # already decoded, nothing to do
    if np.issubdtype(ds[time_var].dtype, np.datetime64):
        return ds

    # if units attr is missing, set default
    if "units" not in ds[time_var].attrs:
        ds[time_var].attrs["units"] = "seconds since 1970-01-01 00:00:00 UTC"

    # let xarray handle the epoch from the units attribute
    ds = xr.decode_cf(ds)

    return ds

def lin2db(x: xr.DataArray, floor: float = 1.0e-20) -> xr.DataArray:
    return 10.0 * np.log10(x.clip(min=floor))


def db2lin(x_db: xr.DataArray) -> xr.DataArray:
    return 10.0 ** (x_db / 10.0)


def as_vel_ref(vel_ref) -> np.ndarray | None:
    if vel_ref is None:
        return None
    if isinstance(vel_ref, xr.DataArray):
        return np.asarray(vel_ref.values, dtype=float)
    return np.asarray(vel_ref, dtype=float)


def decode_time_if_needed(ds: xr.Dataset) -> xr.Dataset:
    # Fix attribute name if needed
    if "time" in ds:
        # Already decoded — nothing to do
        if np.issubdtype(ds["time"].dtype, np.datetime64):
            return ds

        # Accept both 'units' and 'Units'
        units = ds["time"].attrs.get("units") or ds["time"].attrs.get("Units") or ds["time"].attrs.get("long_name")
        print('units', units)

        if units is None:
            ds["time"].attrs["units"] = "seconds since 1970-01-01 00:00:00"
        else:
            # Try to parse non-standard units string
            if "since" in units:
                # Try to extract epoch
                import re
                match = re.search(r"since\s+([0-9/:\-\s]+)", units)
                if match:
                    epoch = match.group(1).strip()
                    from datetime import datetime
                    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%d.%m.%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y"):
                        try:
                            dt = datetime.strptime(epoch, fmt)
                            epoch_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                            ds["time"].attrs["units"] = f"seconds since {epoch_str}"
                            break
                        except Exception:
                            continue
                    else:
                        ds["time"].attrs["units"] = "seconds since 1970-01-01 00:00:00"
                else:
                    ds["time"].attrs["units"] = "seconds since 1970-01-01 00:00:00"
            else:
                ds["time"].attrs["units"] = "seconds since 1970-01-01 00:00:00"

        # Remove non-standard 'Units' attribute if present
        if "Units" in ds["time"].attrs:
            del ds["time"].attrs["Units"]

        # Move units/calendar from attrs to encoding so decode_cf doesn't conflict
        time_units = ds["time"].attrs.pop("units", None)
        calendar = ds["time"].attrs.pop("calendar", "standard")
        if time_units is not None:
            ds["time"].encoding["units"] = time_units
            ds["time"].encoding["calendar"] = calendar

    try:
        return xr.decode_cf(ds)
    except Exception:
        return ds

def compute_spectral_moments(
    spec: xr.DataArray,
    vel_dim: str = "Vel",
    prefix: str = "",
    name_m0: str | None = None,
    name_m1: str | None = None,
    name_m2: str | None = None,
    name_ze_db: str | None = None,
) -> xr.Dataset:
    s = spec.where(np.isfinite(spec), 0.0).clip(min=0.0).sortby(vel_dim)
    v = s[vel_dim]

    m0_name = name_m0 or f"{prefix}M0"
    m1_name = name_m1 or f"{prefix}M1"
    m2_name = name_m2 or f"{prefix}M2"

    m0 = s.integrate(vel_dim).rename(m0_name)
    m1 = ((s * v).integrate(vel_dim) / m0.where(m0 > 0)).rename(m1_name)
    var = ((s * (v - m1) ** 2).integrate(vel_dim) / m0.where(m0 > 0)).clip(min=0.0)
    m2 = np.sqrt(var).rename(m2_name)

    out = [m0, m1, m2]
    if name_ze_db is not None:
        ze_db = lin2db(m0).rename(name_ze_db)
        out.append(ze_db)

    return xr.merge(out)


def compute_ldr(
    spec_co: xr.DataArray,
    spec_cx: xr.DataArray,
    vel_dim: str = "Vel",
) -> xr.Dataset:
    """
    Linear depolarization ratio, cross-polar over co-polar:
      - LDR  = 10*log10(Ze_cx / Ze_co)
      - sLDR = 10*log10(sZe_cx / sZe_co)

    Noise-subtracted bins <= 0 are treated as missing (NaN), not as 0, and are
    skipped when integrating.
    """
    co = spec_co.where(np.isfinite(spec_co) & (spec_co > 0))
    cx = spec_cx.where(np.isfinite(spec_cx) & (spec_cx > 0))

    # Integrated powers: NaN-skipping sum * bin width (integrate() would turn the
    # whole spectrum NaN as soon as one bin is NaN). All-NaN spectra stay NaN.
    dv = float(np.abs(np.nanmedian(np.diff(spec_co[vel_dim].values))))
    ze_co = co.sum(vel_dim, skipna=True, min_count=1) * dv
    ze_cx = cx.sum(vel_dim, skipna=True, min_count=1) * dv

    ldr = (10.0 * np.log10(ze_cx / ze_co)).rename("LDR")

    # Spectral ratio
    sldr = (10.0 * np.log10(cx / co)).rename("sLDR")

    return xr.merge([ldr, sldr])


def dealias_mdv_range_continuity(
    mdv: xr.DataArray,
    vnyq: float,
    start: str = "top",
) -> xr.DataArray:
    """
    Dealias MDV along the range dimension using 1D phase unwrapping.

    Walks range gates from one end, correcting each gate by the integer number
    of Nyquist intervals that minimises the velocity step from the previous gate.
    NaN gates are skipped; the last valid gate is kept as the running reference.

    Parameters
    ----------
    mdv : DataArray with dims (time, range) — aliased mean Doppler velocity
    vnyq : Nyquist velocity [m/s]
    start : 'top' to walk from highest range downward (default),
            'bottom' to walk from lowest range upward
    """
    vals = mdv.values.copy()
    out = vals.copy()
    n_time, n_range = vals.shape
    interval = 2.0 * vnyq

    # Vectorised detection: find only timesteps where a profile jump > vnyq exists
    diffs = np.abs(np.diff(vals, axis=1))
    nan_adj = np.isnan(vals[:, :-1]) | np.isnan(vals[:, 1:])
    diffs[nan_adj] = 0.0
    aliased_times = np.where(np.any(diffs > vnyq, axis=1))[0]

    for t in aliased_times:
        if start == "top":
            walk = range(n_range - 2, -1, -1)
            prev = n_range - 1
        else:
            walk = range(1, n_range)
            prev = 0

        for r in walk:
            if np.isnan(out[t, r]):
                continue
            if np.isnan(out[t, prev]):
                prev = r
                continue
            n = round((out[t, prev] - out[t, r]) / interval)
            out[t, r] += n * interval
            prev = r

    result = mdv.copy(data=out)
    result.attrs = {**mdv.attrs, "dealiasing": "range_continuity", "vnyq": vnyq}
    return result


def dealias_mdv_from_reference(
    mdv: xr.DataArray,
    ref: xr.DataArray | str,
    vnyq: float,
    ref_var: str | None = None,
    time_dim: str = "time",
    range_dim: str = "range",
) -> xr.DataArray:
    """
    Dealias MDV using an external reference velocity field.

    For each pixel, adds the integer multiple of 2*vnyq that brings mdv
    closest to the reference. Pixels where the reference is NaN (outside its
    coverage) are left unchanged.

    Parameters
    ----------
    mdv : DataArray with dims (time, range) — aliased mean Doppler velocity
    ref : either a DataArray already on the same grid as mdv, or a path to a
          NetCDF file. When a path is given, ref_var must name the variable to
          use; the function loads it and linearly interpolates to mdv's grid.
    vnyq : Nyquist velocity of mdv [m/s]
    ref_var : variable name to read from the NetCDF file (required if ref is a path)
    time_dim : name of the time dimension in the reference file (default 'time')
    range_dim : name of the range dimension in the reference file (default 'range')
    """
    from pathlib import Path

    if isinstance(ref, (str, Path)):
        if ref_var is None:
            raise ValueError("ref_var must be given when ref is a file path")
        with xr.open_dataset(ref) as ds_ref:
            ref_da = ds_ref[ref_var].load()
    else:
        ref_da = ref

    interp_coords = {}
    if time_dim in ref_da.dims and time_dim in mdv.coords:
        interp_coords[time_dim] = mdv[time_dim]
    if range_dim in ref_da.dims and range_dim in mdv.coords:
        interp_coords[range_dim] = mdv[range_dim]
    if interp_coords:
        ref_da = ref_da.interp(interp_coords, method="linear")

    n_folds = np.where(
        np.isnan(ref_da.values),
        0,
        np.round((ref_da.values - mdv.values) / (2.0 * vnyq)),
    )
    dealiased = mdv + n_folds * 2.0 * vnyq
    dealiased.attrs = {**mdv.attrs, "dealiasing": "external_reference", "vnyq": vnyq}
    return dealiased


def _nearest_index(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """For each value in `target`, the index of the nearest value in sorted `source`."""
    idx = np.searchsorted(source, target)
    idx = np.clip(idx, 1, len(source) - 1)
    left = idx - 1
    choose_left = np.abs(source[left] - target) <= np.abs(source[idx] - target)
    return np.where(choose_left, left, idx)


def dealias_ka_with_phase_unwrap(
    mdv: xr.DataArray,
    ref: xr.DataArray,
    vnyq: float,
    trust_min_range: float = 2000.0,
    time_dim: str = "time",
    range_dim: str = "range",
) -> xr.DataArray:
    """
    Dealias a Ka-band MDV profile by 2D phase unwrapping along range, using a co-located
    wide-Nyquist radar's MDV (e.g. X-band) only to anchor the absolute fold of each
    unwrapped, internally-consistent segment -- not as a bin-by-bin or gate-by-gate match.

    MDV is periodic with period 2*vnyq, exactly like a wrapped interferometric phase is
    periodic with period 2*pi -- "which fold does each gate belong to" is the textbook
    phase-unwrapping problem, with mature, well-tested algorithms built for exactly this.
    Earlier attempts at this hand-rolled the equivalent of an unwrapper from scratch (peak-
    finding + line-tracing across range, modeled on the MRR backend's approach) and kept
    breaking on real convective spectra: broad, noisy echoes fragment into many spurious
    sub-peaks, isolated fragments guess their own fold independently of their neighbors,
    and a feature whose true velocity crosses a fold boundary as it varies with height
    (very common here, given velocities up to several times vnyq) needs its fold decision
    made *consistently* across that crossing, not fragment-by-fragment. This instead
    converts MDV to a phase-like quantity and unwraps each profile (independently per time
    step -- range continuity only, no dependence on neighboring time steps) with
    `numpy.unwrap`, applied separately to each contiguous run of valid gates (so a real
    data gap is never bridged over, unlike treating missing data as zero would). Unwrapping
    is the "glue the part after a gap to the part before it" reconnection as a single,
    well-tested library operation, rather than something approximated gate-by-gate.

    Unwrapping only fixes *relative* consistency within each contiguous valid segment of a
    profile, not the absolute branch -- for that, each segment is anchored once using
    `ref`, preferring reference values at range >= trust_min_range when the segment has any
    signal there at all (a very high Ze needed to make the reference unreliable at low
    height implies a taller system feeding it, so a trusted anchor above trust_min_range
    should exist whenever it's needed), otherwise using whatever finite reference values
    the segment has.

    Parameters
    ----------
    mdv : (time, range) raw (aliased) mean Doppler velocity
    ref : (time, range) reference MDV (e.g. X-band), on its own time/range grid -- matched
        to `mdv` by nearest neighbor internally. NaN where unavailable.
    vnyq : Nyquist velocity of `mdv` [m/s]
    trust_min_range : minimum range [m] at which `ref` is trusted preferentially (see above)

    Returns
    -------
    Dealiased MDV, same dims/coords as `mdv`.
    """
    import scipy.ndimage

    mdv = mdv.transpose(time_dim, range_dim)
    m_time = mdv[time_dim].values
    m_range = mdv[range_dim].values.astype(np.float64)
    n_time, n_range = mdv.sizes[time_dim], mdv.sizes[range_dim]

    def _to_epoch(t):
        return t.astype("datetime64[s]").astype(np.float64) if np.issubdtype(t.dtype, np.datetime64) else t.astype(np.float64)

    ref = ref.transpose(time_dim, range_dim)
    ref_ti_idx = _nearest_index(_to_epoch(ref[time_dim].values), _to_epoch(m_time))
    ref_ri_idx = _nearest_index(ref[range_dim].values.astype(np.float64), m_range)
    ref_on_grid = ref.values[ref_ti_idx][:, ref_ri_idx]  # (n_time, n_range), on mdv's own grid

    interval = 2.0 * vnyq
    trusted = m_range >= trust_min_range

    mdv_raw = mdv.values.astype(np.float64)
    out = np.full((n_time, n_range), np.nan, dtype=np.float64)

    for it in range(n_time):
        v_row = np.isfinite(mdv_raw[it])
        if not v_row.any():
            continue

        # `unwrap_phase` doesn't support masked 1D input (only unmasked 1D, or 2D/3D
        # masked) -- this is per-profile (range-only), so unwrap each contiguous valid
        # run separately with `numpy.unwrap` on its own plain (unmasked) sub-array
        # instead, which does the same 1D unwrapping and sidesteps that limitation.
        # Two separate runs (split by a genuine data gap) aren't related to each other
        # by this, on purpose -- each is anchored independently below.
        labels, n_labels = scipy.ndimage.label(v_row)
        ref_row = ref_on_grid[it]
        for lab in range(1, n_labels + 1):
            seg = labels == lab
            phase_seg = mdv_raw[it, seg] * np.pi / vnyq
            mdv_unwrapped_seg = np.unwrap(phase_seg) * vnyq / np.pi

            ref_here = ref_row[seg]
            seg_trusted = trusted[seg]
            use_mask = (seg_trusted if seg_trusted.any() else np.ones(seg.sum(), dtype=bool)) & np.isfinite(ref_here)
            if not use_mask.any():
                use_mask = np.isfinite(ref_here)
            if use_mask.any():
                offset = np.mean(ref_here[use_mask] - mdv_unwrapped_seg[use_mask])
                best_n = round(offset / interval)
            else:
                best_n = 0
            out[it, seg] = mdv_unwrapped_seg + best_n * interval

    result = mdv.copy(data=out)
    result.attrs = {**mdv.attrs, "dealiasing": "phase_unwrap_range_with_reference_anchor", "vnyq": vnyq}
    return result


def _find_wrap_bins(above: np.ndarray, max_gap_bins: int) -> int:
    """
    For one gate's above-threshold boolean spectrum (already known to touch both the
    +vnyq and -vnyq edges -- see caller), returns `s`: the number of trailing bins
    (nearest +vnyq) that make up the contiguous run wrapped from the low-velocity edge.

    Ported from a validated Doppler-spectrum dealiasing routine (tripex_pol_scan's
    `dealiazeOneHeight`: find the length of the run of real signal touching the +vnyq
    edge by walking backward from the last bin until a gap is hit; that length is the
    number of bins to roll). Gaps of `max_gap_bins` or fewer are bridged first (treated
    as still part of the same run) so an isolated noise dip doesn't cut the run short --
    the original tolerates this via a per-bin lookahead patch; here the same intent is
    applied more simply by patching whole short gaps before the walk.
    """
    nfft = above.size
    patched = above.copy()
    gap_start = None
    for i in range(nfft):
        if not patched[i]:
            if gap_start is None:
                gap_start = i
        elif gap_start is not None:
            if i - gap_start <= max_gap_bins:
                patched[gap_start:i] = True
            gap_start = None

    if patched.all():
        return 0  # no gap anywhere even after bridging -> nothing to cut at, leave unrolled

    s = 0
    q = nfft - 1
    while patched[q]:
        s += 1
        q -= 1
    return s + 1  # ported convention: roll by (run length + 1), see dealiazeOneHeight


def _step1_glue_gate(above_row: np.ndarray, edge_bins: int, max_gap_bins: int) -> int:
    """
    Step 1's per-gate straddle check, factored out so it can be reused outside
    `dealias_ka_spectrum_with_range_reference` (see `dealias_ka_spectrum_peaktrack`,
    which needs the same glue applied before peak-finding -- a population straddling
    +/-vnyq otherwise reads as two edge-hugging fragments instead of one peak).

    Does this gate's above-threshold spectrum touch both the -vnyq and +vnyq edges?
    If so, find how many trailing bins (nearest +vnyq) form the run wrapped from the
    low-velocity edge (_find_wrap_bins) -- that's `s`, the number of bins to roll so
    the gate's spectrum becomes one contiguous piece on a local (glue-corrected)
    velocity axis. Returns 0 for a non-straddling gate or a straddling gate where no
    clean wrapped run is found.
    """
    if above_row[-edge_bins:].any() and above_row[:edge_bins].any():
        return _find_wrap_bins(above_row, max_gap_bins)
    return 0


def _bool_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) inclusive bin-index pairs for each contiguous True run in `mask`."""
    runs = []
    start = None
    for i in range(mask.size):
        if mask[i] and start is None:
            start = i
        elif not mask[i] and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, mask.size - 1))
    return runs


def _extra_outer_run_mask(local_vel: np.ndarray, above_row: np.ndarray, baseline: float,
                           side: str, min_run_bins: int = 20,
                           merge_gap_tol: float = 0.0) -> np.ndarray | None:
    """
    Find the outermost genuinely separate (gap-bounded) run of above-threshold bins
    in this gate, on the given `side` ("right" or "left") -- i.e. starting from the
    edge and walking inward, everything up to the first real gap. Returns a boolean
    mask selecting that run's bins, or None if there's nothing separate out there
    (only one population present after filtering/merging, see below).

    Deliberately does NOT try to identify which run is "the normal population" --
    an earlier version picked whichever run was numerically closest to `baseline`
    and treated everything past it as the candidate, but `baseline` is a frozen
    value from several gates back and can easily end up sitting closer to the
    leftover piece than to the true main population (e.g. gate 168 in the
    2025-06-04 18:00:40 case: main population edge at -4.37, leftover piece edge at
    -1.37, baseline at -2.71 -- numerically closer to the leftover piece, so it got
    picked as "main" and the real leftover piece was never found). The outermost
    piece is simply the leftover piece by construction: if the gate's own genuine
    edge (`right_run`/`left_run`) didn't look anomalous relative to baseline, Step
    1b would never have called this function in the first place (see the entry/
    continuation check in the caller) -- so whatever sits furthest out here already
    is the anomaly, without needing a separate "which one is normal" decision.
    `baseline` is kept as a parameter for interface symmetry with the caller but is
    no longer used inside this function.

    Runs narrower than `min_run_bins` are dropped first -- a genuine physical
    population should span several bins, so a narrow blip right at the noise floor
    is noise-floor clutter, not a separate echo, and would otherwise get treated as
    the outermost "population" purely for sitting a bin or two further out.

    `_bool_runs(above_row)` finds gaps in RAW bin-index order, which is blind to any
    shift Step 1 already applied -- a piece Step 1 wrapped from the +vnyq edge down to
    the local axis lands, by construction, exactly adjacent to whatever already sits at
    -vnyq (that's what wrapping means), even though it's nowhere near it in raw bin
    index. Two runs are merged here if their LOCAL-velocity ranges are contiguous
    (within `merge_gap_tol`), so an already-fully-resolved gate (one continuous
    population after Step 1, e.g. a straddling echo with no separate leftover piece)
    doesn't get its already-correct glue treated as a fresh candidate to shift again.

    This is used by Step 1b to decide *which* bins are the leftover unglued piece,
    using the real noise-floor gap between populations rather than a velocity offset
    guessed from vnyq.
    """
    runs = [(s, e) for s, e in _bool_runs(above_row) if (e - s + 1) >= min_run_bins]
    if len(runs) < 2:
        return None
    run_info = []
    for s, e in runs:
        lo, hi = local_vel[s], local_vel[e]
        if lo > hi:
            lo, hi = hi, lo
        run_info.append([[(s, e)], lo, hi])

    # Merge runs that are contiguous on the LOCAL (post-Step-1-shift) axis -- they're
    # already one physical population, whatever their raw bin-index separation.
    run_info.sort(key=lambda r: r[1])
    merged = [run_info[0]]
    for sub_ranges, lo, hi in run_info[1:]:
        if lo - merged[-1][2] <= merge_gap_tol:
            merged[-1][0].extend(sub_ranges)
            merged[-1][2] = max(merged[-1][2], hi)
        else:
            merged.append([sub_ranges, lo, hi])
    if len(merged) < 2:
        return None

    # The outermost remaining population, on the requested side, is the leftover piece.
    if side == "right":
        chosen = max(merged, key=lambda r: r[2])
    else:
        chosen = min(merged, key=lambda r: r[1])
    mask = np.zeros_like(above_row)
    for s, e in chosen[0]:
        mask[s:e + 1] = True
    return mask


def _xband_dip_split_mask(w_row: np.ndarray, local_vel: np.ndarray, above_row: np.ndarray,
                           ref_val: float, edge_baseline: float, side: str,
                           interval: float) -> np.ndarray | None:
    """
    Fallback for gates where the Ka spectrum's own structure doesn't give
    `_extra_outer_run_mask` anything to work with -- e.g. a real second population
    sitting right next to the main one, separated only by a shallow dip that doesn't
    fully drop below the detection threshold (see the 18:00:40 gate-170 case this was
    built for: main population and leftover piece are one contiguous raw run, so
    there's no below-threshold gap to split on at all).

    Unlike a velocity-offset cutoff (tried and rejected -- there's no physical reason
    a split point should be expressed as a distance from vnyq_ka; vnyq_ka is an
    instrument property, the spectrum's shape is a physical one, and the two aren't
    related), this finds the split location from the gate's own actual shape.

    It does NOT just take the global minimum of the power trace across the run --
    tried that and it systematically picks a point right next to one of the run's own
    edges instead of a genuine internal valley: any single coherent population's power
    naturally tapers off toward both ends of its above-threshold span (that's just
    what the tail of a peak looks like approaching the noise floor), so the bare
    minimum of the whole span is almost always an edge-taper artifact, not a real
    second population (confirmed at range gate 172 in the 18:00:40 case: the "dip" it
    found was 6 bins from the edge of a 202-bin run that should have moved as a
    whole). Instead this requires a *prominent* local minimum -- one where the power
    rises back up by a real margin (`min_prominence_db`) on both sides -- which is
    what actually distinguishes a genuine two-population dip (like the one visible by
    eye in the 18:00:40 gate-170 spectrum) from ordinary edge roll-off or noise. If no
    sufficiently prominent dip exists, this returns None: that's the correct answer
    when the run is genuinely one coherent population, not two.

    It then only commits to that split if it actually helps, checked two ways:
      1. Preferred: if a trusted X-band reference (`ref_val`) is available, accept the
         split if it brings the gate's own reflectivity-weighted mean velocity closer
         to that independent, trusted measurement than leaving the gate alone.
      2. Fallback: if the X-band test doesn't show an improvement (or no trusted
         reference exists at this gate), fall back to range continuity -- accept the
         split if it brings this gate's own edge (right_run/left_run, recomputed after
         the hypothetical shift) closer to `edge_baseline` (the established, trusted
         trend from already-resolved gates) than the unshifted edge is. This is weaker
         evidence than an independent measurement, but still real information Ka's own
         spectrum shape can't provide by itself, and helps gates where X-band isn't
         available or isn't decisive on its own.
    Returns the split mask if either test passes, otherwise None (no split).
    """
    if not above_row.any():
        return None
    # Restrict the dip search to a single raw contiguous run -- specifically the
    # widest one. Searching the full above-threshold span (all the way from the
    # overall min to overall max bin) is wrong when the gate already has more than
    # one genuinely separate raw run (e.g. a piece Step 1 already wrapped correctly,
    # separated from the main blob by a real, large gap): the deepest point in that
    # real gap would get picked as "the dip", and the already-correctly-placed piece
    # would get shifted a second time. The widest single run is where an unresolved
    # *shallow* dip (not a real gap) can actually be hiding.
    raw_runs = _bool_runs(above_row)
    if not raw_runs:
        return None
    lo_bin, hi_bin = max(raw_runs, key=lambda r: r[1] - r[0])
    if hi_bin - lo_bin < 10:
        return None  # too narrow a span to contain a meaningful internal dip

    # Lightly smooth the power trace (in dB, matching how we've been visually judging
    # these dips throughout this session) so a single noisy bin doesn't get picked as
    # "the dip" -- we want the real valley shape.
    from scipy.signal import find_peaks
    k = 5
    span_db = 10.0 * np.log10(np.maximum(w_row[lo_bin:hi_bin + 1], 1e-12))
    smoothed = np.convolve(span_db, np.ones(k) / k, mode="same")
    margin = max(k, 3)
    if smoothed.size <= 2 * margin:
        return None
    interior = smoothed[margin:-margin]

    min_prominence_db = 3.0
    valley_idx, props = find_peaks(-interior, prominence=min_prominence_db)
    if valley_idx.size == 0:
        return None  # no genuine internal dip -- one coherent population, nothing to split
    best = int(valley_idx[np.argmax(props["prominences"])])
    split_rel = best + margin
    split_bin = lo_bin + split_rel

    # Candidate is restricted to this single run's own bin range [lo_bin, hi_bin] --
    # not the full above_row -- so a separate raw run elsewhere in the gate (e.g. a
    # piece Step 1 already wrapped correctly) never gets swept in and re-shifted.
    candidate = np.zeros_like(above_row)
    if side == "right":
        candidate[split_bin:hi_bin + 1] = above_row[split_bin:hi_bin + 1]
        shift = -interval
    else:
        candidate[lo_bin:split_bin] = above_row[lo_bin:split_bin]
        shift = interval
    if not candidate.any():
        return None

    v_shifted = local_vel.copy()
    v_shifted[candidate] += shift

    # Test 1: does the split bring the gate's own weighted-mean velocity closer to
    # the trusted X-band reference?
    if np.isfinite(ref_val):
        w_all = w_row[above_row]
        wsum = w_all.sum()
        if wsum > 0:
            baseline_mdv = float((w_all * local_vel[above_row]).sum() / wsum)
            shifted_mdv = float((w_all * v_shifted[above_row]).sum() / wsum)
            if abs(shifted_mdv - ref_val) < abs(baseline_mdv - ref_val):
                return candidate

    # Test 2 (fallback): does the split bring this gate's own edge closer to the
    # established baseline trend from below?
    if np.isfinite(edge_baseline):
        if side == "right":
            edge_before = float(local_vel[above_row].max())
            edge_after  = float(v_shifted[above_row].max())
        else:
            edge_before = float(local_vel[above_row].min())
            edge_after  = float(v_shifted[above_row].min())
        if abs(edge_after - edge_baseline) < abs(edge_before - edge_baseline):
            return candidate

    return None


def dealias_ka_spectrum_with_range_reference(
    spec_ka: xr.DataArray,
    noise_floor_ka: xr.DataArray,
    ref_mdv: xr.DataArray,
    vnyq_ka: float,
    noise_mult: float = 0.0,
    edge_bins: int = 3,
    max_gap_bins: int = 10,
    trust_min_range: float = 2000.0,
    edge_jump_mad_mult: float = 4.0,
    min_run_bins: int = 20,
    step1b_after_step2: bool = False,
    vel_dim: str = "Vel",
    time_dim: str = "time",
    range_dim: str = "range",
    extra_channels: dict[str, xr.DataArray] | None = None,
    debug_plot_dir: str | None = None,
    debug_it: int = 0,
    debug_range_km: tuple[float, float] = (2.0, 3.0),
) -> xr.Dataset:
    """
    Dealias a Ka-band Doppler spectrum bin-by-bin: glue each gate's spectrum back
    together where it straddles ±vnyq, then resolve the (now per-gate, not per-bin)
    remaining ambiguity -- which absolute Nyquist branch the glued spectrum sits on --
    by unwrapping along range and anchoring to a co-located wide-Nyquist radar's MDV.

    This is the spectral counterpart to `dealias_ka_with_phase_unwrap` (which only
    corrects the first-moment MDV): the previous per-bin attempts here either matched
    each bin independently against a reference spectrum (breaks down when the reference
    itself is unusable at some ranges/velocities, or a broad echo's middle sits too far
    from the reference's one peak) or hand-rolled peak-finding and cross-range line-
    tracing modeled on the MRR backend (fragments broad, noisy echoes into many spurious
    pieces). This instead ports a validated two-step approach:

      1. Per gate, if the spectrum has real signal within `edge_bins` of *both* ±vnyq
         (the straddling signature), find how many trailing bins (nearest +vnyq) form
         the contiguous run wrapped from the low-velocity edge -- gaps of up to
         `max_gap_bins` are bridged first so an isolated noise dip doesn't cut the run
         short (see `_find_wrap_bins`, ported from a validated routine in
         tripex_pol_scan's `processData.py::dealiazeOneHeight`). Those wrapped bins get
         `2*vnyq` subtracted from their apparent velocity, gluing the gate's spectrum
         into one contiguous piece on a "local" velocity axis. A gate that doesn't
         straddle keeps its ordinary apparent velocity unchanged (local axis = apparent
         axis); either way every gate ends up with one local axis and a corresponding
         weighted-centroid scalar.
      2. That per-gate scalar is exactly the same kind of quantity `MDV` is (a
         reflectivity-weighted mean velocity, just corrected for within-gate straddling
         first) -- so the absolute branch is resolved by feeding it through the same
         range-unwrap-and-anchor logic as `dealias_ka_with_phase_unwrap`: unwrap along
         range per contiguous valid run, then anchor each run once using `ref_mdv`,
         preferring range >= trust_min_range when the run has signal there at all (see
         that function's docstring for the reasoning). The resulting per-gate fold is
         then applied to every bin of that gate (both the glued and unglued ones).

    Parameters
    ----------
    spec_ka : (time, range, Vel) noise-subtracted, non-negative Ka spectrum (linear units)
    noise_floor_ka : (time, range) instrument-calibrated noise floor, same linear units
    ref_mdv : (time, range) reference MDV (e.g. X-band VELg or MDV_Co), own time/range
        grid -- matched to spec_ka by nearest neighbor internally. NaN where unavailable.
    vnyq_ka : Nyquist velocity of spec_ka [m/s]
    noise_mult : detection threshold = noise_mult * noise_floor_ka
    edge_bins : a gate straddles if real signal is present within this many bins of
        *both* ±vnyq (not necessarily the exact edge bin)
    max_gap_bins : gaps this size or smaller are bridged before walking back to find the
        wrapped run's length, so an isolated noise dip doesn't cut it short
    trust_min_range : minimum range [m] at which ref_mdv is trusted preferentially (see
        `dealias_ka_with_phase_unwrap`)
    extra_channels : additional co-located spectra (e.g. {"Cx": spec_ka_cross}) that
        reuse the fold decisions made from spec_ka rather than being independently glued
    step1b_after_step2 : experimental reordering (default False = original, validated
        order Step1 -> Step1b -> Step2). When True: run Step 2's range-unwrap + X-band
        anchor once right after Step 1 (on the unmodified Step-1 centroid) to get each
        gate's absolute fold, then run Step 1b's same edge-jump hysteresis on the now-
        absolute (anchored) edges instead of the local ones, so its jump-detection
        compares against an already-anchored trend and its X-band fallback is always
        directly comparable rather than a last-resort special case. The absolute fold
        from that first Step-2 pass is reused as-is afterward (not recomputed), since
        Step 1b only ever adjusts within-gate local_shift, never the whole-gate branch.

    Returns
    -------
    Dataset with MDV, n_folds (per-bin, dims of spec_ka, int8), n_valid (bool)
    """
    import scipy.ndimage

    spec_ka = spec_ka.transpose(time_dim, range_dim, vel_dim)
    vel = spec_ka[vel_dim].values.astype(np.float64)
    nfft = vel.size
    ka_time = spec_ka[time_dim].values
    ka_range = spec_ka[range_dim].values.astype(np.float64)
    n_time, n_range = spec_ka.sizes[time_dim], spec_ka.sizes[range_dim]

    Ka_power = np.nan_to_num(spec_ka.values, nan=0.0).astype(np.float64)
    noise = noise_floor_ka.transpose(time_dim, range_dim).values.astype(np.float64)

    def _to_epoch(t):
        return t.astype("datetime64[s]").astype(np.float64) if np.issubdtype(t.dtype, np.datetime64) else t.astype(np.float64)

    ref_mdv = ref_mdv.transpose(time_dim, range_dim)
    ref_ti_idx = _nearest_index(_to_epoch(ref_mdv[time_dim].values), _to_epoch(ka_time))
    ref_ri_idx = _nearest_index(ref_mdv[range_dim].values.astype(np.float64), ka_range)
    ref_on_ka = ref_mdv.values[ref_ti_idx][:, ref_ri_idx]  # (n_time, n_range)

    # The spectrum wraps with period 2*vnyq: a velocity shifted by exactly 2*vnyq looks
    # identical to the radar.  "interval" is used throughout as the size of one full fold.
    interval = 2.0 * vnyq_ka
    # Tolerance (in m/s) for treating two runs as already-contiguous on the local axis
    # (see _extra_outer_run_mask) -- a couple of bin-widths, to allow for the shifted
    # piece landing right at (not necessarily bin-exact) the other piece's edge.
    merge_gap_tol = 2.0 * interval / nfft

    # At low range the X-band reference MDV can be contaminated (strong Ze causes spectral
    # leakage in X, making its VELg unreliable below ~2 km).  Gates at or above
    # trust_min_range are flagged as "trusted" and are preferred when anchoring; if no
    # trusted gates have a finite reference the code falls back to whatever reference is
    # available anywhere in the segment.
    trusted = ka_range >= trust_min_range

    extra_channels = extra_channels or {}
    extra_power = {
        k: np.nan_to_num(v.transpose(time_dim, range_dim, vel_dim).values, nan=0.0)
        for k, v in extra_channels.items()
    }

    # n_folds_out[time, range, bin]: integer fold count for every spectrum bin.
    #   0  → bin stays at its apparent velocity (no correction needed)
    #  +1  → bin's true velocity is apparent + 2*vnyq  (signal was folded down by radar)
    #  -1  → bin's true velocity is apparent - 2*vnyq  (signal was folded up by radar)
    # This is what build_extended / dealias_spectrum use to reconstruct the dealiased spectrum.
    n_folds_out = np.zeros((n_time, n_range, nfft), dtype=np.int8)
    # n_valid[time, range]: True if this gate has any signal above the noise floor.
    n_valid = np.zeros((n_time, n_range), dtype=bool)
    # MDV[time, range]: reflectivity-weighted mean Doppler velocity after full dealiasing.
    MDV = {ch: np.full((n_time, n_range), np.nan, dtype=np.float32) for ch in (["main"] + list(extra_channels))}
    # Diagnostic arrays: spectral edge positions before and after Step-1b correction.
    dbg_vel_right_raw    = np.full((n_time, n_range), np.nan, dtype=np.float32)
    dbg_vel_right_smooth = np.full((n_time, n_range), np.nan, dtype=np.float32)
    dbg_vel_left_raw     = np.full((n_time, n_range), np.nan, dtype=np.float32)
    dbg_vel_left_smooth  = np.full((n_time, n_range), np.nan, dtype=np.float32)
    # gate_fold per gate: the Step-2 whole-gate fold correction (integer).
    dbg_gate_fold        = np.zeros((n_time, n_range), dtype=np.int8)
    # Step-1b derivative: the ACTUAL gate-to-gate difference of right_run (and left_run)
    # as seen by the bottom-to-top pass of the derivative loop.  Because right_run is
    # updated in place when a gate is corrected, this differs from np.diff(dbg_vel_right_raw):
    # a corrected gate's updated edge becomes the reference for the gate above it.
    # NaN where no valid neighbour exists.
    dbg_deriv_right = np.full((n_time, n_range), np.nan, dtype=np.float32)
    dbg_deriv_left  = np.full((n_time, n_range), np.nan, dtype=np.float32)
    # Whether the bottom-to-top pass triggered a correction at each gate (1=yes, 0=no).
    dbg_deriv_triggered = np.zeros((n_time, n_range), dtype=np.int8)
    # The frozen reference baseline actually used by the bottom-to-top pass at each
    # gate (see Step 1b hysteresis) -- this is the velocity the entry/continuation
    # criterion is checked against, i.e. "is there signal past baseline (+/-) threshold".
    dbg_baseline_right = np.full((n_time, n_range), np.nan, dtype=np.float32)
    dbg_baseline_left  = np.full((n_time, n_range), np.nan, dtype=np.float32)
    # Per-time-step data-driven jump threshold actually used (see Step 1b): a multiple
    # of the profile's own robust gate-to-gate edge variability, not a fraction of vnyq.
    dbg_right_jump_threshold = np.full(n_time, np.nan, dtype=np.float32)
    dbg_left_jump_threshold  = np.full(n_time, np.nan, dtype=np.float32)

    for it in range(n_time):
        # w[range, bin]: raw Ka spectrum power for this time step (NaN → 0 so arithmetic is safe).
        w = Ka_power[it]
        # above[range, bin]: True wherever sZeCo > 0.  sZeCo is stored as noise-subtracted
        # non-negative power, so exactly-zero bins are below or at the noise floor.  Using
        # > 0 (rather than a noise_mult × noise_floor threshold) avoids excluding real weak
        # signal near the Nyquist edges; near-zero noise bins contribute negligibly to the
        # power-weighted centroid.  noise_mult is kept in the signature for future use.
        above = w > 0

        # -----------------------------------------------------------------------
        # STEP 1 — Per-gate glue: detect straddling and compute a single centroid
        # -----------------------------------------------------------------------
        # A "centroid" for a gate is the reflectivity-weighted mean velocity of all
        # above-threshold bins in that gate, computed on a "local" velocity axis that
        # has already corrected for within-gate wrapping.  It is a single number (m/s)
        # summarising where the bulk of the echo in that gate actually is, after fixing
        # the within-gate fold.  Step 2 then works with these scalars (one per gate)
        # rather than with the full spectrum, which makes the range-continuity logic
        # much simpler.
        centroid = np.full(n_range, np.nan)
        # vel_left_arr / vel_right_arr: min/max velocity of above-threshold bins at each
        # gate AFTER the initial local_shift glue.  Used in the median-filter correction
        # block below to detect gates where a middle piece was left unshifted.
        vel_left_arr = np.full(n_range, np.nan)
        vel_right_arr = np.full(n_range, np.nan)
        # local_shift[range, bin]: velocity offset to apply to each bin before computing
        # the centroid.  For a non-straddling gate every entry is 0 (apparent velocity =
        # local velocity).  For a straddling gate the s trailing bins (near +vnyq) get
        # -interval = -2*vnyq, which moves them from the +vnyq edge to just below the
        # main piece of the spectrum so the two pieces sit adjacent on the local axis.
        # This is the "glue" — conceptually identical to xarray.roll() but expressed as
        # a velocity offset per bin rather than a physical shift of the array.
        local_shift = np.zeros((n_range, nfft))
        for ir in range(n_range):
            above_row = above[ir]
            # No signal at all → skip, centroid stays NaN.
            if not above_row.any():
                continue
            # Straddling check + wrapped-run length (see _step1_glue_gate): does real
            # signal touch BOTH the -vnyq edge (first edge_bins bins) AND the +vnyq edge
            # (last edge_bins bins)?  If so, the physical echo is split across the
            # folding boundary and must be glued.
            s = _step1_glue_gate(above_row, edge_bins, max_gap_bins)
            if s > 0:
                # Shift those s trailing bins down by one full period so they appear
                # immediately below the negative piece on the local velocity axis.
                local_shift[ir, nfft - s:] = -interval
            # local_vel[bin]: apparent velocity + the per-bin correction just computed.
            # For non-straddling gates this equals vel; for straddling gates the trailing
            # s bins are placed at vel - 2*vnyq (their true local position).
            local_vel = vel + local_shift[ir]
            w_row = w[ir]
            wsum = w_row[above_row].sum()
            if wsum > 0:
                # Reflectivity-weighted mean of local_vel over all above-threshold bins.
                # This is the centroid: one scalar representing the gate's mean velocity
                # on the local (within-gate-corrected) axis.  It is still potentially
                # offset by a whole multiple of 2*vnyq from the true absolute velocity —
                # that absolute ambiguity is resolved in Step 2.
                centroid[ir] = float((w_row[above_row] * local_vel[above_row]).sum() / wsum)
            # Record min/max of the local (glue-corrected) velocity for above-threshold
            # bins.  These are the spectral edges used in the median-filter step below.
            valid_local = local_vel[above_row]
            vel_left_arr[ir] = valid_local.min()
            vel_right_arr[ir] = valid_local.max()

        # -----------------------------------------------------------------------
        # STEP 1b — Edge-jump correction: catch missed middle pieces
        # -----------------------------------------------------------------------
        # vel_right_arr / vel_left_arr track the rightmost/leftmost above-threshold
        # velocity per gate on the local (glue-corrected) axis.  For a smoothly
        # varying echo these edges change gradually with range.  A gate where a
        # wrapped piece was missed by Step 1 shows up as a sudden jump: the right
        # edge leaps up (piece still at apparent +vnyq-side position) or the left
        # edge drops down (piece still at apparent -vnyq-side position).
        #
        # Detection threshold: a multiple (edge_jump_mad_mult) of the profile's own
        # robust gate-to-gate edge variability (median absolute deviation of
        # consecutive differences), rather than a fixed fraction of vnyq_ka -- the
        # size of a genuine missed-fold jump has no physical reason to scale with
        # the radar's Nyquist velocity, so the threshold is calibrated from this
        # profile's own "normal" behaviour instead.
        #
        # Once a gate is flagged, the bins to shift are found from the actual run
        # structure of the gate (a real noise-floor gap separates the correctly
        # glued piece from the leftover piece -- see _extra_outer_run_mask), not
        # from a velocity offset guessed from vnyq.
        #
        # Hysteresis: once triggered, the reference baseline freezes at the last
        # normal gate's edge and keeps being used (instead of re-deriving a fresh,
        # noisy reference from each newly-corrected neighbour) for as long as a
        # separate outer run keeps showing up -- so one persistent blob doesn't
        # have to independently re-clear the entry threshold at every gate.
        #
        # Two passes (bottom-to-top, then top-to-bottom) so blobs above OR below
        # a correctly-processed region are both caught.

        dbg_vel_right_raw[it] = vel_right_arr
        dbg_vel_left_raw[it]  = vel_left_arr

        ref_row = ref_on_ka[it]

        def _robust_jump_scale(edge_arr: np.ndarray) -> float:
            d = np.diff(edge_arr[np.isfinite(edge_arr)])
            if d.size == 0:
                return np.nan
            return float(np.median(np.abs(d - np.median(d))))

        right_scale = _robust_jump_scale(vel_right_arr)
        left_scale  = _robust_jump_scale(vel_left_arr)
        # Fallback/floor: if too few valid gates to estimate a scale, or the profile
        # is suspiciously perfectly smooth (scale ~0), fall back to a small fraction
        # of vnyq_ka so the threshold doesn't collapse to ~0 and trigger on noise.
        floor = 0.05 * vnyq_ka
        right_scale = right_scale if (np.isfinite(right_scale) and right_scale > floor) else floor
        left_scale  = left_scale  if (np.isfinite(left_scale)  and left_scale  > floor) else floor

        right_threshold = edge_jump_mad_mult * right_scale
        left_threshold  = edge_jump_mad_mult * left_scale
        dbg_right_jump_threshold[it] = right_threshold
        dbg_left_jump_threshold[it]  = left_threshold

        def _compute_gate_fold(centroid_arr: np.ndarray) -> np.ndarray:
            # Step 2's range-unwrap + X-band anchor, factored out so it can be called
            # either after Step 1b (default order) or right after Step 1 (see
            # step1b_after_step2 docstring above).
            v_row_local = np.isfinite(centroid_arr)
            gate_fold_local = np.zeros(n_range, dtype=np.int8)
            if not v_row_local.any():
                return gate_fold_local
            labels_local, n_labels_local = scipy.ndimage.label(v_row_local)
            for lab in range(1, n_labels_local + 1):
                seg = labels_local == lab
                phase_seg = centroid_arr[seg] * np.pi / vnyq_ka
                unwrapped_seg = np.unwrap(phase_seg) * vnyq_ka / np.pi
                ref_here = ref_row[seg]
                seg_trusted = trusted[seg]
                use_mask = (seg_trusted if seg_trusted.any() else np.ones(seg.sum(), dtype=bool)) & np.isfinite(ref_here)
                if not use_mask.any():
                    use_mask = np.isfinite(ref_here)
                if use_mask.any():
                    offset = np.mean(ref_here[use_mask] - unwrapped_seg[use_mask])
                    best_n = round(offset / interval)
                else:
                    best_n = 0
                unwrap_per_gate = np.round((unwrapped_seg - centroid_arr[seg]) / interval).astype(np.int8)
                gate_fold_local[seg] = (best_n + unwrap_per_gate).astype(np.int8)
            return gate_fold_local

        gate_fold_pre = None
        if step1b_after_step2:
            # Anchor first (using the unmodified Step-1 centroid), then let Step 1b's
            # hysteresis operate on the now-absolute edges.
            gate_fold_pre = _compute_gate_fold(centroid)
            right_run = vel_right_arr + gate_fold_pre.astype(np.float64) * interval
            left_run  = vel_left_arr  + gate_fold_pre.astype(np.float64) * interval
        else:
            right_run = vel_right_arr.copy()
            left_run  = vel_left_arr.copy()

        for pass_dir in (+1,):  # single-pass (bottom-to-top only) -- see docstring/notes
            # pass_dir=+1: ir increases (bottom-to-top)
            # pass_dir=-1: ir decreases (top-to-bottom)
            gate_iter = range(n_range) if pass_dir == +1 else range(n_range - 1, -1, -1)
            baseline_right = np.nan
            baseline_left  = np.nan
            active_right   = False
            active_left    = False
            for ir in gate_iter:
                if not np.isfinite(right_run[ir]):
                    continue
                above_row = above[ir]
                if not above_row.any():
                    continue
                local_vel = vel + local_shift[ir]
                changed   = False

                if not np.isfinite(baseline_right):
                    baseline_right = right_run[ir]
                if not np.isfinite(baseline_left):
                    baseline_left = left_run[ir]

                dbg_baseline_right[it, ir] = baseline_right
                dbg_baseline_left[it, ir]  = baseline_left

                # Record the jump actually seen by the (single, top-to-bottom) pass,
                # relative to the current baseline -- frozen while "active", i.e.
                # inside an already-flagged run.
                dbg_deriv_right[it, ir] = right_run[ir] - baseline_right
                dbg_deriv_left[it, ir]  = baseline_left - left_run[ir]

                # Trusted X-band reference for this gate, used as a fallback only
                # when Ka's own spectral structure gives no clean gap to split on
                # (see _xband_dip_split_mask) -- never as the first choice, since the
                # structural approach is more precise where it actually finds something.
                ref_val_ir = ref_row[ir] if trusted[ir] else np.nan
                w_row_ir = w[ir]

                # --- right edge ---
                if active_right:
                    extra = _extra_outer_run_mask(local_vel, above_row, baseline_right, "right", min_run_bins, merge_gap_tol)
                    if extra is None:
                        extra = _xband_dip_split_mask(w_row_ir, local_vel, above_row, ref_val_ir, baseline_right, "right", interval)
                    if extra is not None:
                        local_shift[ir, extra] -= interval
                        local_vel = vel + local_shift[ir]
                        changed = True
                    else:
                        active_right = False
                if not active_right:
                    if right_run[ir] - baseline_right > right_threshold:
                        extra = _extra_outer_run_mask(local_vel, above_row, baseline_right, "right", min_run_bins, merge_gap_tol)
                        if extra is None:
                            extra = _xband_dip_split_mask(w_row_ir, local_vel, above_row, ref_val_ir, baseline_right, "right", interval)
                        if extra is not None:
                            local_shift[ir, extra] -= interval
                            local_vel = vel + local_shift[ir]
                            changed = True
                            active_right = True

                # --- left edge ---
                if active_left:
                    extra = _extra_outer_run_mask(local_vel, above_row, baseline_left, "left", min_run_bins, merge_gap_tol)
                    if extra is None:
                        extra = _xband_dip_split_mask(w_row_ir, local_vel, above_row, ref_val_ir, baseline_left, "left", interval)
                    if extra is not None:
                        local_shift[ir, extra] += interval
                        local_vel = vel + local_shift[ir]
                        changed = True
                    else:
                        active_left = False
                if not active_left:
                    if baseline_left - left_run[ir] > left_threshold:
                        extra = _extra_outer_run_mask(local_vel, above_row, baseline_left, "left", min_run_bins, merge_gap_tol)
                        if extra is None:
                            extra = _xband_dip_split_mask(w_row_ir, local_vel, above_row, ref_val_ir, baseline_left, "left", interval)
                        if extra is not None:
                            local_shift[ir, extra] += interval
                            local_vel = vel + local_shift[ir]
                            changed = True
                            active_left = True

                # Per-gate debug plot: save one PNG for every gate inside
                # debug_range_km so the user can step through the loop and see
                # the exact state the code sees at each gate (including cascade
                # effects from corrections applied to earlier gates).
                if (debug_plot_dir is not None
                        and it == debug_it
                        and (debug_range_km[0] * 1000 <= ka_range[ir] <= debug_range_km[1] * 1000)):
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as _plt
                    import os as _os
                    _os.makedirs(debug_plot_dir, exist_ok=True)
                    _fig, _ax = _plt.subplots(figsize=(10, 6))
                    _ax.set_facecolor("#111111")
                    # Current right_run across ALL gates (shows cascade state so far)
                    _ax.plot(right_run, ka_range / 1000,
                             color="orange", lw=0.8, alpha=0.6, label="right_run (current state)")
                    _ax.plot(left_run,  ka_range / 1000,
                             color="cyan",   lw=0.8, alpha=0.6, label="left_run (current state)")
                    # Horizontal line = current gate
                    _ax.axhline(ka_range[ir] / 1000, color="white", lw=0.8, ls="--")
                    _dr = right_run[ir] - baseline_right
                    _dl = baseline_left - left_run[ir]
                    _color = "red" if (_dr > right_threshold or _dl > left_threshold or active_right or active_left) else "lime"
                    _ax.scatter([right_run[ir]], [ka_range[ir] / 1000],
                                color=_color, s=60, zorder=6, label=f"Δright={_dr:.2f} m/s")
                    _ax.scatter([left_run[ir]],  [ka_range[ir] / 1000],
                                color=_color, s=60, zorder=6, marker="^",
                                label=f"Δleft(inv)={_dl:.2f} m/s")
                    # Threshold lines (data-driven, per this time step)
                    _ax.axvline( baseline_right + right_threshold, color="red", ls="--", lw=1,
                                 label=f"right baseline+thr={baseline_right + right_threshold:.1f} m/s")
                    _ax.axvline( baseline_left  - left_threshold,  color="red", ls="--", lw=1,
                                 label=f"left baseline-thr={baseline_left - left_threshold:.1f} m/s")
                    _ax.axvline( vnyq_ka,  color="gray", ls=":", lw=0.7, label=f"±vnyq={vnyq_ka:.1f} m/s")
                    _ax.axvline(-vnyq_ka,  color="gray", ls=":", lw=0.7)
                    _ax.set_xlabel("velocity [m/s]")
                    _ax.set_ylabel("range [km]")
                    _ax.set_xlim(-35, 35)
                    _ax.set_title(f"gate ir={ir}  r={ka_range[ir]/1000:.3f} km  "
                                  f"(triggered={changed}, active_right={active_right}, active_left={active_left})")
                    _ax.legend(fontsize=8)
                    _plt.tight_layout()
                    _plt.savefig(
                        f"{debug_plot_dir}/gate_{ir:04d}_r{ka_range[ir]/1000:.3f}km.png",
                        dpi=100,
                    )
                    _plt.close(_fig)

                if changed:
                    dbg_deriv_triggered[it, ir] = 1
                    valid_local    = local_vel[above_row]
                    right_run[ir]  = valid_local.max()
                    left_run[ir]   = valid_local.min()
                    w_row = w[ir]
                    wsum  = w_row[above_row].sum()
                    if wsum > 0:
                        centroid[ir] = float(
                            (w_row[above_row] * local_vel[above_row]).sum() / wsum
                        )

                # Baseline only tracks the trend while in the "normal" state --
                # frozen while a flagged run is still being peeled off (active_*).
                if not active_right:
                    baseline_right = right_run[ir]
                if not active_left:
                    baseline_left = left_run[ir]

        vel_right_arr[:] = right_run
        vel_left_arr[:]  = left_run

        # Store post-Step-1b edges in the "smooth" debug slots so the debug
        # plots show what the algorithm settled on after correction.
        dbg_vel_right_smooth[it] = right_run
        dbg_vel_left_smooth[it]  = left_run

        # -----------------------------------------------------------------------
        # STEP 2 — Per-segment range-unwrap + X-band anchor: resolve absolute fold
        # -----------------------------------------------------------------------
        # After Step 1 each gate has a centroid on the local axis.  The centroid is still
        # ambiguous by multiples of 2*vnyq (it could be the true velocity, or true+2*vnyq,
        # or true-2*vnyq, etc.).  Step 2 resolves that ambiguity using two ideas:
        #   a) Range continuity: adjacent range gates should have similar velocities
        #      (the atmosphere is smooth), so we can unwrap along range just like phase
        #      unwrapping — at each step, add the multiple of 2*vnyq that minimises the
        #      jump from the previous gate.
        #   b) Absolute anchor: range-unwrapping only gives internally consistent
        #      relative velocities; it doesn't tell us which absolute branch the whole
        #      segment is on.  For that we compare the unwrapped centroid profile against
        #      the X-band MDV (which has a much larger Nyquist velocity and is therefore
        #      not aliased) and pick the integer fold offset that minimises the mean
        #      difference.
        #
        # IMPORTANT LIMITATION: range-unwrapping (np.unwrap) assumes the TRUE velocity
        # never jumps by more than vnyq between adjacent range gates.  In deep convection
        # this can be violated (e.g. a strong updraft core next to a downdraft).  If that
        # happens, np.unwrap will take a wrong branch at that gate, and everything above
        # it in the same segment will be shifted by one fold.  The X-band anchor corrects
        # the whole segment but cannot fix a mid-segment unwrapping error.

        # v_row[range]: True wherever centroid is finite (i.e. gate has signal).
        v_row = np.isfinite(centroid)
        if not v_row.any():
            continue
        # gate_fold[range]: the whole-gate fold correction that will be added on top of
        # the within-gate local_shift already computed in Step 1 (and possibly refined
        # by Step 1b above). In the default order this is computed fresh here, from the
        # Step-1b-corrected centroid (see _compute_gate_fold, factored out above so it
        # can also be called earlier when step1b_after_step2 -- in that case gate_fold_pre
        # was already computed before Step 1b ran, from the unmodified Step-1 centroid,
        # and is reused as-is: Step 1b only ever adjusts local_shift/centroid within a
        # gate, never the whole-gate branch, so there is nothing to recompute here.
        gate_fold = gate_fold_pre if step1b_after_step2 else _compute_gate_fold(centroid)
        dbg_gate_fold[it] = gate_fold

        # -----------------------------------------------------------------------
        # STEP 3 — Assemble final n_folds per bin and recompute MDV
        # -----------------------------------------------------------------------
        # true_vel[range, bin]: the physically correct velocity for every above-threshold
        # bin.  It combines:
        #   - vel[bin]:                the apparent (raw) velocity from the spectrum axis
        #   - local_shift[range, bin]: the within-gate glue correction from Step 1
        #                              (-2*vnyq for wrapped trailing bins, 0 otherwise)
        #   - gate_fold[range]*interval: the whole-gate branch correction from Step 2
        true_vel = vel[None, :] + local_shift + gate_fold[:, None] * interval
        # n_folds_out: encode the net velocity shift as an integer fold count so that
        # build_extended / dealias_spectrum can reconstruct the spectrum on the extended
        # axis without storing the full true_vel array.  Only computed for above-threshold
        # bins; below-threshold bins get fold=0 (they won't be displayed/used anyway).
        n_folds_out[it] = np.where(above, np.round((true_vel - vel[None, :]) / interval), 0).astype(np.int8)
        n_valid[it] = v_row

        # Recompute the reflectivity-weighted MDV using true_vel (dealiased) rather than
        # the raw apparent velocity, so the stored MDV matches the dealiased spectrum.
        w_masked = np.where(above, w, 0.0)
        sum_w = w_masked.sum(axis=-1)
        MDV["main"][it] = np.where(v_row, (w_masked * true_vel).sum(axis=-1) / np.where(v_row, sum_w, 1.0), np.nan)
        for ch, chan in extra_power.items():
            cw = np.where(above, chan[it], 0.0)
            csum = cw.sum(axis=-1)
            MDV[ch][it] = np.where(v_row, (cw * true_vel).sum(axis=-1) / np.where(v_row, csum, 1.0), np.nan)

    out_dims = (time_dim, range_dim)
    coords = {time_dim: spec_ka[time_dim], range_dim: spec_ka[range_dim]}
    data_vars = {
        "MDV": xr.DataArray(MDV["main"], dims=out_dims, coords=coords,
                             attrs={"long_name": "dealiased mean Doppler velocity", "units": "m s-1",
                                    "dealiasing": "spectral_glue_range_unwrap_with_reference_anchor", "vnyq": vnyq_ka}),
        "n_folds": xr.DataArray(n_folds_out, dims=spec_ka.dims, coords=spec_ka.coords,
                                 attrs={"long_name": "Doppler aliasing fold count", "units": "1"}),
        "n_valid": xr.DataArray(n_valid, dims=out_dims, coords=coords,
                                 attrs={"long_name": "gate has signal above the calibrated noise floor"}),
        "dbg_vel_right_raw":    xr.DataArray(dbg_vel_right_raw,    dims=out_dims, coords=coords,
                                 attrs={"long_name": "right spectral edge before median filter", "units": "m s-1"}),
        "dbg_vel_right_smooth": xr.DataArray(dbg_vel_right_smooth, dims=out_dims, coords=coords,
                                 attrs={"long_name": "right spectral edge after median filter",  "units": "m s-1"}),
        "dbg_vel_left_raw":     xr.DataArray(dbg_vel_left_raw,     dims=out_dims, coords=coords,
                                 attrs={"long_name": "left spectral edge before median filter",  "units": "m s-1"}),
        "dbg_vel_left_smooth":  xr.DataArray(dbg_vel_left_smooth,  dims=out_dims, coords=coords,
                                 attrs={"long_name": "left spectral edge after median filter",   "units": "m s-1"}),
        "dbg_gate_fold":        xr.DataArray(dbg_gate_fold,        dims=out_dims, coords=coords,
                                 attrs={"long_name": "Step-2 whole-gate fold correction (integer)", "units": "1"}),
        "dbg_deriv_right":      xr.DataArray(dbg_deriv_right,      dims=out_dims, coords=coords,
                                 attrs={"long_name": "Step-1b right-edge derivative seen by bottom-to-top pass", "units": "m s-1"}),
        "dbg_deriv_left":       xr.DataArray(dbg_deriv_left,       dims=out_dims, coords=coords,
                                 attrs={"long_name": "Step-1b left-edge derivative seen by bottom-to-top pass",  "units": "m s-1"}),
        "dbg_deriv_triggered":  xr.DataArray(dbg_deriv_triggered,  dims=out_dims, coords=coords,
                                 attrs={"long_name": "1 where bottom-to-top pass triggered a correction", "units": "1"}),
        "dbg_right_jump_threshold": xr.DataArray(dbg_right_jump_threshold, dims=(time_dim,), coords={time_dim: spec_ka[time_dim]},
                                 attrs={"long_name": "Step-1b right-edge jump threshold actually used (edge_jump_mad_mult * robust profile scale)", "units": "m s-1"}),
        "dbg_left_jump_threshold":  xr.DataArray(dbg_left_jump_threshold,  dims=(time_dim,), coords={time_dim: spec_ka[time_dim]},
                                 attrs={"long_name": "Step-1b left-edge jump threshold actually used (edge_jump_mad_mult * robust profile scale)", "units": "m s-1"}),
        "dbg_baseline_right":   xr.DataArray(dbg_baseline_right, dims=out_dims, coords=coords,
                                 attrs={"long_name": "Step-1b frozen right-edge baseline seen by bottom-to-top pass", "units": "m s-1"}),
        "dbg_baseline_left":    xr.DataArray(dbg_baseline_left,  dims=out_dims, coords=coords,
                                 attrs={"long_name": "Step-1b frozen left-edge baseline seen by bottom-to-top pass",  "units": "m s-1"}),
    }
    for ch in extra_channels:
        data_vars[f"MDV_{ch}"] = xr.DataArray(
            MDV[ch], dims=out_dims, coords=coords,
            attrs={"long_name": f"dealiased mean Doppler velocity ({ch})", "units": "m s-1",
                   "dealiasing": "spectral_glue_range_unwrap_with_reference_anchor (fold shared with main channel)",
                   "vnyq": vnyq_ka},
        )
    return xr.Dataset(data_vars)


# =============================================================================
# Peak-tracking dealiasing (2025-06-04/05 session): an alternative to
# `dealias_ka_spectrum_with_range_reference` above, which is kept fully intact.
# See that function's Step 1b for the earlier, hand-rolled approach (edge-jump
# hysteresis, run-structure "which run is outermost", the X-band dip-split
# fallback) and everything that was learned debugging it -- in particular that a
# *single* scalar baseline per gate breaks down whenever more than one real
# population coexists nearby, since they end up fighting over the same reference.
#
# This instead borrows PEAKO/peakTree's approach (Kalesse et al. 2019 AMT;
# Radenz et al. 2019 AMT, from /project/meteo/work/L.Terzi/peako_peaktree):
#   1. Find distinct populations ("peaks") per gate with a *trained* prominence
#      threshold via scipy.signal.find_peaks, not a guessed constant -- see
#      `_find_spectral_peaks`.
#   2. Track each population across range gates independently (nearest-neighbor
#      matching by velocity, tolerant of a few gates of gap), so each one gets its
#      own continuity/fold state instead of sharing one baseline.
#   3. Resolve each track's absolute fold once, the same way Step 2 already does
#      for a whole gate (range-unwrap + X-band anchor where trusted) -- just
#      applied per track instead of per-gate scalar centroid.
# =============================================================================

def _find_spectral_peaks(
    w_row: np.ndarray,
    prom_db: float,
    savgol_window: int,
    savgol_polyorder: int,
    width_bins: float = 0.0,
    floor_db: float = -40.0,
    max_peaks: int | None = None,
) -> list[dict]:
    """
    Find distinct spectral populations ("peaks") in one gate's raw power spectrum,
    ported from PEAKO/peakTree's approach rather than hand-rolled: smooth with a
    Savitzky-Golay filter, find local maxima via scipy.signal.find_peaks with a
    trained prominence (and optionally width) threshold, then take the boundary
    between two adjacent peaks as the real local minimum strictly between them --
    not a blind minimum search across the whole gate (see the 2025-06-04 18:00:40
    gate-172 case: a blind search there picks up ordinary edge-taper, not a genuine
    second population).

    max_peaks caps how many peaks are kept when more than that clear the prominence
    threshold -- the max_peaks most prominent ones survive, weakest dropped.

    Returns a list of dicts {"lo", "hi", "peak_bin", "power_db"}, one per detected
    peak, partitioning this gate's above-threshold bins between them. Empty list if
    there's no signal at all in this gate.
    """
    above = w_row > 0
    if not above.any():
        return []
    idx = np.where(above)[0]
    lo_all, hi_all = int(idx.min()), int(idx.max())
    db = np.where(above, 10.0 * np.log10(np.maximum(w_row, 1e-30)), floor_db)
    db = np.maximum(db, floor_db)

    window = min(savgol_window, hi_all - lo_all + 1)
    if window < 5:
        # Too narrow to smooth meaningfully -- treat as one peak, no internal split.
        pk = lo_all + int(np.argmax(db[lo_all:hi_all + 1]))
        return [{"lo": lo_all, "hi": hi_all, "peak_bin": pk, "power_db": float(db[pk])}]
    if window % 2 == 0:
        window -= 1
    polyorder = min(savgol_polyorder, window - 1)

    from scipy.signal import savgol_filter, find_peaks
    smoothed = savgol_filter(db, window, polyorder)
    find_kwargs = {"prominence": prom_db}
    if width_bins and width_bins > 0:
        find_kwargs["width"] = width_bins
    peak_bins, props = find_peaks(smoothed, **find_kwargs)
    prominences = props["prominences"]
    # Savitzky-Golay can ring slightly past the true above/below-threshold boundary;
    # only keep peaks that land on a genuinely above-threshold bin.
    keep = np.array([above[p] for p in peak_bins], dtype=bool)
    peak_bins = peak_bins[keep]
    prominences = prominences[keep]

    if peak_bins.size == 0:
        # No peak clears the prominence bar -- one coherent, low-contrast population.
        pk = lo_all + int(np.argmax(db[lo_all:hi_all + 1]))
        return [{"lo": lo_all, "hi": hi_all, "peak_bin": pk, "power_db": float(db[pk])}]

    if max_peaks is not None and peak_bins.size > max_peaks:
        # Keep the max_peaks most prominent peaks, not the leftmost-N (PEAKO/peakTree's
        # own -- disabled -- truncation cuts the rightmost peaks by bin index instead;
        # that's an arbitrary velocity bias, not a signal-quality criterion).
        top = np.sort(np.argsort(prominences)[::-1][:max_peaks])
        peak_bins = peak_bins[top]
        prominences = prominences[top]

    # Boundaries: the real local minimum strictly between each pair of adjacent
    # confirmed peaks (PEAKO's find_edges "merged peaks" logic) -- not a search
    # across the whole gate.
    bounds = [lo_all]
    for i in range(len(peak_bins) - 1):
        seg = smoothed[peak_bins[i]:peak_bins[i + 1] + 1]
        bounds.append(peak_bins[i] + int(np.argmin(seg)))
    bounds.append(hi_all)

    peaks = []
    for i, pb in enumerate(peak_bins):
        lo = bounds[i] if i == 0 else bounds[i] + 1
        hi = bounds[i + 1]
        peaks.append({"lo": int(lo), "hi": int(hi), "peak_bin": int(pb), "power_db": float(db[pb])})
    return peaks


def dealias_ka_spectrum_peaktrack(
    spec_ka: xr.DataArray,
    ref_mdv: xr.DataArray,
    vnyq_ka: float,
    prom_thres_db: float = 1.0,
    width_thres_ms: float = 0.0,
    span_ms: float = 0.4,
    polyorder: int = 3,
    max_gap_gates: int = 2,
    max_jump_frac: float = 1.0,
    trust_min_range: float = 2000.0,
    max_peaks: int | None = None,
    edge_bins: int = 3,
    max_gap_bins: int = 10,
    precomputed_gate_fold: xr.DataArray | None = None,
    vel_dim: str = "Vel",
    time_dim: str = "time",
    range_dim: str = "range",
) -> xr.Dataset:
    """
    Peak-tracking dealiasing -- see the module-level comment above this function for
    the full rationale relative to `dealias_ka_spectrum_with_range_reference`.

    Parameters
    ----------
    prom_thres_db, width_thres_ms, span_ms, polyorder : PEAKO-trained peak-finding
        parameters (see /project/meteo/work/L.Terzi/peako_peaktree/tripex_pol/plots/
        results_20260205_Ka-Band.txt for the Ka-band-trained values -- note these
        were trained on a different, apparently quieter campaign and have not been
        validated on this noisier convective case; defaults here are that trained
        result, not yet re-validated).
    max_gap_gates : how many consecutive gates a track may go undetected in before
        it's considered ended (mirrors `max_gap_bins` in `_find_wrap_bins`, just
        applied along range to a track instead of along velocity to bins).
    max_jump_frac : maximum allowed true-velocity change between adjacent gates for
        a peak to still count as the same track, as a fraction of vnyq_ka -- the
        same range-continuity assumption Step 2's own np.unwrap already relies on
        elsewhere in this module, reused here per-track instead of per-gate-centroid.
    max_peaks : if set, caps the number of peaks kept per gate to the most prominent
        max_peaks (see `_find_spectral_peaks`) -- a safety valve against untrained/
        mismatched peak-finding params over-detecting spurious peaks in noisy gates.
    edge_bins, max_gap_bins : same meaning as in `dealias_ka_spectrum_with_range_reference`
        -- this function reuses that function's Step 1 (per-gate glue at +/-vnyq,
        `_step1_glue_gate`) before peak-finding, so a population straddling the fold
        boundary reads as one contiguous peak instead of two edge-hugging fragments.
        This function only replaces Step 1b (the missed/partially-unglued-piece catch);
        Step 1's own glue and Step 2's range-unwrap-and-anchor role are still needed and
        are reused here, not reimplemented.
    precomputed_gate_fold : if given (dims time,range, same convention as this module's
        `dbg_gate_fold` output -- e.g. from `dealias_ka_spectrum_with_range_reference(...,
        edge_jump_mad_mult=1e6)` to get a Step1b-free Step1+Step2 anchor), peak-finding
        and tracking run in the ORDER Step1 -> Step2 -> peaktrack: each gate's velocity
        axis is shifted by this precomputed absolute fold *before* peak-finding, and the
        internal per-track X-band anchor is skipped entirely (the absolute frame is
        already established). When None (default), the order is Step1 -> peaktrack ->
        (per-track anchor, playing Step 2's role) -- this function's original behaviour.

    Returns
    -------
    Dataset with MDV, n_folds (per-bin, dims of spec_ka, int8), n_valid (bool) --
    same convention as `dealias_ka_spectrum_with_range_reference`.
    """
    spec_ka = spec_ka.transpose(time_dim, range_dim, vel_dim)
    vel = spec_ka[vel_dim].values.astype(np.float64)
    nfft = vel.size
    ka_time = spec_ka[time_dim].values
    ka_range = spec_ka[range_dim].values.astype(np.float64)
    n_time, n_range = spec_ka.sizes[time_dim], spec_ka.sizes[range_dim]
    Ka_power = np.nan_to_num(spec_ka.values, nan=0.0).astype(np.float64)

    interval = 2.0 * vnyq_ka
    bin_width = interval / nfft
    savgol_window = int(round(span_ms / bin_width))
    if savgol_window % 2 == 0:
        savgol_window += 1
    width_bins = width_thres_ms / bin_width if width_thres_ms > 0 else 0.0
    max_jump = max_jump_frac * vnyq_ka

    def _to_epoch(t):
        return t.astype("datetime64[s]").astype(np.float64) if np.issubdtype(t.dtype, np.datetime64) else t.astype(np.float64)

    ref_mdv = ref_mdv.transpose(time_dim, range_dim)
    ref_ti_idx = _nearest_index(_to_epoch(ref_mdv[time_dim].values), _to_epoch(ka_time))
    ref_ri_idx = _nearest_index(ref_mdv[range_dim].values.astype(np.float64), ka_range)
    ref_on_ka = ref_mdv.values[ref_ti_idx][:, ref_ri_idx]
    trusted = ka_range >= trust_min_range

    gate_fold_arr = None
    if precomputed_gate_fold is not None:
        gate_fold_arr = precomputed_gate_fold.transpose(time_dim, range_dim).values.astype(np.float64)

    n_folds_out = np.zeros((n_time, n_range, nfft), dtype=np.int8)
    n_valid = np.zeros((n_time, n_range), dtype=bool)
    MDV = np.full((n_time, n_range), np.nan, dtype=np.float32)

    for it in range(n_time):
        w = Ka_power[it]
        ref_row = ref_on_ka[it]

        # Active tracks: dicts with the population's current fold-resolved
        # ("unwrapped") velocity, bookkeeping for gap tolerance, and every gate
        # entry seen so far (needed to write out final per-bin folds afterward).
        active: list[dict] = []
        finished: list[dict] = []

        for ir in range(n_range):
            above_row = w[ir] > 0
            # Step 1's own glue (reused, not reimplemented -- see docstring): if this
            # gate's spectrum straddles +/-vnyq, roll it by s bins so the wrapped piece
            # sits contiguous with the rest on a local velocity axis, before peak-finding.
            # Non-straddling gates get s=0 (no-op: w_row_glued/vel_glued == w[ir]/vel).
            s = _step1_glue_gate(above_row, edge_bins, max_gap_bins)
            if s > 0:
                w_row_glued = np.roll(w[ir], s)
                vel_glued = np.roll(vel, s)
                vel_glued[:s] -= interval
            else:
                w_row_glued = w[ir]
                vel_glued = vel
            if gate_fold_arr is not None:
                # Step 2 already ran (precomputed_gate_fold) -- shift this gate's local
                # (Step-1-glued) axis into absolute velocity space before peak-finding,
                # so peaktrack operates entirely downstream of the anchor instead of
                # establishing its own (see docstring). `+` always allocates a new array,
                # so this never mutates the shared `vel`/`vel_glued` (== vel when s==0).
                vel_glued = vel_glued + gate_fold_arr[it, ir] * interval

            peaks = _find_spectral_peaks(w_row_glued, prom_thres_db, savgol_window, polyorder, width_bins, max_peaks=max_peaks)
            if not peaks:
                still_active = []
                for tr in active:
                    tr["gap_count"] += 1
                    (finished if tr["gap_count"] > max_gap_gates else still_active).append(tr)
                active = still_active
                continue

            # Greedy nearest-neighbor matching: for each peak, find the closest
            # not-yet-claimed active track (after unwrapping the peak's raw velocity
            # to whichever fold branch is nearest that track's current position).
            # raw_vel is read from vel_glued (the local, glue-corrected axis), not the
            # unglued vel, so a peak straddling the fold boundary matches on its real
            # local velocity rather than wherever its bin happens to sit pre-glue.
            matched_ids: set[int] = set()
            gate_matches = []
            for peak in peaks:
                raw_vel = vel_glued[peak["peak_bin"]]
                best_ti, best_jump, best_uv = None, None, None
                for ti, tr in enumerate(active):
                    if ti in matched_ids:
                        continue
                    n = round((tr["unwrapped_vel"] - raw_vel) / interval)
                    cand_uv = raw_vel + n * interval
                    jump = abs(cand_uv - tr["unwrapped_vel"])
                    if jump <= max_jump and (best_jump is None or jump < best_jump):
                        best_ti, best_jump, best_uv = ti, jump, cand_uv
                if best_ti is not None:
                    matched_ids.add(best_ti)
                gate_matches.append((peak, best_ti, best_uv if best_ti is not None else raw_vel))

            new_active = []
            for ti, tr in enumerate(active):
                if ti in matched_ids:
                    continue  # re-added below with its update applied
                tr["gap_count"] += 1
                (finished if tr["gap_count"] > max_gap_gates else new_active).append(tr)

            for peak, ti, uv in gate_matches:
                entry = (ir, peak["lo"], peak["hi"], peak["peak_bin"], uv, s)
                if ti is not None:
                    tr = active[ti]
                    tr["unwrapped_vel"] = uv
                    tr["gap_count"] = 0
                    tr["entries"].append(entry)
                    new_active.append(tr)
                else:
                    new_active.append({
                        "unwrapped_vel": uv,
                        "gap_count": 0,
                        "entries": [entry],
                    })
            active = new_active

        finished.extend(active)

        # Resolve each track's absolute fold once, then write out bins. When
        # gate_fold_arr is set, Step 2 already anchored everything before peak-finding
        # (uv is already absolute) so the per-track X-band anchor is skipped (best_n=0);
        # otherwise this is the X-band anchor, playing Step 2's role.
        for tr in finished:
            entries = tr["entries"]
            if not entries:
                continue
            if gate_fold_arr is not None:
                best_n = 0
            else:
                gate_idx = np.array([e[0] for e in entries])
                uvs = np.array([e[4] for e in entries])
                ref_here = ref_row[gate_idx]
                seg_trusted = trusted[gate_idx]
                use_mask = (seg_trusted if seg_trusted.any() else np.ones(len(gate_idx), dtype=bool)) & np.isfinite(ref_here)
                if not use_mask.any():
                    use_mask = np.isfinite(ref_here)
                if use_mask.any():
                    offset = np.mean(ref_here[use_mask] - uvs[use_mask])
                    best_n = round(offset / interval)
                else:
                    best_n = 0

            for (ir, lo, hi, peak_bin, uv, s) in entries:
                # peak_bin/lo/hi index the ROLLED (glued) array for this gate -- convert
                # back to original bin indices, and add back the local glue fold (-1 for
                # bins that came from the rolled tail, i.e. were already folded down by
                # Step 1's own glue) on top of this track's resolved absolute fold, so a
                # single physically-glued peak still gets the correct *combined* per-bin
                # fold even though its two halves came from different original folds.
                orig_pb = (peak_bin - s) % nfft
                vel_glued_pb = vel[orig_pb] - interval if (s > 0 and orig_pb >= nfft - s) else vel[orig_pb]
                if gate_fold_arr is not None:
                    vel_glued_pb = vel_glued_pb + gate_fold_arr[it, ir] * interval
                final_vel = uv + best_n * interval
                base_fold = int(round((final_vel - vel_glued_pb) / interval))
                for i in range(lo, hi + 1):
                    ob = (i - s) % nfft
                    local_glue_fold = -1 if (s > 0 and ob >= nfft - s) else 0
                    n_folds_out[it, ir, ob] = base_fold + local_glue_fold
                n_valid[it, ir] = True

        above_full = w > 0
        true_vel = vel[None, :] + n_folds_out[it].astype(np.float64) * interval
        w_masked = np.where(above_full, w, 0.0)
        sum_w = w_masked.sum(axis=-1)
        v_row = n_valid[it]
        MDV[it] = np.where(v_row, (w_masked * true_vel).sum(axis=-1) / np.where(v_row, sum_w, 1.0), np.nan)

    out_dims = (time_dim, range_dim)
    coords = {time_dim: spec_ka[time_dim], range_dim: spec_ka[range_dim]}
    return xr.Dataset({
        "MDV": xr.DataArray(MDV, dims=out_dims, coords=coords,
                             attrs={"long_name": "dealiased mean Doppler velocity", "units": "m s-1",
                                    "dealiasing": "peak_tracking_range_continuity_with_reference_anchor",
                                    "vnyq": vnyq_ka}),
        "n_folds": xr.DataArray(n_folds_out, dims=spec_ka.dims, coords=spec_ka.coords,
                                 attrs={"long_name": "Doppler aliasing fold count", "units": "1"}),
        "n_valid": xr.DataArray(n_valid, dims=out_dims, coords=coords,
                                 attrs={"long_name": "gate has signal above the calibrated noise floor"}),
    })


# =============================================================================
# Ridge-tracking dealiasing (2025-09-30 session): a second alternative to
# dealias_ka_spectrum_peaktrack above, using dynamic-programming ("Viterbi-style")
# ridge tracking instead of greedy nearest-neighbor peak matching for Step 1b's
# role. Validated first on the co-located X-band spectrum (no aliasing at all, so
# a clean test of the tracking mechanism alone): greedy peak matching fragmented
# one clearly-visible single band into 60+ disconnected, criss-crossing tracks,
# because it commits gate-by-gate with no memory of the recent trend and no way
# to recover from one bad local match. DP instead finds the single path that
# maximises TOTAL reward across the whole profile at once, which tracked that
# same band cleanly end to end. See the session's search for prior art: this is
# the same idea as sinusoidal partial tracking (audio DSP) and MATLAB's tfridge.
#
# A second, real population (present at some gates, absent at most) is found by
# "peeling": suppress ridge 1's own extent (found via _find_spectral_peaks'
# already-existing real-local-minimum boundary logic, NOT a fixed dB-drop guess
# -- an earlier attempt at a fixed-window/fixed-dB-drop suppression left enough
# of ridge 1's own shoulder that peeling just re-found the same population from
# the other side), then look for a second, independently-anchored ridge only
# where a genuine local bimodality test says one exists (two real peaks,
# comparable enough in power, NOT just "technically above the noise floor" --
# that check let through almost the entire profile), persisting over a minimum
# run of consecutive gates (an isolated bimodal gate is far more likely a fluke
# than a real second population). Everywhere that test fails, there simply IS no
# second ridge -- this is masked out rather than force-fit, unlike the earlier
# peel-then-filter attempt where DP always found *something* even in pure noise.
#
# Both ridges are then fold-resolved independently by reusing Step 2's own
# range-unwrap + X-band-anchor idea (see _compute_gate_fold in
# dealias_ka_spectrum_with_range_reference above) -- applied once per ridge
# instead of once per gate-centroid, since two real populations can legitimately
# sit on two different absolute branches at the same gates.
# =============================================================================


def _ridge_dp(db: np.ndarray, max_jump_bins: int, penalty_per_bin: float) -> np.ndarray:
    """
    Single-ridge dynamic-programming ("Viterbi-style") path-finder: the path
    through `db` (n_gates, n_vel_bins) that maximises total reward (sum of db
    along the path) minus a penalty for jumping between adjacent gates, allowing
    jumps of up to `max_jump_bins`. Unlike greedy nearest-neighbor matching (see
    dealias_ka_spectrum_peaktrack), this considers the WHOLE profile at once, so
    one noisy/ambiguous gate can't derail the rest of the path.
    """
    n_gates, n_bins = db.shape
    DP = np.full((n_gates, n_bins), -np.inf)
    back = np.zeros((n_gates, n_bins), dtype=int)
    DP[0] = db[0]
    offsets = np.arange(-max_jump_bins, max_jump_bins + 1)
    for ig in range(1, n_gates):
        prev = DP[ig - 1]
        best_val = np.full(n_bins, -np.inf)
        best_off = np.zeros(n_bins, dtype=int)
        for off in offsets:
            shifted = np.roll(prev, off)
            if off > 0:
                shifted[:off] = -np.inf
            elif off < 0:
                shifted[off:] = -np.inf
            cand = shifted - penalty_per_bin * abs(off)
            better = cand > best_val
            best_val[better] = cand[better]
            best_off[better] = off
        DP[ig] = db[ig] + best_val
        back[ig] = np.arange(n_bins) - best_off
    path = np.zeros(n_gates, dtype=int)
    path[-1] = int(np.argmax(DP[-1]))
    for ig in range(n_gates - 1, 0, -1):
        path[ig - 1] = back[ig, path[ig]]
    return path


def dealias_ka_spectrum_ridge(
    spec_ka: xr.DataArray,
    ref_mdv: xr.DataArray,
    vnyq_ka: float,
    prom_db: float = 3.0,
    span_ms: float = 0.4,
    polyorder: int = 3,
    max_jump_ms: float = 3.0,
    jump_penalty_per_ms: float = 2.0,
    min_sep_ms: float = 1.0,
    max_sep_drop_db: float = 30.0,
    min_run_gates: int = 5,
    local_window_ms: float = 3.0,
    edge_bins: int = 3,
    max_gap_bins: int = 10,
    trust_min_range: float = 2000.0,
    vel_dim: str = "Vel",
    time_dim: str = "time",
    range_dim: str = "range",
) -> xr.Dataset:
    """
    Ridge-tracking dealiasing -- see the module-level comment above for the full
    rationale and how this differs from dealias_ka_spectrum_peaktrack. Reuses
    Step 1's glue (_step1_glue_gate) exactly like that function does; this only
    replaces Step 1b + Step 2's role, not Step 1 itself.

    Parameters
    ----------
    prom_db, span_ms, polyorder : peak-finding parameters, same meaning as in
        dealias_ka_spectrum_peaktrack.
    max_jump_ms, jump_penalty_per_ms : DP path-finding parameters -- how far a
        ridge may jump between adjacent gates, and how much each m/s of jump is
        penalised (discourages wandering even when technically allowed).
    min_sep_ms, max_sep_drop_db : bimodality test -- the two strongest peaks at
        a gate must be at least min_sep_ms apart in velocity, and the second
        must be within max_sep_drop_db of the first's power, to count as a
        genuine second population rather than a noise ripple. (Validated on
        X-band: a version of this check that only looked at "above the
        calibrated noise floor" let through ~90% of gates -- far too weak a
        bar. Power-comparability to the DOMINANT peak, not to the noise floor,
        is what actually distinguishes a real secondary population.)
    min_run_gates : a gate passing the bimodality test in isolation is more
        likely a fluke than a real population; only trust runs of at least this
        many consecutive bimodal gates.
    local_window_ms : once a run is validated, ridge 2 within it is found by a
        DP pass restricted to this window around the per-gate detected second
        peak -- enough to smooth gate-to-gate jitter without re-opening a wide
        search that could wander into noise or back onto ridge 1.
    edge_bins, max_gap_bins : passed straight through to _step1_glue_gate, same
        meaning as in dealias_ka_spectrum_with_range_reference.

    Returns
    -------
    Dataset with MDV, n_folds (per-bin, dims of spec_ka, int8), n_valid (bool) --
    same convention as the other two dealiasing functions in this module.
    """
    import scipy.ndimage

    spec_ka = spec_ka.transpose(time_dim, range_dim, vel_dim)
    vel = spec_ka[vel_dim].values.astype(np.float64)
    nfft = vel.size
    ka_time = spec_ka[time_dim].values
    ka_range = spec_ka[range_dim].values.astype(np.float64)
    n_time, n_range = spec_ka.sizes[time_dim], spec_ka.sizes[range_dim]
    Ka_power = np.nan_to_num(spec_ka.values, nan=0.0).astype(np.float64)

    interval = 2.0 * vnyq_ka
    bin_width = interval / nfft
    FLOOR_DB = -40.0

    savgol_window = int(round(span_ms / bin_width))
    if savgol_window % 2 == 0:
        savgol_window += 1

    max_jump_bins = max(1, int(round(max_jump_ms / bin_width)))
    penalty_per_bin = jump_penalty_per_ms * bin_width
    local_window_bins = max(1, int(round(local_window_ms / bin_width)))

    def _to_epoch(t):
        return t.astype("datetime64[s]").astype(np.float64) if np.issubdtype(t.dtype, np.datetime64) else t.astype(np.float64)

    ref_mdv = ref_mdv.transpose(time_dim, range_dim)
    ref_ti_idx = _nearest_index(_to_epoch(ref_mdv[time_dim].values), _to_epoch(ka_time))
    ref_ri_idx = _nearest_index(ref_mdv[range_dim].values.astype(np.float64), ka_range)
    ref_on_ka = ref_mdv.values[ref_ti_idx][:, ref_ri_idx]
    trusted = ka_range >= trust_min_range

    # Common extended velocity axis every gate's (glued) spectrum gets placed onto,
    # so DP's "small index jump = small velocity jump" assumption holds even though
    # different gates may have been glued (rolled) by different amounts s. Wide
    # enough to hold Step 1's glue shift (up to one interval) plus range-unwrap.
    N_EXT = 5
    center_k = N_EXT // 2
    ext_vel = np.concatenate([vel + k * interval for k in range(-center_k, center_k + 1)])
    ext_len = ext_vel.size
    center_offset = center_k * nfft

    n_folds_out = np.zeros((n_time, n_range, nfft), dtype=np.int8)
    n_valid = np.zeros((n_time, n_range), dtype=bool)
    MDV = np.full((n_time, n_range), np.nan, dtype=np.float32)

    def _anchor_ridge(ext_positions: np.ndarray, valid_mask: np.ndarray, ref_row: np.ndarray) -> np.ndarray:
        """Step 2's range-unwrap + X-band anchor, applied to one ridge's own
        per-gate extended-axis position sequence instead of a per-gate centroid."""
        fold = np.zeros(n_range, dtype=np.int8)
        if not valid_mask.any():
            return fold
        labels, n_labels = scipy.ndimage.label(valid_mask)
        raw_vel = np.where(valid_mask, ext_vel[np.clip(ext_positions, 0, ext_len - 1)], np.nan)
        for lab in range(1, n_labels + 1):
            seg = labels == lab
            phase_seg = raw_vel[seg] * np.pi / vnyq_ka
            unwrapped_seg = np.unwrap(phase_seg) * vnyq_ka / np.pi
            ref_here = ref_row[seg]
            seg_trusted = trusted[seg]
            use_mask = (seg_trusted if seg_trusted.any() else np.ones(seg.sum(), dtype=bool)) & np.isfinite(ref_here)
            if not use_mask.any():
                use_mask = np.isfinite(ref_here)
            if use_mask.any():
                offset = np.mean(ref_here[use_mask] - unwrapped_seg[use_mask])
                best_n = round(offset / interval)
            else:
                best_n = 0
            unwrap_per_gate = np.round((unwrapped_seg - raw_vel[seg]) / interval).astype(np.int8)
            fold[seg] = (best_n + unwrap_per_gate).astype(np.int8)
        return fold

    def _write_ridge_bins(it: int, ir: int, s: int, lo: int, hi: int, fold: int) -> None:
        """Write one ridge's resolved fold into n_folds_out for this gate's peak
        span [lo, hi] (rolled/glued-array coords) -- same per-bin bookkeeping as
        dealias_ka_spectrum_peaktrack: convert back to original bin indices and
        add back the local glue fold for bins that came from the rolled tail."""
        for i in range(lo, hi + 1):
            ob = (i - s) % nfft
            local_glue_fold = -1 if (s > 0 and ob >= nfft - s) else 0
            n_folds_out[it, ir, ob] = fold + local_glue_fold
        n_valid[it, ir] = True

    for it in range(n_time):
        w = Ka_power[it]
        ref_row = ref_on_ka[it]

        s_arr = np.zeros(n_range, dtype=int)
        has_signal = np.zeros(n_range, dtype=bool)
        ext_db = np.full((n_range, ext_len), FLOOR_DB)
        peaks_per_gate: list[list[dict]] = []

        for ir in range(n_range):
            above_row = w[ir] > 0
            has_signal[ir] = above_row.any()
            s = _step1_glue_gate(above_row, edge_bins, max_gap_bins)
            s_arr[ir] = s
            db_row = np.where(above_row, 10.0 * np.log10(np.maximum(w[ir], 1e-30)), FLOOR_DB)
            db_row = np.maximum(db_row, FLOOR_DB)

            ext_idx = np.arange(nfft) + center_offset
            if s > 0:
                ext_idx[:s] -= nfft
                rolled_db = np.roll(db_row, s)
                rolled_w = np.roll(w[ir], s)
            else:
                rolled_db = db_row
                rolled_w = w[ir]
            ext_db[ir, ext_idx] = rolled_db

            peaks = _find_spectral_peaks(rolled_w, prom_db, savgol_window, polyorder, 0.0)
            conv_peaks = []
            for p in peaks:
                pb = p["peak_bin"]
                ext_peak = pb + center_offset - (nfft if (s > 0 and pb < s) else 0)
                conv_peaks.append({
                    "lo": p["lo"], "hi": p["hi"], "peak_bin": pb,
                    "ext_peak": ext_peak, "power_db": p["power_db"],
                })
            peaks_per_gate.append(conv_peaks)

        if not has_signal.any():
            continue

        # ---- ridge 1: global DP, the one dominant path (always exists) ----
        path1 = _ridge_dp(ext_db, max_jump_bins, penalty_per_bin)

        # ---- ridge 2: local bimodality test -> persistence filter -> local DP ----
        bimodal = np.zeros(n_range, dtype=bool)
        peak2_ext = np.full(n_range, -1, dtype=int)
        for ir in range(n_range):
            conv_peaks = peaks_per_gate[ir]
            if len(conv_peaks) < 2:
                continue
            ps = sorted(conv_peaks, key=lambda p: -p["power_db"])
            p1, p2 = ps[0], ps[1]
            sep_ms = abs(ext_vel[p1["ext_peak"]] - ext_vel[p2["ext_peak"]])
            drop_db = p1["power_db"] - p2["power_db"]
            if sep_ms >= min_sep_ms and drop_db <= max_sep_drop_db:
                bimodal[ir] = True
                peak2_ext[ir] = p2["ext_peak"]

        long_runs = [(s2, e2) for (s2, e2) in _bool_runs(bimodal) if (e2 - s2 + 1) >= min_run_gates]

        path2 = np.full(n_range, -1, dtype=int)
        ridge2_valid = np.zeros(n_range, dtype=bool)
        for (s2, e2) in long_runs:
            sub_db = ext_db[s2:e2 + 1]
            restricted = np.full_like(sub_db, FLOOR_DB)
            for k, ir in enumerate(range(s2, e2 + 1)):
                center_bin = int(peak2_ext[ir])
                lo = max(0, center_bin - local_window_bins)
                hi = min(ext_len, center_bin + local_window_bins + 1)
                restricted[k, lo:hi] = sub_db[k, lo:hi]
            sub_path = _ridge_dp(restricted, max_jump_bins, penalty_per_bin)
            path2[s2:e2 + 1] = sub_path
            ridge2_valid[s2:e2 + 1] = True

        # ---- resolve each ridge's absolute fold independently ----
        fold1 = _anchor_ridge(path1, has_signal, ref_row)
        fold2 = _anchor_ridge(path2, ridge2_valid, ref_row)

        # ---- assign bins to whichever ridge they belong to, write folds ----
        for ir in range(n_range):
            if not has_signal[ir]:
                continue
            conv_peaks = peaks_per_gate[ir]
            if not conv_peaks:
                continue
            s = s_arr[ir]

            ridge2_here = ridge2_valid[ir] and path2[ir] >= 0
            if ridge2_here:
                p2_peak = min(conv_peaks, key=lambda p: abs(p["ext_peak"] - path2[ir]))
                _write_ridge_bins(it, ir, s, p2_peak["lo"], p2_peak["hi"], int(fold2[ir]))
                remaining = [p for p in conv_peaks if p is not p2_peak]
            else:
                remaining = conv_peaks

            if remaining:
                p1_peak = min(remaining, key=lambda p: abs(p["ext_peak"] - path1[ir]))
                _write_ridge_bins(it, ir, s, p1_peak["lo"], p1_peak["hi"], int(fold1[ir]))

        above_full = w > 0
        true_vel = vel[None, :] + n_folds_out[it].astype(np.float64) * interval
        w_masked = np.where(above_full, w, 0.0)
        sum_w = w_masked.sum(axis=-1)
        v_row = n_valid[it]
        MDV[it] = np.where(v_row, (w_masked * true_vel).sum(axis=-1) / np.where(v_row, sum_w, 1.0), np.nan)

    out_dims = (time_dim, range_dim)
    coords = {time_dim: spec_ka[time_dim], range_dim: spec_ka[range_dim]}
    return xr.Dataset({
        "MDV": xr.DataArray(MDV, dims=out_dims, coords=coords,
                             attrs={"long_name": "dealiased mean Doppler velocity", "units": "m s-1",
                                    "dealiasing": "ridge_tracking_dp_with_reference_anchor",
                                    "vnyq": vnyq_ka}),
        "n_folds": xr.DataArray(n_folds_out, dims=spec_ka.dims, coords=spec_ka.coords,
                                 attrs={"long_name": "Doppler aliasing fold count", "units": "1"}),
        "n_valid": xr.DataArray(n_valid, dims=out_dims, coords=coords,
                                 attrs={"long_name": "gate has signal above the calibrated noise floor"}),
    })


def dealias_spectrum(
    spec: xr.DataArray,
    n_folds: np.ndarray,
    vel_dim: str = "Vel",
) -> xr.DataArray:
    """
    Place each pixel's spectrum on an extended [-3·Vnyq, +3·Vnyq] velocity axis.

    For each (time, range) pixel, the Nfft spectrum bins are written into the
    slot corresponding to its fold count n:
      n= 0 → middle slot  [-Vnyq,  +Vnyq]
      n=+1 → right slot   [+Vnyq, +3Vnyq]
      n=-1 → left slot    [-3Vnyq, -Vnyq]
    All other positions are NaN. Straddling is handled because the signal
    already appears at its measured position within the original Nfft, which
    maps to the physically correct velocity in the extended axis.

    Parameters
    ----------
    spec    : DataArray with dims (..., vel_dim)
    n_folds : int array with the same leading dims as spec (no vel_dim)
    vel_dim : name of the velocity dimension
    """
    vel = spec[vel_dim].values
    vnyq = float(np.abs(vel).max())
    nfft = len(vel)
    vel_ext = np.concatenate([vel - 2 * vnyq, vel, vel + 2 * vnyq])

    vals = spec.values                        # (..., nfft)
    leading = vals.shape[:-1]
    flat_spec = vals.reshape(-1, nfft)
    flat_folds = n_folds.reshape(-1)

    out_flat = np.full((flat_spec.shape[0], 3 * nfft), np.nan, dtype=vals.dtype)
    for n in [int(x) for x in np.unique(flat_folds)]:
        start = (1 + n) * nfft
        end = start + nfft
        if 0 <= start and end <= 3 * nfft:
            idx = flat_folds == n
            out_flat[idx, start:end] = flat_spec[idx]

    out = out_flat.reshape(leading + (3 * nfft,))
    new_coords = {k: v for k, v in spec.coords.items() if k != vel_dim}
    new_coords[vel_dim] = vel_ext
    return xr.DataArray(out, coords=new_coords, dims=spec.dims, attrs=spec.attrs)


def finalize_metadata(out: xr.Dataset, *, backend_name: str, include_moments: bool, include_ldr: bool) -> xr.Dataset:
    out = out.copy()
    out.attrs["radar_backend"] = backend_name
    out.attrs["processing_schema"] = "radarviz_spectra_v1"
    out.attrs["includes_moments"] = str(bool(include_moments))
    out.attrs["includes_ldr"] = str(bool(include_ldr))
    return out
def add_standard_variable_attrs(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()

    meta = {
        # Co / Cross convention
        "sZeCo": {"long_name": "Doppler spectrum, co-polar channel", "units": "mm6 m-3 (m s-1)-1"},
        "sZeCx": {"long_name": "Doppler spectrum, cross-polar channel", "units": "mm6 m-3 (m s-1)-1"},
        "ZeCo": {"long_name": "Equivalent reflectivity factor, co-polar channel (integrated spectrum)", "units": "mm6 m-3"},
        "ZeCx": {"long_name": "Equivalent reflectivity factor, cross-polar channel (integrated spectrum)", "units": "mm6 m-3"},
        "MDV_Co": {"long_name": "Mean Doppler velocity, co-polar channel", "units": "m s-1"},
        "MDV_Cx": {"long_name": "Mean Doppler velocity, cross-polar channel", "units": "m s-1"},
        "WIDTH_Co": {"long_name": "Doppler spectral width, co-polar channel", "units": "m s-1"},
        "WIDTH_Cx": {"long_name": "Doppler spectral width, cross-polar channel", "units": "m s-1"},
        "NoisePowCo": {"long_name": "Estimated noise power, co-polar channel", "units": "mm6 m-3 (m s-1)-1"},
        "NoisePowCx": {"long_name": "Estimated noise power, cross-polar channel", "units": "mm6 m-3 (m s-1)-1"},
        "LDR": {"long_name": "Linear depolarization ratio", "units": "dB"},
        "sLDR": {"long_name": "Spectral linear depolarization ratio", "units": "dB"},

        # H / V convention for future radars
        "sZeH": {"long_name":  "Doppler spectrum, horizontal polarization", "units": "mm6 m-3 (m s-1)-1"},
        "sZeV": {"long_name": "Doppler spectrum, vertical polarization", "units": "mm6 m-3 (m s-1)-1"},
        "ZeH": {"long_name": "Equivalent reflectivity factor, horizontal polarization (integrated spectrum)", "units": "mm6 m-3"},
        "ZeV": {"long_name": "Equivalent reflectivity factor, vertical polarization (integrated spectrum)", "units": "mm6 m-3"},
        "MDV_H": {"long_name": "Mean Doppler velocity, horizontal polarization", "units": "m s-1"},
        "MDV_V": {"long_name": "Mean Doppler velocity, vertical polarization", "units": "m s-1"},
        "WIDTH_H": {"long_name": "Doppler spectral width, horizontal polarization", "units": "m s-1"},
        "WIDTH_V": {"long_name": "Doppler spectral width, vertical polarization", "units": "m s-1"},
        "NoisePowH": {"long_name": "Estimated noise power, horizontal polarization", "units": "mm6 m-3 (m s-1)-1"},
        "NoisePowV": {"long_name": "Estimated noise power, vertical polarization", "units": "mm6 m-3 (m s-1)-1"},
        "slanted_LDR": {"long_name": "slanted Linear Depolarization Ratio", "units": "dB"},
        "s_slanted_LDR": {"long_name": "Spectral slanted Linear Depolarization Ratio", "units": "dB"},
        "ZDR": {"long_name": "Differential reflectivity", "units": "dB"},
        "sZDR": {"long_name": "Spectral differential reflectivity", "units": "dB"},
        "KDP": {"long_name": "Specific differential phase", "units": "degrees km-1"},
        "SLDR": {"long_name": "Slanted Linear Depolarization Ratio", "units": "dB"},
        
    }

    for var, attrs in meta.items():
        if var in ds:
            ds[var].attrs.update(attrs)

    if "Vel" in ds.coords:
        ds["Vel"].attrs.update({"long_name": "Doppler velocity", "units": "m s-1"})
    if "range" in ds.coords:
        ds["range"].attrs.update({"long_name": "Range from radar", "units": "m"})
    if "time" in ds.coords:
        ds["time"].attrs.update({"long_name": "Time"})

    return ds
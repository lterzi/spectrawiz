from __future__ import annotations

"""
High-level processing entrypoints for radar raw data.
"""

from pathlib import Path
from typing import Literal
import glob
import re

import numpy as np
import pandas as pd
import xarray as xr

from .backends import (
    MetekBackend,
    RPGBackend,
    MRRBackend,
    available_backends,
    register_backend,
    select_backend,
)
from .processing_common import (
    dealias_mdv_range_continuity,
    dealias_mdv_from_reference,
    dealias_ka_with_phase_unwrap,
)
from .dealias_ridge import dealias_ka_ridge, dealiased_moments

RadarType = Literal["auto", "metek", "rpg", "mrr"]

register_backend(MetekBackend(), overwrite=True)
register_backend(RPGBackend(), overwrite=True)
register_backend(MRRBackend(), overwrite=True)


def detect_radar_type(ds: xr.Dataset) -> str:
    return select_backend(ds, radar_type="auto").name


def _normalize_time_name(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalize RPG-style 'Time' to 'time' and ensure it is a coordinate if possible.
    """
    if "time" in ds.coords:
        return ds

    if "Time" in ds.coords:
        return ds.rename({"Time": "time"})

    if "Time" in ds.variables:
        ds = ds.rename({"Time": "time"})
        if "time" in ds.variables and "time" not in ds.coords:
            ds = ds.set_coords("time")
    return ds


def process_raw_radar(
    file_or_ds: str | Path | xr.Dataset,
    radar_type: RadarType = "auto",
    velRef=None,
    include_moments: bool = True,
    include_ldr: bool = True,
    include_pol: bool = False,
) -> xr.Dataset:
    ds = xr.open_dataset(file_or_ds) if not isinstance(file_or_ds, xr.Dataset) else file_or_ds
    ds = _normalize_time_name(ds)
    backend = select_backend(ds, radar_type=radar_type)

    try:
        return backend.process(
            ds,
            velRef=velRef,
            include_moments=include_moments,
            include_ldr=include_ldr,
            include_pol=include_pol,
        )
    except TypeError:
        return backend.process(
            ds,
            velRef=velRef,
            include_moments=include_moments,
            include_ldr=include_ldr,
        )


def _vars_for_backend(backend_name: str) -> list[str]:
    if backend_name == "metek":
        return [
            "SPCco",
            "SPCcx",
            "SNRCorFaCo",
            "SNRCorFaCx",
            "doppler",
            "HSDco",
            "HSDcx",
            "npw1",
            "npw2",
            "RadarConst",
            "NyquistVelocity",
            "range",
            "time",
        ]
    return []


def _subset_for_backend(ds: xr.Dataset, backend_name: str, vars_to_keep: list[str]) -> xr.Dataset:
    """
    RPG keeps full dataset (dynamic C* variables needed later).
    Others use whitelist if available.
    """
    if backend_name == "rpg":
        return ds

    keep = [v for v in vars_to_keep if v in ds.variables or v in ds.coords]
    if not keep:
        return ds
    return ds[keep]


def _deduplicate_dim(ds: xr.Dataset, dim: str) -> xr.Dataset:
    if dim not in ds.dims:
        return ds
    if dim not in ds.coords:
        return ds
    vals = np.asarray(ds.coords[dim].values)
    _, idx = np.unique(vals, return_index=True)
    return ds.isel({dim: np.sort(idx)})


def _validate_raw_ds(
    ds: xr.Dataset,
    *,
    require_time: bool = True,
    max_doppler_bins: int = 16384,
) -> tuple[bool, str]:
    if require_time:
        has_time = (
            ("time" in ds.coords) or ("Time" in ds.coords) or
            ("time" in ds.variables) or ("Time" in ds.variables)
        )
        if not has_time:
            return False, "missing time"

        tdim = "time" if "time" in ds.sizes else ("Time" if "Time" in ds.sizes else None)
        if tdim is not None and int(ds.sizes.get(tdim, 0)) == 0:
            return False, "empty time dimension"

    if "doppler" in ds.dims:
        n_dop = int(ds.sizes.get("doppler", 0))
        if n_dop <= 0:
            return False, "empty doppler dimension"
        if n_dop > max_doppler_bins:
            return False, f"suspicious doppler size: {n_dop}"

    return True, ""


def _fix_time_units(ds: xr.Dataset, time_var: str = "time") -> xr.Dataset:
    ds = _normalize_time_name(ds)

    if time_var not in ds.coords and time_var in ds.variables:
        ds = ds.set_coords(time_var)
    if time_var not in ds.coords:
        return ds
    if np.issubdtype(ds[time_var].dtype, np.datetime64):
        return ds

    ds = ds.copy()
    units_raw = str(ds[time_var].attrs.get("units", ds[time_var].attrs.get("Units", ""))).strip()

    unit_hits = re.findall(r"([A-Za-z]+)\s+since", units_raw, flags=re.IGNORECASE)
    unit_token = unit_hits[-1].lower() if unit_hits else "seconds"
    unit_map = {
        "s": "seconds", "sec": "seconds", "secs": "seconds", "second": "seconds", "seconds": "seconds",
        "ms": "milliseconds", "millisecond": "milliseconds", "milliseconds": "milliseconds",
        "us": "microseconds", "microsecond": "microseconds", "microseconds": "microseconds",
        "ns": "nanoseconds", "nanosecond": "nanoseconds", "nanoseconds": "nanoseconds",
        "m": "minutes", "min": "minutes", "minute": "minutes", "minutes": "minutes",
        "h": "hours", "hr": "hours", "hour": "hours", "hours": "hours",
        "d": "days", "day": "days", "days": "days",
    }
    unit_cf = unit_map.get(unit_token, "seconds")

    # ISO-like: YYYY-MM-DD or YYYYMMDD
    m_iso = re.search(
        r"(\d{4})-?(\d{2})-?(\d{2})(?:[ T](\d{2}):?(\d{2}):?(\d{2}))?",
        units_raw,
    )
    if m_iso:
        y, mo, d = m_iso.group(1), m_iso.group(2), m_iso.group(3)
        hh = m_iso.group(4) or "00"
        mm = m_iso.group(5) or "00"
        ss = m_iso.group(6) or "00"
        epoch = f"{y}-{mo}-{d} {hh}:{mm}:{ss}"
    else:
        # US-like: M/D/YYYY HH:MM:SS
        m_us = re.search(
            r"(\d{1,2})/(\d{1,2})/(\d{4})(?:[ T](\d{2}):(\d{2}):(\d{2}))?",
            units_raw,
        )
        if m_us:
            mo, d, y = int(m_us.group(1)), int(m_us.group(2)), int(m_us.group(3))
            hh = m_us.group(4) or "00"
            mm = m_us.group(5) or "00"
            ss = m_us.group(6) or "00"
            epoch = f"{y:04d}-{mo:02d}-{d:02d} {hh}:{mm}:{ss}"
        else:
            epoch = "1970-01-01 00:00:00"

    ds[time_var].attrs["units"] = f"{unit_cf} since {epoch}"

    try:
        ds = xr.decode_cf(ds)
    except Exception:
        return ds
    return ds


def _extend_spectrum_chunk(
    spec_chunk: np.ndarray,  # (time_chunk, range, nfft)
    folds_chunk: np.ndarray,  # (time_chunk, range, nfft) int8 -- per-bin fold count
    nfft: int,
    n_vel_ext: int,
    min_fold: int,  # smallest fold value present anywhere in the data (block 0 of vel_ext)
) -> np.ndarray:
    """
    Place each Doppler bin at its dealiased position on the extended velocity axis.

    folds_chunk is shaped per *bin*, not per gate, though every current dealiasing method
    (range-continuity, moment-reference, phase-unwrap) decides one fold per whole gate and
    the caller broadcasts it across that gate's bins before calling this -- kept per-bin
    here in case a future method ever needs to place different bins of one gate at
    different positions (e.g. a population straddling ±vnyq).

    vel_ext (built by the caller) is exactly `(max_fold - min_fold + 1) * nfft` long,
    covering only the fold range actually present -- block index for fold n is
    `n - min_fold`, not a fixed offset into a fixed-size axis. An earlier version
    assumed exactly 3 possible folds (n in {-1, 0, 1}); once a method can produce
    folds outside that range (as the spectrum-matched one can), that assumption
    silently truncated the output array instead of raising, corrupting the write.
    """
    nt, nrange, nv = spec_chunk.shape
    v_idx = np.arange(nfft)[None, None, :]
    # int8 (plenty for a fold count) overflows immediately once multiplied by nfft
    # (a few hundred/thousand) -- widen before the arithmetic, not just at the end.
    target = (folds_chunk.astype(np.int32) - min_fold) * nfft + v_idx
    out = np.full((nt, nrange, n_vel_ext), np.nan, dtype=np.float32)
    valid = (target >= 0) & (target < n_vel_ext)
    ti, ri, _vi = np.nonzero(valid)
    out[ti, ri, target[valid]] = spec_chunk[valid]
    return out


def _write_ds_with_dealias(
    ds: xr.Dataset,
    path: Path,
    compress: bool = True,
    chunk_size: int = 50,
) -> None:
    """
    Write dataset to NetCDF.  When n_folds_* variables are present the
    Vel-dependent spectra are written on the extended velocity axis in
    time chunks so peak memory ≈ chunk_size × range × n_vel_ext per variable.
    """
    import netCDF4 as nc4

    # Identify channels that were dealiased
    channels = {}  # "Co" -> int8 array (time, range)
    for v in list(ds.data_vars):
        if v.startswith("n_folds_"):
            ch = v[len("n_folds_"):]
            channels[ch] = ds[v].values.astype(np.int8)

    if not channels:
        _write_ds(ds, path, compress=compress)
        return

    vel = ds["Vel"].values
    vnyq = float(np.abs(vel).max())
    nfft = len(vel)

    # Determine the fold range actually present → smallest possible Vel axis that
    # still covers every fold seen (not hardcoded to n in {-1,0,1} -- the spectrum-
    # matched method can produce folds well outside that range in strong convection).
    all_folds = set()
    for arr in channels.values():
        all_folds.update(int(f) for f in np.unique(arr))
    min_fold = min(all_folds)
    max_fold = max(all_folds)
    n_vel_ext = (max_fold - min_fold + 1) * nfft
    vel_ext = np.concatenate([vel + n * 2 * vnyq for n in range(min_fold, max_fold + 1)])

    # Sort and deduplicate time before anything else
    ds = _fix_time_units(ds, time_var="time")
    if "time" in ds.coords:
        ds = ds.sortby("time")
        _, idx = np.unique(ds["time"].values, return_index=True)
        ds = ds.isel(time=np.sort(idx))
        # refresh channels arrays after time reindex
        for ch in list(channels):
            channels[ch] = ds[f"n_folds_{ch}"].values.astype(np.int8)

    n_time = ds.sizes["time"]
    n_range = ds.sizes["range"]

    # Separate variables: non-Vel (write with xarray), Vel-dep (write chunked)
    vel_dep_vars = {}  # spec_var -> channel key
    non_vel_data_vars = []
    for v in ds.data_vars:
        if v.startswith("n_folds_"):
            continue  # implementation detail, not written to output
        if "Vel" not in ds[v].dims:
            non_vel_data_vars.append(v)
        else:
            # Match channel by name (e.g. "Cx" in "sZeCx")
            matched = next((ch for ch in channels if ch in v), next(iter(channels)))
            vel_dep_vars[v] = matched

    # Step 1 — write all non-Vel variables with xarray
    path.parent.mkdir(parents=True, exist_ok=True)
    ds_simple = ds[non_vel_data_vars]
    enc = {v: {"zlib": True, "complevel": 4} for v in ds_simple.data_vars} if compress else {}
    ds_simple.to_netcdf(path, mode="w", encoding=enc, unlimited_dims=["time"])

    # Step 2 — append Vel-dependent variables via netCDF4 in time chunks
    with nc4.Dataset(path, "a") as ncf:
        ncf.createDimension("Vel", n_vel_ext)
        vel_nc = ncf.createVariable("Vel", "f4", ("Vel",))
        vel_nc[:] = vel_ext.astype(np.float32)
        for k, val in ds["Vel"].attrs.items():
            setattr(vel_nc, k, val)

        spec_nc_vars = {}
        for v, ch in vel_dep_vars.items():
            attrs = {k: val for k, val in ds[v].attrs.items() if k != "_FillValue"}
            kw = {"zlib": compress, "complevel": 4} if compress else {}
            vn = ncf.createVariable(v, "f4", ("time", "range", "Vel"),
                                    fill_value=np.nan, **kw)
            for k, val in attrs.items():
                try:
                    setattr(vn, k, val)
                except Exception:
                    pass
            spec_nc_vars[v] = (vn, ch)

        for t0 in range(0, n_time, chunk_size):
            t1 = min(t0 + chunk_size, n_time)
            for v, (vn, ch) in spec_nc_vars.items():
                spec_chunk = ds[v].isel(time=slice(t0, t1)).values.astype(np.float32)
                folds_chunk = channels[ch][t0:t1]
                ext = _extend_spectrum_chunk(spec_chunk, folds_chunk, nfft, n_vel_ext, min_fold)
                vn[t0:t1, :, :] = ext


def _write_ds(ds: xr.Dataset, path: Path, compress: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ds = _fix_time_units(ds, time_var="time")

    if "time" in ds.coords:
        ds = ds.sortby("time")
        _, idx = np.unique(ds["time"].values, return_index=True)
        ds = ds.isel(time=np.sort(idx))

    encoding = {v: {"zlib": True} for v in ds.data_vars} if compress else {}
    ds.to_netcdf(path, mode="w", encoding=encoding)


def _filter_files_by_date(
    files: list[str],
    start: str | None,
    end: str | None,
) -> list[str]:
    """Filter a file list to those whose filename contains a YYYYMMDD date within [start, end]."""
    if start is None and end is None:
        return files
    date_re = re.compile(r"(\d{8})")
    t_start = pd.Timestamp(start) if start else None
    t_end = pd.Timestamp(end) if end else None
    result = []
    for f in files:
        m = date_re.search(Path(f).name)
        if not m:
            result.append(f)
            continue
        try:
            file_date = pd.Timestamp(m.group(1))
        except Exception:
            result.append(f)
            continue
        if t_start is not None and file_date < t_start:
            continue
        if t_end is not None and file_date > t_end:
            continue
        result.append(f)
    return result


def _load_preprocessing_datasets(
    preprocessing_path: str | Path | None,
    preprocessing_pattern: str = "*.nc",
    preprocessing_start: str | None = None,
    preprocessing_end: str | None = None,
    debugging: bool = False,
) -> list[xr.Dataset]:
    """Load datasets from preprocessing_path to use for campaign-wide preprocessing."""
    if preprocessing_path is None:
        return []
    prep_files = sorted(glob.glob(str(Path(preprocessing_path) / "**" / preprocessing_pattern), recursive=True))
    prep_files = _filter_files_by_date(prep_files, preprocessing_start, preprocessing_end)
    if debugging:
        print(f"Preprocessing: found {len(prep_files)} files in {preprocessing_path}"
              + (f" between {preprocessing_start} and {preprocessing_end}" if preprocessing_start or preprocessing_end else ""))
    datasets = []
    for f in prep_files:
        try:
            ds = xr.open_dataset(f)
            ds = _normalize_time_name(ds)
            datasets.append(ds)
        except Exception as e:
            if debugging:
                print(f"Warning: could not read preprocessing file {Path(f).name}: {e}")
    return datasets


def _merge_process_regrid_hour(
    hour_files: list[str],
    *,
    hour: pd.Timestamp,
    backend,
    vars_to_keep: list[str],
    velRef=None,
    include_moments: bool = True,
    include_ldr: bool = True,
    include_pol: bool = False,
    regrid_time: bool = True,
    time_step: str = "4s",
    regrid_tolerance: str | None = None,
    preprocessing_datasets: list[xr.Dataset] | None = None,
    preprocessing_spectra_q=None,
    debugging: bool = False,
    dealias_method: str = "none",
    dealias_ref: xr.DataArray | None = None,
    dealias_xband: xr.Dataset | None = None,
    dealias_n_jobs: int = 1,
    time_start: pd.Timestamp | None = None,
    time_end: pd.Timestamp | None = None,
) -> xr.Dataset | None:
    parts: list[xr.Dataset] = []
    hour = pd.Timestamp(hour)
    hour_end = hour + pd.Timedelta(hours=1)
    # optional sub-hour time window (--start_time/--end_time)
    win_start = max(hour, pd.Timestamp(time_start)) if time_start is not None else hour
    win_end = min(hour_end, pd.Timestamp(time_end)) if time_end is not None else hour_end

    for f in hour_files:
        try:
            with xr.open_dataset(f) as data_tmp:
                data_tmp = _normalize_time_name(data_tmp)

                ok, reason = _validate_raw_ds(data_tmp)
                if not ok:
                    if debugging:
                        print(f"Skipping {Path(f).name}: {reason}")
                    continue

                part = _subset_for_backend(data_tmp, backend.name, vars_to_keep)
                part = _fix_time_units(part, time_var="time")

                # Keep only requested hour when datetime decoding is available
                if "time" in part.coords and np.issubdtype(part["time"].dtype, np.datetime64):
                    part = part.sel(time=slice(win_start, win_end))
                    if int(part.sizes.get("time", 0)) == 0:
                        continue

                part = _deduplicate_dim(part, "time")
                part = part.load()

            parts.append(part)

        except Exception as e:
            if debugging:
                print(f"Warning: could not read {Path(f).name}: {e}")
            continue

    if not parts:
        return None

    ds_hour = xr.concat(parts, dim="time", data_vars="minimal", coords="minimal", compat="override")
    ds_hour = _fix_time_units(ds_hour, time_var="time")
    if "time" in ds_hour.coords:
        ds_hour = ds_hour.sortby("time")
        ds_hour = _deduplicate_dim(ds_hour, "time")

    try:
        ds_proc = backend.process(
            ds_hour,
            velRef=velRef,
            include_moments=include_moments,
            include_ldr=include_ldr,
            include_pol=include_pol,
            preprocessing_datasets=preprocessing_datasets or [],
            preprocessing_spectra_q=preprocessing_spectra_q,
        )
    except TypeError:
        ds_proc = backend.process(
            ds_hour,
            velRef=velRef,
            include_moments=include_moments,
            include_ldr=include_ldr,
        )

    if regrid_time and "time" in ds_proc.coords and ds_proc.sizes.get("time", 0) > 0:
        ds_proc = _fix_time_units(ds_proc, time_var="time")
        if np.issubdtype(ds_proc["time"].dtype, np.datetime64):
            ds_proc = ds_proc.sortby("time")
            ds_proc = _deduplicate_dim(ds_proc, "time")

            grid = pd.date_range(win_start, win_end, freq=time_step, inclusive="left")
            tol = pd.Timedelta(regrid_tolerance) if regrid_tolerance else (pd.Timedelta(time_step) / 2)
            ds_proc = ds_proc.reindex({"time": grid}, method="nearest", tolerance=tol)

    if dealias_method == "ridge" and "Vel" in ds_proc.coords:
        # Spectral ridge dealiasing anchored to X-band spectra (dealias_ridge.py,
        # docs/dealiasing_ka_ridge.md). Produces per-bin folds n_folds_Co, applied by the
        # writer to every Vel-dependent variable (sZeCo, sZeCx, sLDR, ...).
        if dealias_xband is None or dealias_xband.sizes.get("time", 0) == 0:
            print(f"No X-band spectra for {hour}, skipping ridge dealiasing")
            return ds_proc
        vnyq_raw = float(ds_hour["NyquistVelocity"].values) if "NyquistVelocity" in ds_hour else None
        ka_in = ds_proc.assign_attrs(vnyq=vnyq_raw) if vnyq_raw is not None else ds_proc
        if debugging:
            print(f"Ridge dealiasing {ds_proc.sizes['time']} time steps (n_jobs={dealias_n_jobs})")
        res = dealias_ka_ridge(ka_in, dealias_xband, n_jobs=dealias_n_jobs)
        ds_proc = xr.merge([ds_proc, res], compat="override")
        vnyq = vnyq_raw if vnyq_raw is not None else float(res.attrs.get("vnyq", ds_proc["Vel"].max()))
        for ch in ("Co", "Cx"):
            if f"sZe{ch}" in ds_proc and f"MDV_{ch}" in ds_proc:
                mdv, wid = dealiased_moments(ds_proc[f"sZe{ch}"], ds_proc["n_folds_Co"].values, vnyq)
                ds_proc[f"MDV_{ch}"] = mdv.assign_attrs(ds_proc[f"MDV_{ch}"].attrs)
                if f"WIDTH_{ch}" in ds_proc:
                    ds_proc[f"WIDTH_{ch}"] = wid.assign_attrs(ds_proc[f"WIDTH_{ch}"].attrs)
        ds_proc.attrs["dealias_method"] = "ridge"
        return ds_proc

    if dealias_method != "none" and "Vel" in ds_proc.coords:
        vnyq = float(ds_proc["Vel"].max())
        mdv_vars = [v for v in ds_proc.data_vars if v.startswith("MDV_")]
        nfft = ds_proc.sizes["Vel"]

        for var in mdv_vars:
            mdv_orig = ds_proc[var].copy()
            if dealias_method == "range":
                ds_proc[var] = dealias_mdv_range_continuity(ds_proc[var], vnyq=vnyq)
            elif dealias_method in ("reference", "phase_unwrap"):
                if dealias_ref is None:
                    if debugging:
                        print("No reference data loaded, skipping dealiasing")
                    continue
                if debugging:
                    print(f"Dealiasing {var} using {dealias_method}")
                if dealias_method == "reference":
                    ds_proc[var] = dealias_mdv_from_reference(
                        ds_proc[var],
                        ref=dealias_ref,
                        vnyq=vnyq,
                    )
                else:
                    ds_proc[var] = dealias_ka_with_phase_unwrap(
                        ds_proc[var],
                        ref=dealias_ref,
                        vnyq=vnyq,
                    )
            n_folds_float = np.round((ds_proc[var].values - mdv_orig.values) / (2.0 * vnyq))
            n_folds = np.where(np.isfinite(n_folds_float), n_folds_float, 0).astype(np.int8)
            channel = var.replace("MDV_", "")  # "Co" or "Cx"
            # Broadcast to per-bin (whole gate shifts uniformly) so the writer's per-bin
            # placement logic (see _extend_spectrum_chunk) can treat every method uniformly.
            n_folds_per_bin = np.broadcast_to(n_folds[..., None], n_folds.shape + (nfft,)).copy()
            ds_proc[f"n_folds_{channel}"] = xr.DataArray(
                n_folds_per_bin,
                dims=ds_proc[var].dims + ("Vel",),
                coords={"Vel": ds_proc["Vel"]},
                attrs={"long_name": f"Doppler aliasing fold count ({channel} channel)", "units": "1"},
            )

    return ds_proc


def _load_xband_spectra(files: list[str], t_start, t_end, debugging: bool = False,
                        margin: str = "60s") -> xr.Dataset | None:
    """Processed X-band spectra (sZeCo) for [t_start - margin, t_end + margin].
    Each file is cut to that window BEFORE backend processing (whole-file processing of
    the X/Ka spectra needs several GB)."""
    lo, hi = pd.Timestamp(t_start) - pd.Timedelta(margin), pd.Timestamp(t_end) + pd.Timedelta(margin)
    parts = []
    for f in files:
        with xr.open_dataset(f) as dsx:
            dsx = _fix_time_units(_normalize_time_name(dsx), time_var="time")
            if "time" not in dsx.coords or not np.issubdtype(dsx["time"].dtype, np.datetime64):
                continue
            dsx = dsx.sel(time=slice(lo, hi))
            if dsx.sizes.get("time", 0) == 0:
                continue
            dsx = dsx.load()
        px = select_backend(dsx, radar_type="auto").process(dsx, include_moments=False, include_ldr=False)
        parts.append(px[["sZeCo"]])
        if debugging:
            print(f"X-band spectra: {Path(f).name}, {px.sizes['time']} time steps")
    if not parts:
        return None
    out = xr.concat(parts, dim="time").sortby("time")
    return _deduplicate_dim(out, "time")


def process_day(
    date,
    raw_path: str | Path,
    output_dir: str | Path,
    file_pattern: str = "*.nc",
    radar_type: RadarType = "auto",
    velRef=None,
    include_moments: bool = True,
    include_ldr: bool = True,
    include_pol: bool = False,
    compress: bool = True,
    overwrite: bool = False,
    hourly: bool = True,
    regrid_time: bool = True,
    time_step: str = "4s",
    regrid_tolerance: str | None = None,
    preprocessing_path: str | Path | None = None,
    preprocessing_pattern: str = "*.nc",
    preprocessing_start: str | None = None,
    preprocessing_end: str | None = None,
    debugging: bool = False,
    dealias_method: str = "none",
    dealias_ref_pattern: str | None = None,
    dealias_ref_var: str = "MDV_Co",
    dealias_n_jobs: int = 1,
    start_time: str | None = None,
    end_time: str | None = None,
) -> list[Path]:
    date2proc = pd.Timestamp(date).normalize()
    raw_path = Path(raw_path)
    out_dir = Path(output_dir)

    all_files = sorted(glob.glob(str(raw_path / file_pattern)))
    if not all_files:
        raise FileNotFoundError(f"No files found matching {raw_path / file_pattern}")

    backend = None
    for f in all_files:
        try:
            with xr.open_dataset(f) as ds_probe:
                ds_probe = _normalize_time_name(ds_probe)
                ok, _ = _validate_raw_ds(ds_probe)
                if not ok:
                    continue
                backend = select_backend(ds_probe, radar_type=radar_type)
                break
        except Exception:
            continue

    if backend is None:
        raise ValueError(f"Could not detect backend for valid files in {raw_path}")

    vars_to_keep = _vars_for_backend(backend.name)
    prep_datasets = _load_preprocessing_datasets(
        preprocessing_path, preprocessing_pattern, preprocessing_start, preprocessing_end, debugging
    )
    # Pre-compute spectra_q once so it isn't recomputed on every backend.process() call.
    preprocessing_spectra_q = None
    if prep_datasets and hasattr(backend, "compute_preprocessing_spectra_q"):
        if debugging:
            print("Preprocessing: computing spectra_q from preprocessing files (once)...")
        preprocessing_spectra_q = backend.compute_preprocessing_spectra_q(prep_datasets)

    written: list[Path] = []

    if not hourly:
        for f in all_files:
            out_path = out_dir / f"{Path(f).stem}_proc.nc"

            if not overwrite and out_path.exists():
                if debugging:
                    print(f"Skipping {Path(f).name} — already exists")
                written.append(out_path)
                continue

            if debugging:
                print(f"Processing {Path(f).name}")

            try:
                with xr.open_dataset(f) as ds_in:
                    ds_in = _normalize_time_name(ds_in)

                    ok, reason = _validate_raw_ds(ds_in)
                    if not ok:
                        if debugging:
                            print(f"Skipping {Path(f).name}: {reason}")
                        continue

                    ds_raw = _subset_for_backend(ds_in, backend.name, vars_to_keep)
                    ds_raw = _deduplicate_dim(ds_raw, "doppler")
                    ds_raw = _deduplicate_dim(ds_raw, "time")

                    try:
                        ds_proc = backend.process(
                            ds_raw,
                            velRef=velRef,
                            include_moments=include_moments,
                            include_ldr=include_ldr,
                            include_pol=include_pol,
                            preprocessing_datasets=prep_datasets,
                            preprocessing_spectra_q=preprocessing_spectra_q,
                        )
                    except TypeError:
                        ds_proc = backend.process(
                            ds_raw,
                            velRef=velRef,
                            include_moments=include_moments,
                            include_ldr=include_ldr,
                        )

                _write_ds(ds_proc, out_path, compress=compress)
                written.append(out_path)
                if debugging:
                    print(f"Saved {out_path}")

            except Exception as e:
                print(f"Failed {Path(f).name}: {e}")

        return written

    dealias_ref_da: xr.DataArray | None = None
    if dealias_method in ("reference", "phase_unwrap"):
        if not dealias_ref_pattern:
            raise ValueError(f"--dealias_ref_pattern is required when --dealias_method={dealias_method}")
        ref_files = sorted(glob.glob(dealias_ref_pattern))
        if not ref_files:
            raise FileNotFoundError(f"No files found matching dealias_ref_pattern: {dealias_ref_pattern}")
        if debugging:
            print(f"Loading reference for dealiasing: {len(ref_files)} file(s) matching '{dealias_ref_pattern}'")
        ref_parts = []
        for rf in ref_files:
            with xr.open_dataset(rf) as ds_ref:
                ds_ref = _normalize_time_name(ds_ref)
                ds_ref = _fix_time_units(ds_ref, time_var="time")
                if dealias_ref_var not in ds_ref:
                    raise KeyError(f"Variable '{dealias_ref_var}' not found in {rf}. Available: {list(ds_ref.data_vars)}")
                ref_parts.append(ds_ref[[dealias_ref_var]].load())
        dealias_ref_da = xr.concat(ref_parts, dim="time")[dealias_ref_var]

    xband_files: list[str] = []
    if dealias_method == "ridge":
        if not dealias_ref_pattern:
            raise ValueError("--dealias_ref_pattern (raw X-band spectra files) is required when --dealias_method=ridge")
        xband_files = sorted(glob.glob(dealias_ref_pattern))
        if not xband_files:
            raise FileNotFoundError(f"No files found matching dealias_ref_pattern: {dealias_ref_pattern}")

    def _tod(t):  # "HH:MM" or "HH:MM:SS" -> Timedelta
        return pd.Timedelta(t if t.count(":") == 2 else f"{t}:00")

    t_win_start = date2proc + _tod(start_time) if start_time else None
    t_win_end = date2proc + _tod(end_time) if end_time else None

    time_hours = pd.date_range(date2proc, date2proc + pd.Timedelta(hours=24), freq="1h", inclusive="left")

    for hour in time_hours:
        backend_tag = backend.name.lower()
        out_path = out_dir / f"{hour:%Y%m%d_%H}_{backend_tag}_hourly_proc.nc"
        h_start = max(hour, t_win_start) if t_win_start is not None else hour
        h_end = min(hour + pd.Timedelta(hours=1), t_win_end) if t_win_end is not None else hour + pd.Timedelta(hours=1)
        if h_start >= h_end:
            continue
        if t_win_start is not None or t_win_end is not None:
            out_path = out_dir / f"{hour:%Y%m%d_%H}_{backend_tag}_hourly_proc_{h_start:%H%M%S}-{h_end:%H%M%S}.nc"

        if not overwrite and out_path.exists():
            if debugging:
                print(f"Skipping {hour} — already exists")
            written.append(out_path)
            continue

        hour_files = [
            f for f in all_files
            if (
                hour.strftime("%Y%m%d_%H") in Path(f).name
                or hour.strftime("%Y%m%d%H") in Path(f).name
                or hour.strftime("%y%m%d_%H") in Path(f).name
                or hour.strftime("%y%m%d%H") in Path(f).name
            )
        ]

        if debugging:
            print(f"Processing {hour} — {len(hour_files)} files")

        if not hour_files:
            continue

        try:
            ds_proc = _merge_process_regrid_hour(
                hour_files,
                hour=hour,
                backend=backend,
                vars_to_keep=vars_to_keep,
                velRef=velRef,
                include_moments=include_moments,
                include_ldr=include_ldr,
                include_pol=include_pol,
                regrid_time=regrid_time,
                time_step=time_step,
                regrid_tolerance=regrid_tolerance,
                preprocessing_datasets=prep_datasets,
                preprocessing_spectra_q=preprocessing_spectra_q,
                debugging=debugging,
                dealias_method=dealias_method,
                dealias_ref=dealias_ref_da,
                dealias_xband=(_load_xband_spectra(xband_files, h_start, h_end, debugging)
                               if dealias_method == "ridge" else None),
                dealias_n_jobs=dealias_n_jobs,
                time_start=t_win_start,
                time_end=t_win_end,
            )

            if ds_proc is None:
                if debugging:
                    print(f"Empty dataset for {hour}, skipping")
                continue

            if dealias_method != "none":
                _write_ds_with_dealias(ds_proc, out_path, compress=compress)
            else:
                _write_ds(ds_proc, out_path, compress=compress)
            written.append(out_path)

            if debugging:
                print(f"Saved {out_path}")

        except Exception as e:
            print(f"Processing failed for {hour}: {e}")

    return written


def ensure_day_processed(
    date,
    *,
    raw_path: str | Path,
    output_dir: str | Path,
    file_pattern: str = "*.nc",
    processed_pattern: str = "*_proc.nc",
    **process_kwargs,
) -> list[Path]:
    out_dir = Path(output_dir)
    existing = sorted(out_dir.glob(processed_pattern))
    if existing:
        return existing

    return process_day(
        date=date,
        raw_path=raw_path,
        output_dir=output_dir,
        file_pattern=file_pattern,
        **process_kwargs,
    )


__all__ = [
    "available_backends",
    "detect_radar_type",
    "process_raw_radar",
    "process_day",
    "ensure_day_processed",
]

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process radar files for a day.")
    parser.add_argument("--date", required=True, help="Date to process (YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument("--raw_path", required=True, help="Path to raw radar files")
    parser.add_argument("--output_dir", required=True, help="Directory to write processed files")
    parser.add_argument("--file_pattern", default="*.nc", help="Glob pattern for raw files")
    parser.add_argument("--radar_type", default="auto", choices=["auto", "metek", "rpg", "mrr"])
    parser.add_argument("--include_moments", action="store_true")
    parser.add_argument("--include_ldr", action="store_true")
    parser.add_argument("--include_pol", action="store_true")
    parser.add_argument("--hourly", action="store_true")
    parser.add_argument("--regrid_time", action="store_true")
    parser.add_argument("--time_step", default="4s")
    parser.add_argument("--regrid_tolerance", default=None)
    parser.add_argument(
        "--preprocessing_path", default=None,
        help="Directory of raw files to use for campaign-wide preprocessing (MRR backend only). "
             "When set, the interference mask and border correction are estimated from all files "
             "in this directory rather than from the current file only. "
             "Subdirectories are searched recursively. "
             "Tip: use clear-sky files to avoid precipitation contaminating the interference mask.",
    )
    parser.add_argument(
        "--preprocessing_pattern", default="*.nc",
        help="Glob pattern for files in --preprocessing_path (default: *.nc)",
    )
    parser.add_argument(
        "--preprocessing_start", default=None,
        help="Start date (YYYYMMDD) for filtering files in --preprocessing_path. "
             "Files are matched by the YYYYMMDD date string in their filename.",
    )
    parser.add_argument(
        "--preprocessing_end", default=None,
        help="End date (YYYYMMDD) for filtering files in --preprocessing_path.",
    )
    parser.add_argument("--debugging", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dealias_method", default="none",
        choices=["none", "range", "reference", "phase_unwrap", "ridge"],
        help="Dealiasing method: 'range' for range-continuity unwrapping on MDV; "
             "'reference' to compare MDV directly against an external radar's MDV "
             "(unreliable once Mie scattering makes the two wavelengths' reflectivity-"
             "weighted means diverge); 'phase_unwrap' (recommended) to 2D-unwrap the Ka "
             "MDV profile along range (treating it as a wrapped phase) and anchor each "
             "unwrapped segment's absolute fold using an external wide-Nyquist radar's "
             "MDV, preferentially at range >= 2000 m where that reference is reliable "
             "(requires --dealias_ref_pattern pointing at that radar's processed files); "
             "'ridge' for spectral ridge dealiasing of Metek Ka spectra anchored to X-band "
             "spectra (--dealias_ref_pattern = RAW X-band spectra files, see "
             "docs/dealiasing_ka_ridge.md)",
    )
    parser.add_argument("--dealias_n_jobs", type=int, default=1,
                        help="Parallel processes for --dealias_method=ridge (default 1)")
    parser.add_argument("--start_time", default=None,
                        help="Only process from this time of day (e.g. 18:00 or 18:00:00), hourly mode")
    parser.add_argument("--end_time", default=None,
                        help="Only process until this time of day (e.g. 18:10), hourly mode")
    parser.add_argument(
        "--dealias_ref_pattern", default=None,
        help="Glob pattern for reference files used when --dealias_method=reference or "
             "phase_unwrap (e.g. '/data/xmacs/20250604*.nc'). All matching files are "
             "concatenated along time and read as-is for the variable named by "
             "--dealias_ref_var.",
    )
    parser.add_argument(
        "--dealias_ref_var", default="MDV_Co",
        help="Variable name for the reference velocity in the reference files, used for "
             "--dealias_method=reference or phase_unwrap (default: MDV_Co)",
    )
    args = parser.parse_args()

    written = process_day(
        date=args.date,
        raw_path=args.raw_path,
        output_dir=args.output_dir,
        file_pattern=args.file_pattern,
        radar_type=args.radar_type,
        include_moments=args.include_moments,
        include_ldr=args.include_ldr,
        include_pol=args.include_pol,
        hourly=args.hourly,
        regrid_time=args.regrid_time,
        time_step=args.time_step,
        regrid_tolerance=args.regrid_tolerance,
        preprocessing_path=args.preprocessing_path,
        preprocessing_pattern=args.preprocessing_pattern,
        preprocessing_start=args.preprocessing_start,
        preprocessing_end=args.preprocessing_end,
        debugging=args.debugging,
        overwrite=args.overwrite,
        dealias_method=args.dealias_method,
        dealias_ref_pattern=args.dealias_ref_pattern,
        dealias_ref_var=args.dealias_ref_var,
        dealias_n_jobs=args.dealias_n_jobs,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    for path in written:
        print(f"✓ {path}")

if __name__ == "__main__":
    main()
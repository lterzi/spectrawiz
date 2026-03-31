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
    available_backends,
    register_backend,
    select_backend,
)

RadarType = Literal["auto", "metek", "rpg"]

register_backend(MetekBackend(), overwrite=True)
register_backend(RPGBackend(), overwrite=True)


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


def _write_ds(ds: xr.Dataset, path: Path, compress: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ds = _fix_time_units(ds, time_var="time")

    if "time" in ds.coords:
        ds = ds.sortby("time")
        _, idx = np.unique(ds["time"].values, return_index=True)
        ds = ds.isel(time=np.sort(idx))

    encoding = {v: {"zlib": True} for v in ds.data_vars} if compress else {}
    ds.to_netcdf(path, mode="w", encoding=encoding)


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
    debugging: bool = False,
) -> xr.Dataset | None:
    parts: list[xr.Dataset] = []
    hour = pd.Timestamp(hour)
    hour_end = hour + pd.Timedelta(hours=1)

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
                    part = part.sel(time=slice(hour, hour_end))
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

            grid = pd.date_range(hour, hour_end, freq=time_step, inclusive="left")
            tol = pd.Timedelta(regrid_tolerance) if regrid_tolerance else (pd.Timedelta(time_step) / 2)
            ds_proc = ds_proc.reindex({"time": grid}, method="nearest", tolerance=tol)

    return ds_proc


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
    debugging: bool = False,
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

    time_hours = pd.date_range(date2proc, date2proc + pd.Timedelta(hours=24), freq="1h", inclusive="left")

    for hour in time_hours:
        backend_tag = backend.name.lower()
        out_path = out_dir / f"{hour:%Y%m%d_%H}_{backend_tag}_hourly_proc.nc"

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
                debugging=debugging,
            )

            if ds_proc is None:
                if debugging:
                    print(f"Empty dataset for {hour}, skipping")
                continue

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
    parser.add_argument("--radar_type", default="auto", choices=["auto", "metek", "rpg"])
    parser.add_argument("--include_moments", action="store_true")
    parser.add_argument("--include_ldr", action="store_true")
    parser.add_argument("--include_pol", action="store_true")
    parser.add_argument("--hourly", action="store_true")
    parser.add_argument("--regrid_time", action="store_true")
    parser.add_argument("--time_step", default="4s")
    parser.add_argument("--regrid_tolerance", default=None)
    parser.add_argument("--debugging", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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
        debugging=args.debugging,
        overwrite=args.overwrite,
    )
    for path in written:
        print(f"✓ {path}")

if __name__ == "__main__":
    main()
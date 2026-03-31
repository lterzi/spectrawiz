from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .base import RadarBackend
from ..processing_common import (
    add_standard_variable_attrs,
    as_vel_ref,
    compute_spectral_moments,
    decode_time_if_needed,
    ensure_datetime_time,
    finalize_metadata,
)


def _scalar_from_var(v: Any, idx: int = 0, default: float | int | None = None):
    """
    Return a scalar value from scalar/array-like input.

    - Handles scalar, 1D/ND arrays, and empty values.
    - Uses index `idx` for 1D-like arrays when available.
    - Falls back to first element or `default`.
    """
    if v is None:
        return default
    arr = np.asarray(v)
    if arr.size == 0:
        return default
    if arr.ndim == 0:
        return arr.item()
    if idx < arr.shape[0]:
        return arr[idx].item()
    return arr.flat[0].item()


def _chirp_num(ds: xr.Dataset) -> int:
    """
    Read number of chirps from `ChirpNum`.

    Returns 0 if the field is missing or empty.
    """
    if "ChirpNum" not in ds:
        return 0
    c = np.asarray(ds["ChirpNum"].values)
    if c.size == 0:
        return 0
    return int(c.flat[0])


def _first_existing(ds: xr.Dataset, names: list[str]) -> str | None:
    """
    Return the first existing variable/coordinate name from `names`.
    """
    for n in names:
        if n in ds.variables or n in ds.coords:
            return n
    return None


def _merge_chirps_spec(
    ds: xr.Dataset,
    velRef=None,
    include_moments: bool = True,
    include_pol: bool = False,
) -> xr.Dataset:
    """
    Merge RPG chirp spectra (LV0) into a single dataset on common `range` and `Vel`.

    Steps:
    1) Read chirp count and per-chirp velocity metadata (MaxVel, DoppLen).
    2) Build a common reference velocity grid (`vel_ref`).
    3) For each chirp, regrid H/V spectra to `vel_ref`.
    4) Concatenate all chirps along `range`.
    5) Optionally compute moments and ZDR/sZDR.
    """
    n_chirps = _chirp_num(ds)
    if n_chirps <= 0:
        raise ValueError("RPG dataset has no valid ChirpNum")

    maxvel_name = _first_existing(ds, ["MaxVel", "maxVel"])
    dopplen_name = _first_existing(ds, ["DoppLen", "doppLen"])
    if maxvel_name is None or dopplen_name is None:
        raise ValueError("RPG dataset missing MaxVel/DoppLen")

    maxVel_all = np.asarray(ds[maxvel_name].values).squeeze()
    doppLen_all = np.asarray(ds[dopplen_name].values).squeeze()

    # Use chirp-1 settings as default reference if no external velRef is provided.
    maxVel0 = float(_scalar_from_var(maxVel_all, idx=0, default=0.0))
    doppLen0 = int(_scalar_from_var(doppLen_all, idx=0, default=0))
    if doppLen0 <= 1:
        raise ValueError("Invalid DoppLen for chirp 1")

    vel_ref = as_vel_ref(velRef)
    if vel_ref is None:
        vel_ref = np.linspace(-maxVel0, maxVel0, doppLen0, dtype=np.float32)

    parts: list[xr.Dataset] = []

    for i in range(1, n_chirps + 1):
        ch_range = f"C{i}Range"
        ch_h = f"C{i}HSpec"
        ch_v = f"C{i}VSpec"
        ch_hn = f"C{i}HNoisePow"
        ch_vn = f"C{i}VNoisePow"

        # Skip chirps that do not contain full H/V spectrum + noise information.
        required = [ch_range, ch_h, ch_v, ch_hn, ch_vn]
        if any(v not in ds.variables and v not in ds.coords for v in required):
            continue

        maxVel_i = float(_scalar_from_var(maxVel_all, idx=i - 1, default=maxVel0))
        doppLen_i = int(_scalar_from_var(doppLen_all, idx=i - 1, default=doppLen0))
        if doppLen_i <= 1:
            continue

        # Chirp-native velocity grid and spacing.
        vel_i = np.linspace(-maxVel_i, maxVel_i, doppLen_i, dtype=np.float32)
        dv_i = float(np.abs(np.diff(vel_i)[0]))
        dv_ref = float(np.abs(np.diff(vel_ref)[0])) if len(vel_ref) > 1 else dv_i

        ch = ds[[ch_h, ch_v, ch_hn, ch_vn]].copy()
        vel_dim = f"C{i}Vel"
        ch = ch.assign_coords({vel_dim: vel_i})

        # Convert integrated noise power to per-bin density for consistent remapping.
        noise_dens_h = ch[ch_hn] / doppLen_i
        noise_dens_v = ch[ch_vn] / doppLen_i
        spec_threshold = 10 ** (-90 / 10)

        # Apply weak threshold and normalize by dv before regridding.
        h = ch[ch_h].where(ch[ch_h] > spec_threshold, np.nan) / dv_i
        v = ch[ch_v].where(ch[ch_v] > spec_threshold, np.nan) / dv_i

        # Regrid to common velocity axis; convert back to integrated bin units.
        h = h.reindex({vel_dim: vel_ref}, method="nearest", tolerance=0.05) * dv_ref
        v = v.reindex({vel_dim: vel_ref}, method="nearest", tolerance=0.05) * dv_ref

        # Align noise to (time, range) grid (no velocity dependency).
        base = h.isel({vel_dim: 0}, drop=True)
        noise_dens_h = noise_dens_h.broadcast_like(base)
        noise_dens_v = noise_dens_v.broadcast_like(base)

        out_ch = xr.Dataset(
            data_vars={
                "sZeH": h.rename({vel_dim: "Vel"}),
                "sZeV": v.rename({vel_dim: "Vel"}),
                "NoisePowH": noise_dens_h * doppLen_i,
                "NoisePowV": noise_dens_v * doppLen_i,
            },
            coords={ch_range: ds[ch_range]},
        ).rename({ch_range: "range"})

        parts.append(out_ch)

    if not parts:
        raise ValueError("No valid chirp spectra found in RPG file")

    # Stack chirps into one continuous range dimension.
    out = xr.concat(parts, dim="range")
    out = out.sortby("range")
    _, idx = np.unique(out["range"].values, return_index=True)
    out = out.isel(range=np.sort(idx))
    # optional pol calculation
    if include_pol:
        out["sZDR"] = 10.0 * np.log10(out["sZeH"] / out["sZeV"])
    # Optional moment products from spectra.
    if include_moments:
        mom_h = compute_spectral_moments(out["sZeH"], vel_dim="Vel", name_m0="ZeH", name_m1="MDV_H", name_m2="WIDTH_H")
        mom_v = compute_spectral_moments(out["sZeV"], vel_dim="Vel", name_m0="ZeV", name_m1="MDV_V", name_m2="WIDTH_V")
        out = xr.merge([out, mom_h, mom_v])
        if include_pol:
            out["ZDR"] = 10.0 * np.log10(out["ZeH"] / out["ZeV"])
        

    return out


def _infer_lv1_path_from_lv0(ds: xr.Dataset) -> Path | None:
    """
    Infer companion LV1 filename from LV0 source path stored in encoding.

    Expected pattern:
      *LV0.nc -> *LV1.nc
    Also supports lowercase lv0/lv1 replacements.
    """
    src = ds.encoding.get("source", None)
    if not src:
        return None
    p = Path(src)

    name1 = p.name.replace("LV0", "LV1")
    cand = p.with_name(name1)
    if cand.exists():
        return cand

    name2 = p.name.replace("lv0", "lv1")
    cand2 = p.with_name(name2)
    if cand2.exists():
        return cand2

    return None


def _merge_chirp_field(ds: xr.Dataset, field_base: str) -> xr.DataArray | None:
    """
    Merge a per-chirp LV1 field (e.g., SLDR or KDP) into one DataArray over `range`.

    Looks for:
      C{i}{field_base}, C{i}{FIELD_BASE}, C{i}{field_base.lower()}
    with corresponding C{i}Range.
    """
    n_chirps = _chirp_num(ds)
    parts: list[xr.DataArray] = []

    for i in range(1, n_chirps + 1):
        rname = f"C{i}Range"
        cands = [f"C{i}{field_base}", f"C{i}{field_base.upper()}", f"C{i}{field_base.lower()}"]
        vname = next((c for c in cands if c in ds.variables), None)
        if vname is None or rname not in ds.variables:
            continue

        da = ds[vname]
        if rname in da.dims:
            da = da.rename({rname: "range"})
        else:
            # If chirp range is available as variable only, attach as coordinate.
            da = da.assign_coords(range=ds[rname]).expand_dims("range")

        parts.append(da)

    if not parts:
        return None

    out = xr.concat(parts, dim="range")
    if "range" in out.coords:
        out = out.sortby("range")
        _, idx = np.unique(out["range"].values, return_index=True)
        out = out.isel(range=np.sort(idx))
    return out


def _load_lv1_polars(ds_lv0: xr.Dataset) -> xr.Dataset:
    """
    Load companion LV1 file and extract polarimetric fields merged over chirps.

    Currently adds:
      - SLDR (or LDR as fallback name)
      - KDP
    Returns empty dataset if companion file is not found.
    """
    lv1_path = _infer_lv1_path_from_lv0(ds_lv0)
    if lv1_path is None:
        return xr.Dataset()

    with xr.open_dataset(lv1_path) as d1:
        if "time" not in d1.coords and "Time" in d1.coords:
            d1 = d1.rename({"Time": "time"})
        #print('d1 attrs',d1.time.attrs)
        d1 = decode_time_if_needed(d1)
        #d1 = ensure_datetime_time(d1, time_var="time")
        #print('d1 attrs',d1.time.attrs)
        d1 = d1.load()

    out = xr.Dataset()
    sldr = _merge_chirp_field(d1, "SLDR")
    if sldr is None:
        sldr = _merge_chirp_field(d1, "LDR")
    if sldr is not None:
        out["SLDR"] = sldr

    kdp = _merge_chirp_field(d1, "KDP")
    #print(np.isnan(kdp).all())
    if kdp is not None:
        out["KDP"] = kdp

    return out


def _align_polars_to_out(pol: xr.Dataset, out: xr.Dataset) -> xr.Dataset:
    """
    Align LV1 polarimetric variables to the processed LV0 output grid.

    Reindexes by nearest neighbor on:
      - time (2 s tolerance)
      - range (5 m tolerance)
      - Vel (0.05 m/s tolerance, if present)
    """
    if not pol.data_vars:
        return pol

    aligned = xr.Dataset()
    for v in pol.data_vars:
        da = pol[v]
        if "time" in da.dims and "time" in out.coords:
            da = da.reindex(time=out["time"], method="nearest", tolerance=np.timedelta64(4, "s"))
        if "range" in da.dims and "range" in out.coords:
            da = da.reindex(range=out["range"], method="nearest", tolerance=16)
        #if "Vel" in da.dims and "Vel" in out.coords:
        #    da = da.reindex(Vel=out["Vel"], method="nearest", tolerance=0.05)
        aligned[v] = da
    return aligned


class RPGBackend(RadarBackend):
    """Backend implementation for RPG radar files (LV0 spectra + optional LV1 polars)."""

    name = "rpg"

    @staticmethod
    def can_handle(ds: xr.Dataset) -> bool:
        """
        Auto-detection rule for RPG input.

        Uses presence of `C1Range` as a signature variable.
        """
        return "C1Range" in ds.variables or "C1Range" in ds.coords

    def process(
        self,
        ds: xr.Dataset,
        *,
        velRef=None,
        include_moments: bool = True,
        include_ldr: bool = False,
        include_pol: bool = False,
    ) -> xr.Dataset:
        """
        Process RPG LV0 spectra into common output variables.

        Options:
          - include_moments: compute Ze/MDV/WIDTH (+ ZDR/sZDR)
          - include_ldr: add SLDR from LV1 (if available)
          - include_pol: add additional LV1 polars (currently KDP, and SLDR if missing)
        """
        # Normalize/decode time first.
        if "time" not in ds.coords and "Time" in ds.coords:
            ds = ds.rename({"Time": "time"})
        #print('ds attrs',ds.time.attrs)
        ds = decode_time_if_needed(ds)
        #print(ds.time.attrs)
        #ds = ensure_datetime_time(ds, time_var="time")
        #print(ds.time.attrs)
        # Core LV0 chirp merge + spectra processing.
        out = _merge_chirps_spec(ds,velRef=velRef,include_moments=include_moments,include_pol=include_pol)

        # Optional LV1 ingestion for polarimetric fields.
        if include_ldr or include_pol:
            pol = _load_lv1_polars(ds)
            #print(out)
            #print(pol)
            pol = _align_polars_to_out(pol, out)
            #print(np.isnan(pol.KDP).all())
            #quit()
            if include_ldr and "SLDR" in pol:
                out["SLDR"] = pol["SLDR"]

            if include_pol:
                if "KDP" in pol:
                    out["KDP"] = pol["KDP"]
                if "SLDR" in pol and "SLDR" not in out:
                    out["SLDR"] = pol["SLDR"]

        # Final metadata and coordinate hygiene.
        out = add_standard_variable_attrs(out)
        #out = ensure_datetime_time(out, time_var="time")
        if "time" in out.coords:
            out = out.sortby("time")

        return finalize_metadata(
            out,
            backend_name=self.name,
            include_moments=include_moments,
            include_ldr=include_ldr,
        )
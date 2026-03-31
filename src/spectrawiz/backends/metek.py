from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from .base import RadarBackend
from ..processing_common import (
    add_standard_variable_attrs,
    as_vel_ref,
    compute_ldr,
    compute_spectral_moments,
    decode_time_if_needed,
    ensure_datetime_time,
    finalize_metadata,
)


def _get_spec_metek(data: xr.Dataset, ch: Literal["o", "x"], velRef=None) -> tuple[xr.DataArray, xr.DataArray]:
    npw = data["npw1"] if ch == "o" else data["npw2"]

    cal_spec = (
        data["RadarConst"]
        * data[f"SNRCorFaC{ch}"]
        * (data["range"] ** 2 / (5000.0 ** 2))
        * data[f"SPCc{ch}"]
        / npw
    )
    cal_noise = (
        data["RadarConst"]
        * data[f"SNRCorFaC{ch}"]
        * (data["range"] ** 2 / (5000.0 ** 2))
        * data[f"HSDc{ch}"]
        / npw
    )

    spec = (cal_spec - cal_noise).where(np.isfinite(cal_spec), np.nan)
    spec = spec.sortby("doppler")
    spec = spec.reindex(doppler=spec.doppler[::-1])
    spec = spec.assign_coords(doppler=spec["doppler"] * -1.0)
    spec = spec.where(spec != 0.0)

    vr = as_vel_ref(velRef)
    if vr is not None:
        old_v = np.asarray(spec["doppler"].values, dtype=float)
        old_dv = float(np.abs(np.nanmedian(np.diff(old_v)))) if old_v.size > 1 else 1.0

        spec = spec.reindex({"doppler": vr}, method="nearest", tolerance=0.05)

        new_v = np.asarray(spec["doppler"].values, dtype=float)
        new_dv = float(np.abs(np.nanmedian(np.diff(new_v)))) if new_v.size > 1 else old_dv

        if old_dv > 0 and new_dv > 0:
            spec = spec / old_dv * new_dv

    spec = spec.rename({"doppler": "Vel"})
    return spec, cal_noise


class MetekBackend(RadarBackend):
    name = "metek"

    @staticmethod
    def can_handle(ds: xr.Dataset) -> bool:
        return "SPCco" in ds.variables

    def process(
        self,
        ds: xr.Dataset,
        *,
        velRef=None,
        include_moments: bool = True,
        include_ldr: bool = True,
    ) -> xr.Dataset:
        ds = decode_time_if_needed(ds)
        ds = ensure_datetime_time(ds, time_var="time")

        have_cross = "SPCcx" in ds.variables
        base_vars = ["SPCco", "SNRCorFaCo", "doppler", "HSDco", "npw1", "npw2", "RadarConst", "range", "time"]
        cross_vars = ["SPCcx", "SNRCorFaCx", "HSDcx"]
        keep = [v for v in base_vars if v in ds.variables or v in ds.coords]
        if have_cross:
            keep += [v for v in cross_vars if v in ds.variables]
        ds = ds[keep]

        spec_co, noise_co = _get_spec_metek(ds, "o", velRef=velRef)
        out = xr.merge([spec_co.rename("sZeCo"), noise_co.rename("NoisePowCo")])

        if have_cross:
            spec_cross, noise_cross = _get_spec_metek(ds, "x", velRef=velRef)
            out = xr.merge([out, spec_cross.rename("sZeCx"), noise_cross.rename("NoisePowCx")])

        if include_moments:
            out = xr.merge(
                [
                    out,
                    compute_spectral_moments(
                        out["sZeCo"],
                        vel_dim="Vel",
                        name_m0="ZeCo",
                        name_m1="MDV_Co",
                        name_m2="WIDTH_Co",
                    ),
                ]
            )
            if "sZeCx" in out:
                out = xr.merge(
                    [
                        out,
                        compute_spectral_moments(
                            out["sZeCx"],
                            vel_dim="Vel",
                            name_m0="ZeCx",
                            name_m1="MDV_Cx",
                            name_m2="WIDTH_Cx",
                        ),
                    ]
                )

        if include_ldr and ("sZeCo" in out) and ("sZeCx" in out):
            out = xr.merge([out, compute_ldr(out["sZeCo"], out["sZeCx"], vel_dim="Vel")])

        out = add_standard_variable_attrs(out)
        out = ensure_datetime_time(out, time_var="time")
        if "time" in out.coords:
            out = out.sortby("time")

        return finalize_metadata(
            out,
            backend_name=self.name,
            include_moments=include_moments,
            include_ldr=include_ldr,
        )
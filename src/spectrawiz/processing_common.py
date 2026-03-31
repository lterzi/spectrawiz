from __future__ import annotations

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
        # Accept both 'units' and 'Units'
        units = ds["time"].attrs.get("units") or ds["time"].attrs.get("Units")
        if units is None:
            # fallback default
            ds["time"].attrs["units"] = "seconds since 1970-01-01 00:00:00 UTC"
        else:
            # Try to parse non-standard units string
            if "since" in units:
                # Try to extract epoch
                import re
                match = re.search(r"since\s+([0-9/:\-\s]+)", units)
                if match:
                    epoch = match.group(1).strip()
                    # Try to parse various date formats
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
                        # fallback to 2001-01-01 if not parseable
                        ds["time"].attrs["units"] = "seconds since 2001-01-01 00:00:00"
                else:
                    ds["time"].attrs["units"] = "seconds since 2001-01-01 00:00:00"
            else:
                ds["time"].attrs["units"] = "seconds since 2001-01-01 00:00:00"
        # Remove non-standard 'Units' attribute if present
        if "Units" in ds["time"].attrs:
            del ds["time"].attrs["Units"]
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
    spec_h: xr.DataArray,
    spec_v: xr.DataArray,
    vel_dim: str = "Vel",
) -> xr.Dataset:
    """
    Returns:
      - LDR  = 10*log10(ZeH/ZeV)
      - sLDR = 10*log10(sZeH/sZeV)
    """
    # Integrated powers
    ze_h = spec_h.where(np.isfinite(spec_h), 0.0).clip(min=0.0).integrate(vel_dim)
    ze_v = spec_v.where(np.isfinite(spec_v), 0.0).clip(min=0.0).integrate(vel_dim)

    ldr = (10.0 * np.log10(ze_h / ze_v.where(ze_v > 0))).rename("LDR")

    # Spectral ratio
    sldr = (10.0 * np.log10(spec_h / spec_v.where(spec_v > 0))).rename("sLDR")

    return xr.merge([ldr, sldr])


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
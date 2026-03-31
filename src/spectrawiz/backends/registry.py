from __future__ import annotations

import xarray as xr
from .base import RadarBackend


_BACKENDS: dict[str, RadarBackend] = {}


def register_backend(backend: RadarBackend, overwrite: bool = False) -> None:
    key = backend.name.lower()
    if key in _BACKENDS and not overwrite:
        raise ValueError(f"Backend '{key}' is already registered.")
    _BACKENDS[key] = backend


def available_backends() -> list[str]:
    return sorted(_BACKENDS.keys())


def select_backend(ds: xr.Dataset, radar_type: str = "auto") -> RadarBackend:
    if radar_type != "auto":
        key = radar_type.lower()
        if key not in _BACKENDS:
            raise ValueError(f"Unknown radar_type '{radar_type}'. Available: {available_backends()}")
        return _BACKENDS[key]

    for backend in _BACKENDS.values():
        if backend.can_handle(ds):
            return backend
    raise ValueError("Could not detect radar type from dataset variables.")
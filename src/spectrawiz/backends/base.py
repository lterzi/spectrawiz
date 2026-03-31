from __future__ import annotations

from abc import ABC, abstractmethod
import xarray as xr


class RadarBackend(ABC):
    name: str = "base"

    @staticmethod
    @abstractmethod
    def can_handle(ds: xr.Dataset) -> bool:
        raise NotImplementedError

    @abstractmethod
    def process(
        self,
        ds: xr.Dataset,
        *,
        velRef=None,
        include_moments: bool = True,
        include_ldr: bool = True,
    ) -> xr.Dataset:
        raise NotImplementedError
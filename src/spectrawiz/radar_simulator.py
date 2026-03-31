from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.constants import c
from scipy import signal as sig
import xarray as xr

def _pick_var(ds, candidates):
    for name in candidates:
        if name in ds:
            return ds[name]
    raise KeyError(f"None of {candidates} found in LUT")

def _dB(x):
    return 10.0 * np.log10(np.clip(np.asarray(x, dtype=float), 1e-30, None))


def _marshall_palmer(D_mm, R):
    """Marshall-Palmer DSD: N(D) = N0 * exp(-lambda*D), D in mm."""
    N0 = 8000.0  # m^-3 mm^-1
    lam = 4.1 * (float(R) ** -0.21)
    return N0 * np.exp(-lam * np.asarray(D_mm, dtype=float))

def _gamma_dsd(D_mm, N0, lam, gamma):
    """Generalized gamma DSD: N(D) = N0 * D^gamma * exp(-lam*D), D in mm."""
    return N0 * (D_mm ** gamma) * np.exp(-lam * D_mm)

def _convolve_broad_fft(spec, vel, spec_broad):
    spec = np.asarray(spec, dtype=float)
    vel = np.asarray(vel, dtype=float)
    spec_broad = max(float(spec_broad), 1e-6)

    dv = np.diff(vel)[0]
    kernel = np.exp(-0.5 * (vel / spec_broad) ** 2) * dv
    out = sig.fftconvolve(spec, kernel, mode="same")
    return out / (np.sqrt(2.0 * np.pi) * spec_broad)


def _convolve_noise(spec, vel, noise_pow_db, nave):
    spec = np.asarray(spec, dtype=float)
    vel = np.asarray(vel, dtype=float)

    nfft = len(vel)
    dv = np.diff(vel)[0]
    noise_lin = (10.0 ** (float(noise_pow_db) / 10.0)) * (nfft * dv)
    Ni = noise_lin / (nfft * dv)

    rng = np.random.default_rng()
    S = np.zeros(nfft, dtype=float)
    n_avg = max(int(nave), 1)
    for _ in range(n_avg):
        r = rng.uniform(size=nfft)
        S += -np.log(r) * (spec + Ni)
    return S / n_avg


def simulate_rain_spectrum(
    vel_bins,
    center_height,
    eps_diss,
    noise_pow,
    nave,
    theta_deg,
    uwind,
    time_int,
    lut_path,
    gamma,
    lam,  # default lam for Marshall-Palmer
    N0,  # default N0 for Marshall-Palmer
    wl_mm=None,
    freq_ghz=35.6,
    K2=0.93,
    vertical_wind: float = 0.0,
):
    """
    Returns
    -------
    vel_centers, spec_H_dB, spec_V_dB
    """
    vel_bins = np.asarray(vel_bins, dtype=float)
    vel_centers = 0.5 * (vel_bins[:-1] + vel_bins[1:])
    
    if wl_mm is None:
        wl_mm = c / (float(freq_ghz) * 1e9) * 1e3

    theta = float(theta_deg) / 180.0 * np.pi

    # Drop size and fall speed model
    D_mm = np.linspace(0.01, 4.0, 1000)
    av, bv = 6.18, -0.6
    v_D = -1 * av * np.exp(bv * D_mm) + float(vertical_wind)
    #PSD = _marshall_palmer(D_mm, R)
    PSD = _gamma_dsd(D_mm, N0, lam, gamma)
    # LUT
    lut = pd.read_csv(lut_path)
    lut = (
        lut.set_index("diameter[mm]")
        .to_xarray()
        .rename(
            {
                "diameter[mm]": "Dmax",
                "radarXSh[mm2]": "c_bck_h",
                "radarXSv[mm2]": "c_bck_v",
                "extxs[mm2]": "cext_h",
                "sKdp[mm2]": "sKDP",
            }
        )
    )

    data = pd.DataFrame({"dia": D_mm, "vel": v_D}).to_xarray()
    points = lut.sel(Dmax=data["dia"], method="nearest")

    prefactor = wl_mm**4 / (np.pi**5 * float(K2))
    dD_dv = np.abs(np.gradient(D_mm) / np.gradient(v_D))

    data = data.sortby("vel")
    data["sZePH"] = points.c_bck_h * prefactor * PSD * dD_dv
    data["sZePV"] = points.c_bck_v * prefactor * PSD * dD_dv

    group = data.groupby_bins("vel", vel_bins, labels=vel_centers).mean()
    spec_H = group["sZePH"].rename({"vel_bins": "vel"}).fillna(0.0).values
    spec_V = group["sZePV"].rename({"vel_bins": "vel"}).fillna(0.0).values

    # Broadening
    L_s = float(uwind) * float(time_int) + 2.0 * float(center_height) * np.sin(theta)
    L_lam = wl_mm * 1e-3 / 2.0
    sigma_t2 = 0.75 * (float(eps_diss) / (2.0 * np.pi)) ** (2.0 / 3.0) * (
        max(L_s, 0.0) ** (2.0 / 3.0) - max(L_lam, 0.0) ** (2.0 / 3.0)
    )
    sigma_b2 = float(uwind) ** 2 * theta**2 / 2.76
    spec_broad = max(np.sqrt(abs(sigma_t2) + sigma_b2), 1e-4)

    spec_H = _convolve_broad_fft(spec_H, vel_centers, spec_broad)
    spec_V = _convolve_broad_fft(spec_V, vel_centers, spec_broad)

    # Noise
    spec_H = _convolve_noise(spec_H, vel_centers, noise_pow, nave)
    spec_V = _convolve_noise(spec_V, vel_centers, noise_pow, nave)

    return vel_centers, _dB(spec_H), _dB(spec_V)


def simulate_snow_spectrum(
    vel_bins,
    N0,
    lam,
    gamma,
    center_height,
    eps_diss,
    noise_pow,
    nave,
    theta_deg,
    uwind,
    time_int,
    lut_path,
    wl_mm=None,
    freq_ghz=35.6,
    K2=0.93,
    vertical_wind: float = 0.0,
):
    vel_bins = np.asarray(vel_bins, dtype=float)
    vel_centers = 0.5 * (vel_bins[:-1] + vel_bins[1:])

    if wl_mm is None:
        wl_mm = c / (float(freq_ghz) * 1e9) * 1e3

    theta = float(theta_deg) / 180.0 * np.pi

    try:
        lut = xr.open_dataset(lut_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open LUT {lut_path}: {e}")

    # Always select first value for extra dimensions
    if "elevation" in lut.coords:
        lut = lut.sel(elevation=90, method="nearest")
    if "frequency" in lut.coords:
        freq_vals = lut.coords["frequency"].values
        if freq_vals.max() > 1e6:
            lut = lut.sel(frequency=float(freq_ghz) * 1e9, method="nearest")
        else:
            lut = lut.sel(frequency=float(freq_ghz), method="nearest")
    for tname in ["temperature", "temp", "T"]:
        if tname in lut.coords:
            lut = lut.isel({tname: 0})  # just take the only/first temperature
            break

    try:
        D = _pick_var(lut, ["size", "Dmax", "diameter"]).values
        v_lut = _pick_var(lut, ["vel", "velocity"]).values
        cbck = _pick_var(lut, ["Cbck", "c_bck_h", "radarXSh[mm2]"]).values
    except KeyError as e:
        raise RuntimeError(f"Missing required LUT variable: {e}")

    D = np.asarray(D, dtype=float).flatten()
    v_lut = -1 * np.asarray(v_lut, dtype=float).flatten() + float(vertical_wind)
    cbck = np.asarray(cbck, dtype=float).flatten()

    #PSD = float(N0) * np.exp(-float(lam) * D)
    PSD = _gamma_dsd(D, N0, lam, gamma)
    
    prefactor = (wl_mm * 1e-3) ** 4 / (np.pi**5 * float(K2))
    dD_dv = np.abs(np.gradient(D) / np.gradient(v_lut))
    sZeH_native = 1e18 * cbck * prefactor * PSD * dD_dv

    da = xr.DataArray(sZeH_native, coords={"vel": v_lut}, dims=("vel",))
    g = da.groupby_bins("vel", vel_bins, labels=vel_centers).mean()
    spec_H = g.rename({"vel_bins": "vel"}).reindex(vel=vel_centers, fill_value=0.0).values
    spec_H = g.rename({"vel_bins": "vel"}).fillna(0).values
    
    # Broadening + noise (same model as rain)
    L_s = float(uwind) * float(time_int) + 2.0 * float(center_height) * np.sin(theta)
    L_lam = wl_mm * 1e-3 / 2.0
    sigma_t2 = 0.75 * (float(eps_diss) / (2.0 * np.pi)) ** (2.0 / 3.0) * (
        max(L_s, 0.0) ** (2.0 / 3.0) - max(L_lam, 0.0) ** (2.0 / 3.0)
    )
    sigma_b2 = float(uwind) ** 2 * theta**2 / 2.76
    spec_broad = max(np.sqrt(abs(sigma_t2) + sigma_b2), 1e-4)

    spec_H = _convolve_broad_fft(spec_H, vel_centers, spec_broad)
    #print(f"[simulate_snow] spec_H after broadening range: {spec_H.min():.4e} to {spec_H.max():.4e}")
    spec_H = _convolve_noise(spec_H, vel_centers, noise_pow, nave)

    return vel_centers, _dB(spec_H)

def _centers_to_edges(vel_centers):
    """
    Convert velocity bin centers -> bin edges.
    """
    vc = np.asarray(vel_centers, dtype=float)
    if vc.ndim != 1 or vc.size < 2:
        raise ValueError("measured_vel must be a 1D array with at least 2 points")

    # robust to increasing or decreasing ordering
    mids = 0.5 * (vc[:-1] + vc[1:])
    first = vc[0] - 0.5 * (vc[1] - vc[0])
    last = vc[-1] + 0.5 * (vc[-1] - vc[-2])
    return np.concatenate([[first], mids, [last]])


class RainSimulator:
    """
    Simulation backend for rain panel (row 2, col 0).
    Owns simulation lines and update logic.
    """

    def __init__(
        self,
        ax,
        fig,
        vel_bins=None,
        lut_path: str = "",
        freq_ghz: float = 35.6,
        K2: float = 0.93,
        auto_vel_bins: bool = True,
    ):
        self.ax = ax
        self.fig = fig
        self.vel_bins = None if vel_bins is None else np.asarray(vel_bins, dtype=float)
        self.auto_vel_bins = bool(auto_vel_bins)
        self.lut_path = lut_path
        self.freq_ghz = float(freq_ghz)
        self.K2 = float(K2)

        self.l_sim_H, = self.ax.plot([], [], lw=2, ls="--", color="C1", label="simulated rain H")
        self.ax.legend(loc="best")

    def clear(self):
        self.l_sim_H.set_data([], [])

    def update(
        self,
        measured_vel,
        measured_spec_dB,
        center_height,
        R,
        eps_diss,
        noise_pow,
        nave,
        theta_deg,
        uwind,
        time_int,
        vertical_wind,
    ):
        try:
            if self.auto_vel_bins or self.vel_bins is None:
                sim_vel_bins = _centers_to_edges(measured_vel)
            else:
                sim_vel_bins = self.vel_bins

            vel_sim, sim_H, _sim_V = simulate_rain_spectrum(
                vel_bins=sim_vel_bins,
                R=R,
                center_height=center_height,
                eps_diss=eps_diss,
                noise_pow=noise_pow,
                nave=nave,
                theta_deg=theta_deg,
                uwind=uwind,
                time_int=time_int,
                lut_path=self.lut_path,
                freq_ghz=self.freq_ghz,
                K2=self.K2,
                vertical_wind=vertical_wind,
            )
        except Exception:
            self.clear()
            return

        self.l_sim_H.set_data(vel_sim, sim_H)

        # Expand limits to include measured + simulated(H)
        xv = np.concatenate([np.asarray(measured_vel, dtype=float), np.asarray(vel_sim, dtype=float)])
        yv = np.concatenate([np.asarray(measured_spec_dB, dtype=float), np.asarray(sim_H, dtype=float)])

        xv = xv[np.isfinite(xv)]
        yv = yv[np.isfinite(yv)]

        if xv.size:
            xmin, xmax = float(np.nanmin(xv)), float(np.nanmax(xv))
            if xmin == xmax:
                xmax = xmin + 1e-6
            xpad = 0.03 * (xmax - xmin)
            self.ax.set_xlim(xmin - xpad, xmax + xpad)

        if yv.size:
            ymin, ymax = float(np.nanmin(yv)), float(np.nanmax(yv))
            if ymin == ymax:
                ymax = ymin + 1e-6
            ypad = 0.05 * (ymax - ymin)
            self.ax.set_ylim(ymin - ypad, ymax + ypad)
        return vel_sim, sim_H


class SnowSimulator:
    """Simulation backend for snow panel (row 3, col 0)."""

    def __init__(
        self,
        ax,
        fig,
        vel_bins=None,
        lut_path: str = "",
        freq_ghz: float = 35.6,
        K2: float = 0.93,
        auto_vel_bins: bool = True,
    ):
        self.ax = ax
        self.fig = fig
        self.vel_bins = None if vel_bins is None else np.asarray(vel_bins, dtype=float)
        self.auto_vel_bins = bool(auto_vel_bins)
        self.lut_path = lut_path
        self.freq_ghz = float(freq_ghz)
        self.K2 = float(K2)

        self.l_sim_H, = self.ax.plot([], [], lw=2, ls="--", color="C3", label="simulated snow H")
        self.ax.legend(loc="best")

    def clear(self):
        self.l_sim_H.set_data([], [])

    def update(
        self,
        measured_vel,
        measured_spec_dB,
        center_height,
        N0,
        lam,
        eps_diss,
        noise_pow,
        nave,
        theta_deg,
        uwind,
        time_int,
        vertical_wind,
    ):
        try:
            sim_vel_bins = _centers_to_edges(measured_vel) if (self.auto_vel_bins or self.vel_bins is None) else self.vel_bins
            vel_sim, sim_H = simulate_snow_spectrum(
                vel_bins=sim_vel_bins,
                N0=N0,
                lam=lam,
                center_height=center_height,
                eps_diss=eps_diss,
                noise_pow=noise_pow,
                nave=nave,
                theta_deg=theta_deg,
                uwind=uwind,
                time_int=time_int,
                lut_path=self.lut_path,
                freq_ghz=self.freq_ghz,
                K2=self.K2,
                vertical_wind=vertical_wind,
            )
            
        except Exception as e:
            import traceback
            print(f"[SnowSimulator] ERROR: {e}")
            traceback.print_exc()
            self.clear()
            return

        self.l_sim_H.set_data(vel_sim, sim_H)

        # Expand limits to include measured + simulated(H)
        xv = np.concatenate([np.asarray(measured_vel, dtype=float), np.asarray(vel_sim, dtype=float)])
        yv = np.concatenate([np.asarray(measured_spec_dB, dtype=float), np.asarray(sim_H, dtype=float)])

        xv = xv[np.isfinite(xv)]
        yv = yv[np.isfinite(yv)]

        if xv.size:
            xmin, xmax = float(np.nanmin(xv)), float(np.nanmax(xv))
            if xmin == xmax:
                xmax = xmin + 1e-6
            xpad = 0.03 * (xmax - xmin)
            self.ax.set_xlim(xmin - xpad, xmax + xpad)

        if yv.size:
            ymin, ymax = float(np.nanmin(yv)), float(np.nanmax(yv))
            if ymin == ymax:
                ymax = ymin + 1e-6
            ypad = 0.05 * (ymax - ymin)
            self.ax.set_ylim(ymin - ypad, ymax + ypad)

        return vel_sim, sim_H
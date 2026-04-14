import numpy as np
from scipy.constants import c
from scipy import signal as sig
import xarray as xr

# TODO: have only one routine for radar simulation, this is the same for rain and snow, only LUT should change. 

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
def precipitation_rate_from_psd(D_mm, PSD, v_fall, mass):
    """
    Compute liquid-equivalent precipitation rate from a particle size distribution.

    Parameters
    ----------
    D_mm : array-like
        Particle diameter / maximum dimension in mm.
    PSD : array-like
        N(D) in m^-3 mm^-1.
    v_fall : array-like
        Terminal fall velocities in m/s (positive downward).
    mass : array-like
        Particle masses in kg. For rain: (pi/6) * rho_w * D^3.

    Returns
    -------
    R : float
        Precipitation rate in mm/h liquid water equivalent.
    """
    D_mm = np.asarray(D_mm, dtype=float)
    PSD = np.asarray(PSD, dtype=float)
    v_fall = np.abs(np.asarray(v_fall, dtype=float))
    mass_kg = np.asarray(mass, dtype=float)

    integrand = mass_kg * v_fall * PSD
    flux_kg = np.trapezoid(integrand, D_mm)

    rho_w = 1000.0  # kg/m^3
    R_mm_per_h = (flux_kg / rho_w) * 1e3 * 3600.0
    return R_mm_per_h


def simulate_spectrum(
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
    #time_int,
    lut_path,
    #wl=None,
    prf,
    freq_ghz,
    att,
    K2=0.93,
    vertical_wind: float = 0.0,
):
    vel_bins = np.asarray(vel_bins, dtype=float)
    nfft = len(vel_bins) - 1
    vel_centers = 0.5 * (vel_bins[:-1] + vel_bins[1:])

    #if wl is None:
    wl = c / (float(freq_ghz) * 1e9)

    theta = float(theta_deg) / 180.0 * np.pi

    #try:
    lut = xr.open_dataset(lut_path)
    #except Exception as e:
    #    raise RuntimeError(f"Failed to open LUT {lut_path}: {e}")
    #print(lut)
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
    except KeyError as e:
        raise RuntimeError(f"Missing required LUT variable: {e}")
    try:
        v_lut = _pick_var(lut, ["vel", "velocity"]).values
    except KeyError as e:
        av, bv = 6.18, -0.6
        v_lut = av * np.exp(bv * D*1e3) #+ float(vertical_wind)
        #raise RuntimeError(f"Missing required LUT variable: {e}")
    try:
        cbck = _pick_var(lut, ["Cbck", "c_bck_h", "radarXSh[mm2]"]).values
    except KeyError as e:
        raise RuntimeError(f"Missing required LUT variable: {e}")
    try:
        mass = _pick_var(lut, ["mass", "m"]).values
    except KeyError as e:
        if "aspect_ratio" in lut.coords:
            aspect_ratio = lut.coords["aspect_ratio"].values
            mass = (np.pi / 6.0) * 1000.0 * (D ** 3) * aspect_ratio
        else:
            mass = (np.pi / 6.0) * 1000.0 * (D ** 3)  # assume density of water
        #raise RuntimeError(f"Missing required LUT variable: {e}")
       
        

    D = np.asarray(D, dtype=float).flatten()
    #print('D snow max', D.max())
    v_lut = -1 * np.asarray(v_lut, dtype=float).flatten() + float(vertical_wind)
    cbck = np.asarray(cbck, dtype=float).flatten()
    #print(D.max(),v_lut.max(),cbck.max())

    #PSD = float(N0) * np.exp(-float(lam) * D)
    PSD = _gamma_dsd(D, N0, lam, gamma)
    #print('PSD snow',PSD)
    #print('PSD max', PSD.max())
    
    prefactor = wl ** 4 / (np.pi**5 * float(K2))
    dD_dv = np.abs(np.gradient(D) / np.gradient(v_lut))
    sZeH_native = 1e18 * cbck * prefactor * PSD * dD_dv

    da = xr.DataArray(sZeH_native, coords={"vel": v_lut}, dims=("vel",))
    g = da.groupby_bins("vel", vel_bins, labels=vel_centers).mean()
    spec_H = g.rename({"vel_bins": "vel"}).reindex(vel=vel_centers, fill_value=0.0).values
    spec_H = g.rename({"vel_bins": "vel"}).fillna(0).values

    # add attenuation
    att_lin = 10.0 ** (float(att) / 10.0)
    spec_H = spec_H / att_lin
    #print('spec_H max', spec_H.max())
    
    # Broadening + noise (same model as rain)
    time_int = nave*nfft/prf
    L_s = float(uwind) * float(time_int) + 2.0 * float(center_height) * np.sin(theta)
    L_lam = wl / 2.0
    sigma_t2 = 0.75 * (float(eps_diss) / (2.0 * np.pi)) ** (2.0 / 3.0) * (
        max(L_s, 0.0) ** (2.0 / 3.0) - max(L_lam, 0.0) ** (2.0 / 3.0)
    )
    sigma_b2 = float(uwind) ** 2 * theta**2 / 2.76
    spec_broad = max(np.sqrt(abs(sigma_t2) + sigma_b2), 1e-4)

    spec_H = _convolve_broad_fft(spec_H, vel_centers, spec_broad)
    #print(f"[simulate_snow] spec_H after broadening range: {spec_H.min():.4e} to {spec_H.max():.4e}")
    spec_H = _convolve_noise(spec_H, vel_centers, noise_pow, nave)

    precip_rate = precipitation_rate_from_psd(D, PSD, v_lut, mass)
    #print(D)
    
    return vel_centers, _dB(spec_H), PSD, D, precip_rate
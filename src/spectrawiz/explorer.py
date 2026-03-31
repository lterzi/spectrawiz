import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import matplotlib.dates as mdates
from spectrawiz import radar_simulator

#print('new version of explorer.py loaded')
#print('Matplotlib/slider version of explorer.py loaded')
def main():
    st.set_page_config(layout="wide")
    st.title("SpectraWiz: Interactive Radar Spectra Visualization")

    # --- User Inputs ---
    datapath = st.sidebar.text_input("Data directory", "/project/meteo/work/L.Terzi/MIM_radars/spectra_visualisation/processed_data/2025/09/10/")
    date = st.sidebar.text_input("Date (YYYY-MM-DD)", "2025-09-10")
    pattern = st.sidebar.text_input("File pattern", "*rpg_hourly_proc.nc")

    #lut_path = st.sidebar.text_input("LUT path", value="/path/to/your/lut.csv")  # <-- set your default here

    def find_files(datapath, date, pattern):
        y, m, d = date.split("-")
        search_path = os.path.join(datapath, pattern)
        files = sorted(glob.glob(search_path))
        return files

    files = find_files(datapath, date, pattern)
    if not files:
        st.error("No files found for the selected date and pattern.")
        st.stop()

    st.sidebar.write(f"Found {len(files)} files.")

    @st.cache_data
    def load_moments(files):
        datasets = []
        for f in files:
            ds = xr.open_dataset(f)
            non_spec_vars = [v for v in ds.data_vars if ds[v].ndim == 2 and set(ds[v].dims) == {"time", "range"}]
            datasets.append(ds[non_spec_vars])
        ds_merged = xr.concat(datasets, dim="time")
        return ds_merged

    ds_mom = load_moments(files)

    time_height_vars = list(ds_mom.data_vars)
    if not time_height_vars:
        st.error("No (time, range) variables found in files.")
        st.stop()

    #var = st.sidebar.selectbox("Time-Height Variable", time_height_vars)
    default_var = "ZeH"
    if default_var in time_height_vars:
        default_index = time_height_vars.index(default_var)
    else:
        default_index = 0

    var = st.sidebar.selectbox("Time-Height Variable", time_height_vars, index=default_index)
    range_var = "range"
    time_var = "time"

    time_values = ds_mom[time_var].values
    range_values = ds_mom[range_var].values

    # --- Spectral variable selection ---
    with xr.open_dataset(files[0]) as ds0:
        spec_vars = [
            v for v in ds0.data_vars
            if set(ds0[v].dims) >= {"time", "range"} and len(ds0[v].dims) == 3
        ]
    spec_var = st.sidebar.selectbox("Spectral Variable", spec_vars if spec_vars else ["spec"])

    # --- Default display/units/colorbar config ---
    units = {
        "range": "m",
        "velocity": "m/s",
        "time": "Time"
    }
    colorbar_labels = {
        "panel1": var,
        "panel3": spec_var,
    }
    clim = {
        "panel1": None,
        "panel3": None
    }

    # --- Sliders for selection (always visible and at the top) ---
    time_idx = st.sidebar.select_slider(
        "Time",
        options=list(range(len(time_values))),
        value=len(time_values) // 2,
        format_func=lambda i: pd.to_datetime(time_values[i]).strftime('%Y-%m-%d %H:%M')
    )
    range_idx = st.sidebar.select_slider(
        "Range",
        options=list(range(len(range_values))),
        value=len(range_values) // 2,
        format_func=lambda i: f"{range_values[i]:.1f} {units['range']}"
    )
    profile_offset = st.sidebar.slider("Close Time selection", -10, 10, 0)

    prof_time_idx = np.clip(time_idx + profile_offset, 0, len(time_values)-1)
    profile_time_str = pd.to_datetime(time_values[prof_time_idx]).strftime('%H:%M')

    # --- Display/Units/Colorbar config dictionary (can override defaults) ---
    with st.sidebar.expander("Display Options"):
        units["range"] = st.text_input("Range units (y-axis)", units["range"])
        units["velocity"] = st.text_input("Velocity units (x-axis, panels 3/4)", units["velocity"])
        units["time"] = st.text_input("Time units (panel 1 x-axis)", units["time"])
        colorbar_labels["panel1"] = st.text_input("Colorbar label (panel 1)", colorbar_labels["panel1"])
        colorbar_labels["panel3"] = st.text_input("Colorbar label (panel 3)", colorbar_labels["panel3"])
        colorbar_limits = {
            "panel1": st.text_input("Colorbar limits (panel 1, e.g. 0,30)", ""),
            "panel3": st.text_input("Colorbar limits (panel 3, e.g. 0,1)", ""),
        }
        for key, val in colorbar_limits.items():
            if val:
                try:
                    vmin, vmax = [float(x) for x in val.split(",")]
                    clim[key] = (vmin, vmax)
                except Exception:
                    clim[key] = None
                    st.warning(f"Invalid color limits for {key}. Use format: min,max")
            else:
                clim[key] = None

    def get_units_from_attrs(var):
        """Get units from variable attrs, case-insensitive for 'unit' or 'units'."""
        for key in var.attrs:
            if key.lower() in ["unit", "units"]:
                return var.attrs[key]
        return ""

    def convert_to_db_if_linear(varname, ds, values):
        """
        If units suggest linear reflectivity, convert to dB (10*log10).
        Only convert if not already in dB.
        """
        try:
            var = ds[varname]
            var_units = get_units_from_attrs(var)
        except Exception:
            var_units = ""
        var_units_lc = str(var_units).lower().replace(" ", "")
        # If already in dB, do nothing
        if any(x in var_units_lc for x in ["db", "dbz", "dbm"]):
            return values
        # Typical linear reflectivity units
        linear_patterns = ["mm6", "mm^6", "mm6/m3", "mm^6/m^3", "mm6m-3", "mm^6m^-3"]
        if any(pat in var_units_lc for pat in linear_patterns):
            values = np.where(values > 0, values, np.nan)
            ds[varname].attrs["units"] = "dB"
            return 10 * np.log10(values)
        return values

    # --- Helper: Find which file contains a given time index ---
    def find_file_for_time(files, ds_mom, time_idx):
        time_cumsum = np.cumsum([xr.open_dataset(f)["time"].size for f in files])
        file_idx = np.searchsorted(time_cumsum, time_idx, side="right")
        if file_idx == 0:
            local_time_idx = time_idx
        else:
            local_time_idx = time_idx - time_cumsum[file_idx-1]
        return files[file_idx], local_time_idx

    def load_spectrum(files, ds_mom, time_idx, range_idx, spec_var):
        file, local_time_idx = find_file_for_time(files, ds_mom, time_idx)
        with xr.open_dataset(file) as ds:
            if spec_var not in ds.data_vars:
                return None, None
            vel_dim = [d for d in ds[spec_var].dims if "vel" in d.lower() or d == "Vel"]
            if not vel_dim:
                return None, None
            vel_dim = vel_dim[0]
            spec = ds[spec_var].isel(time=local_time_idx, range=range_idx).load()
            vel = ds[vel_dim].values
        return vel, spec

    def load_spectrogram(files, ds_mom, time_idx, spec_var):
        file, local_time_idx = find_file_for_time(files, ds_mom, time_idx)
        with xr.open_dataset(file) as ds:
            if spec_var not in ds.data_vars:
                return None, None, None
            vel_dim = [d for d in ds[spec_var].dims if "vel" in d.lower() or d == "Vel"]
            if not vel_dim:
                return None, None, None
            vel_dim = vel_dim[0]
            spec = ds[spec_var].isel(time=local_time_idx).load()
            vel = ds[vel_dim].values
            rng = ds["range"].values
        return vel, rng, spec

    fontsize=14

    # --- Panel 1: Time-Height Plot ---
    z_disp = ds_mom[var].T.values
    z_disp = convert_to_db_if_linear(var, ds_mom, z_disp)
    time_values_disp = pd.to_datetime(ds_mom[time_var].values)
    y_disp = ds_mom[range_var].values

    fig1, ax1 = plt.subplots(figsize=(12, 3))
    im = ax1.pcolormesh(
        time_values_disp, y_disp, z_disp, shading="auto", cmap="turbo",
        vmin=clim["panel1"][0] if clim["panel1"] else None,
        vmax=clim["panel1"][1] if clim["panel1"] else None
    )
    ax1.axvline(time_values_disp[time_idx], color="red", linestyle="--")
    ax1.axhline(y_disp[range_idx], color="red", linestyle="--")
    ax1.set_ylabel(f"Range ({units['range']})", fontsize=fontsize)
    ax1.set_xlabel(units["time"], fontsize=fontsize)
    ax1.tick_params(labelsize=fontsize-2)
    ax1.set_title(f'Time x Range {var}', fontsize=fontsize+2)
    cbar = plt.colorbar(im, ax=ax1, pad=0.01)
    cbar.set_label(colorbar_labels["panel1"], fontsize=fontsize)

    # Set x-axis tick format to H:M and minor ticks every hour
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_minor_locator(mdates.HourLocator())
    ax1.tick_params(axis='x', which='minor', length=4)
    ax1.grid(ls='-.')
    st.pyplot(fig1)

    # --- Row 2: Three columns for panels 2, 3, 4 ---
    col1, col2, col3 = st.columns(3)

    with col1:
        # Panel 2: Profile at selected time
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(z_disp[:, time_idx], y_disp)
        ax2.axhline(y_disp[range_idx], color="red", linestyle="--")
        ax2.set_xlabel(colorbar_labels["panel1"], fontsize=fontsize)  # Use colorbar label from panel 1
        ax2.set_ylabel(f"Range ({units['range']})", fontsize=fontsize)
        ax2.tick_params(labelsize=fontsize-2)
        ax2.grid(ls='-.')
        ax2.set_title(f'Profile at {pd.to_datetime(time_values[time_idx]).strftime("%H:%M")}', fontsize=fontsize+2)
        st.pyplot(fig2)

    with col2:
        # Panel 3: Spectrogram
        vel, rng, specgram = load_spectrogram(files, ds_mom, prof_time_idx, spec_var)
        if vel is not None and rng is not None and specgram is not None:
            specgram = convert_to_db_if_linear(spec_var, ds0, specgram)
            # Find velocity indices where there is at least one non-NaN value
            valid_vel_mask = np.any(~np.isnan(specgram), axis=0)
            if np.any(valid_vel_mask):
                vmin_x = vel[np.argmax(valid_vel_mask)]
                vmax_x = vel[::-1][np.argmax(valid_vel_mask[::-1])]
                # Add margin
                margin = 0.05 * (vmax_x - vmin_x)
                vmin_x -= margin
                vmax_x += margin
            else:
                vmin_x, vmax_x = np.nanmin(vel), np.nanmax(vel)
            fig3, ax3 = plt.subplots(figsize=(4, 3.2))
            im3 = ax3.pcolormesh(
                vel, rng, specgram, shading="auto", cmap="turbo",
                vmin=clim["panel3"][0] if clim["panel3"] else None,
                vmax=clim["panel3"][1] if clim["panel3"] else None
            )
            ax3.set_xlim(vmin_x, vmax_x)
            ax3.axhline(rng[range_idx], color="red", linestyle="--")
            ax3.set_xlabel(f"Velocity ({units['velocity']})", fontsize=fontsize)
            ax3.set_ylabel(f"Range ({units['range']})", fontsize=fontsize)
            cbar = plt.colorbar(im3, ax=ax3, pad=0.01)
            cbar.set_label(colorbar_labels["panel3"], fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize-2)
            ax3.tick_params(labelsize=fontsize-2)
            ax3.grid(ls='-.')
            ax3.set_title(f'Spectrogram at {pd.to_datetime(time_values[prof_time_idx]).strftime("%H:%M")}', fontsize=fontsize+2)
            st.pyplot(fig3)

    with col3:
        # Panel 4: Single Spectrum at selected time/range
        vel, spectrum = load_spectrum(files, ds_mom, prof_time_idx, range_idx, spec_var)
        if vel is not None and spectrum is not None:
            spectrum = convert_to_db_if_linear(spec_var, ds0, spectrum)
            fig4, ax4 = plt.subplots(figsize=(4, 3))
            ax4.plot(vel, spectrum)
            ax4.set_xlabel(f"Velocity ({units['velocity']})", fontsize=fontsize)
            ax4.set_ylabel(colorbar_labels["panel3"], fontsize=fontsize)  # Use colorbar label from panel 3
            ax4.tick_params(labelsize=fontsize-2)
            ax4.grid(ls='-.')
            ax4.set_title(f'Spectrum at range {range_values[range_idx]:.1f} {units["range"]}', fontsize=fontsize+2)
            st.pyplot(fig4)

    # --- Simulation controls in sidebar ---
    with st.sidebar.expander("Simulation Parameters"):
        simulator_type = st.radio("Hydrometeor Type in Simulation", options=["Snow", "Rain"], index=0)
        #lut_path = st.text_input("LUT path", value="/project/meteo/work/L.Terzi/McRadarTest/LUT/liquid_273.15_35.6GHz_elv90.csv")
        lut_path_rain = st.text_input("Rain LUT path", value="/project/meteo/work/L.Terzi/McRadarTest/LUT/liquid_273.15_35.6GHz_elv90.csv")
        lut_path_snow = st.text_input("Snow LUT path", value="/project/meteo/work/L.Terzi/McRadarTest/LUT/vonTerzi_dendrite_LUT.nc")
        #log_R = st.slider("log₁₀(Rain rate R [mm/h])", min_value=-2.0, max_value=1.3, value=0.0, step=0.05)
        #R = 10 ** log_R
        #st.write(f"Rain rate R = {R:.2f} mm/h")
        # Gamma DSD parameters
        gamma = st.slider("Gamma DSD shape parameter (gamma)", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
        log_lam = st.slider("log₁₀(lambda) [m⁻¹]", min_value=2.4, max_value=4.0, value=3.0, step=0.01)
        lam = 10 ** log_lam
        st.write(f"lambda = {lam:.2f} m⁻¹")
        log_N0 = st.slider("log₁₀(N0) [m⁻³ mm⁻¹]", min_value=0.0, max_value=8.0, value=4.0, step=0.05)
        N0 = 10 ** log_N0
        st.write(f"N0 = {N0:.2e} m⁻³ mm⁻¹")
        log_eps = st.slider("log₁₀(eps_diss)", min_value=-5.0, max_value=-2.0, value=-3.0, step=0.1)
        eps_diss = 10 ** log_eps
        st.write(f"eps_diss = {eps_diss:.2e}")
        noise_pow = st.slider("Noise power [dB]", min_value=-60, max_value=0, value=-40, step=1)
        nave = st.slider("Averaging (nave)", min_value=1, max_value=100, value=10, step=1)
        theta_deg = st.slider("Beam width [deg]", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
        uwind = st.slider("U wind [m/s]", min_value=-20.0, max_value=20.0, value=0.0, step=0.1)
        time_int = st.slider("Integration time [s]", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        vertical_wind = st.slider("Vertical wind [m/s]", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
        
    # --- Row 3: Two columns for panels 5, 6 ---
    col5, col6 = st.columns(2)

    with col5:
        # Panel 5: Measured and Simulated Spectrum (Rain or Snow)
        vel, rng, specgram = load_spectrogram(files, ds_mom, prof_time_idx, spec_var)
        if vel is not None and rng is not None and specgram is not None:
            specgram = convert_to_db_if_linear(spec_var, ds0, specgram)
            measured_spectrum = specgram[range_idx, :]
            fig5, ax5 = plt.subplots(figsize=(5, 3))
            ax5.plot(vel, measured_spectrum, label="Measured")
            center_height = float(range_values[range_idx])
            try:
                vel_bins = radar_simulator._centers_to_edges(vel)
                if simulator_type == "Rain":
                    vel_sim, sim_H, _ = radar_simulator.simulate_rain_spectrum(
                        vel_bins=vel_bins,
                        #R=R,
                        center_height=center_height,
                        eps_diss=eps_diss,
                        noise_pow=noise_pow,
                        nave=nave,
                        theta_deg=theta_deg,
                        uwind=uwind,
                        time_int=time_int,
                        lut_path=lut_path_rain,
                        N0=N0,
                        gamma=gamma,
                        lam=lam,
                        vertical_wind=vertical_wind,
                    )
                    ax5.plot(vel_sim, sim_H, "--", label="Simulated (Rain)")
                else:
                    vel_sim, sim_H = radar_simulator.simulate_snow_spectrum(
                        vel_bins=vel_bins,
                        N0=N0,
                        lam=lam,
                        gamma=gamma,
                        center_height=center_height,
                        eps_diss=eps_diss,
                        noise_pow=noise_pow,
                        nave=nave,
                        theta_deg=theta_deg,
                        uwind=uwind,
                        time_int=time_int,
                        lut_path=lut_path_snow,
                        vertical_wind=vertical_wind,
                    )
                    ax5.plot(vel_sim, sim_H, "--", label="Simulated (Snow)")
            except Exception as e:
                st.warning(f"Simulation failed: {e}")

            ax5.set_xlabel(f"Velocity ({units['velocity']})", fontsize=fontsize)
            ax5.set_ylabel(colorbar_labels["panel3"], fontsize=fontsize)
            ax5.legend()
            ax5.grid(ls='-.')
            ax5.set_title(f"Measured & Simulated Spectrum ({simulator_type})", fontsize=fontsize+2)
            st.pyplot(fig5)
    with col6:
        # Panel 6: Simulated PSD (D vs PSD)
        #import matplotlib.ticker as mticker

        # Define D range (in mm)
        Dmax = np.linspace(1e-5, 50.0e-3, 1000)
        # Calculate PSD using gamma DSD
        PSD = N0 * (Dmax ** gamma) * np.exp(-lam * Dmax)

        fig6, ax6 = plt.subplots(figsize=(5, 3))
        ax6.plot(Dmax, PSD, label=f"Simulated PSD ({simulator_type})")
        ax6.set_xlabel("Diameter D [m]", fontsize=fontsize)
        ax6.set_ylabel("PSD [m$^{-3}$ m$^{-1}$]", fontsize=fontsize)
        ax6.set_yscale("log")
        ax6.set_xscale("log")
        ax6.grid(ls='-.')
        ax6.legend()
        ax6.set_title(f"Simulated PSD ({simulator_type})", fontsize=fontsize+2)
        # Optional: format y-axis for scientific notation
        #ax6.yaxis.set_major_formatter(mticker.ScalarFormatter())
        st.pyplot(fig6)
        
    st.markdown("""
    ---
    **Tips:**
    - Use the sliders to select time/range for all panels.
    - Panel 1 shows the time-range variable (e.g. reflectivity) with the selected point highlighted.
    - Panel 2 shows the vertical profile at the selected time.
    - Panel 3 shows the spectrogram at the selected time, with the selected range highlighted.
    - Panel 4 shows the spectrum at the selected time and range.
    - Panel 5 compares the measured spectrum with the simulated spectrum based on the selected parameters.
    - Panel 6 shows the simulated PSD based on the gamma DSD parameters.
    """)

if __name__ == "__main__":
    main()
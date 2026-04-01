import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import matplotlib.dates as mdates
from spectrawiz import radar_simulator
import plotly.graph_objects as go

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
    cmap_cfg = {
        "panel1": "turbo",
        "panel3": "turbo",
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
        #units["range"] = st.text_input("Range units (y-axis)", units["range"])
        #units["velocity"] = st.text_input("Velocity units (x-axis, panels 3/4)", units["velocity"])
        #units["time"] = st.text_input("Time units (panel 1 x-axis)", units["time"])
        #colorbar_labels["panel1"] = st.text_input("Colorbar label (panel 1)", colorbar_labels["panel1"])
        #colorbar_labels["panel3"] = st.text_input("Colorbar label (panel 3)", colorbar_labels["panel3"])
        available_cmaps = sorted(plt.colormaps())
        cmap_cfg["panel1"] = st.selectbox(
        "Colormap (panel 1)",
        options=available_cmaps,
        index=available_cmaps.index(cmap_cfg["panel1"]) if cmap_cfg["panel1"] in available_cmaps else 0
        )
        cmap_cfg["panel3"] = st.selectbox(
        "Colormap (panel 3)",
        options=available_cmaps,
        index=available_cmaps.index(cmap_cfg["panel3"]) if cmap_cfg["panel3"] in available_cmaps else 0
        )
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
    def downsample_heatmap(z, x, y, max_x=800, max_y=300):
        """
        Downsample a 2D heatmap z (shape: len(y) x len(x)) by stride.
        Returns downsampled arrays and the applied strides.
        """
        nx = len(x)
        ny = len(y)

        sx = max(1, int(np.ceil(nx / max_x)))
        sy = max(1, int(np.ceil(ny / max_y)))

        z_ds = z[::sy, ::sx]
        x_ds = x[::sx]
        y_ds = y[::sy]

        z_ds = np.asarray(z_ds, dtype=np.float32)

        return z_ds, x_ds, y_ds, sx, sy

    fontsize=14

    # --- Panel 1: Time-Height Plot ---
    z_disp = ds_mom[var].T.values
    z_disp = convert_to_db_if_linear(var, ds_mom, z_disp)
    time_values_disp = pd.to_datetime(ds_mom[time_var].values)
    y_disp = ds_mom[range_var].values

    #dpi=300
    #figsize_x = 12
    #width_px = figsize_x * dpi

    # fig1, ax1 = plt.subplots(figsize=(12, 3),constrained_layout=True)
    # im = ax1.pcolormesh(
    #     time_values_disp, y_disp, z_disp, shading="auto", cmap=cmap_cfg["panel1"],
    #     vmin=clim["panel1"][0] if clim["panel1"] else None,
    #     vmax=clim["panel1"][1] if clim["panel1"] else None
    # )
    # ax1.axvline(time_values_disp[time_idx], color="red", linestyle="--")
    # ax1.axhline(y_disp[range_idx], color="red", linestyle="--")
    # ax1.set_ylabel(f"Range ({units['range']})", fontsize=fontsize)
    # #ax1.set_xlabel(units["time"], fontsize=fontsize)
    # ax1.tick_params(labelsize=fontsize-2)
    # ax1.set_title(f'Time x Range {var}', fontsize=fontsize+2)
    # cbar = plt.colorbar(im, ax=ax1, pad=0.01)
    # print(f"Units for {var}:", {get_units_from_attrs(ds_mom[var])})
    # cbar.set_label(f'{var} ({get_units_from_attrs(ds_mom[var])})',fontsize=fontsize)#colorbar_labels["panel1"], fontsize=fontsize)

    # # Set x-axis tick format to H:M and minor ticks every hour
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # ax1.xaxis.set_minor_locator(mdates.HourLocator())
    # ax1.tick_params(axis='x', which='minor', length=4)
    # ax1.grid(ls='-.')
    # st.pyplot(fig1,width='content')
    # downsample for display
    
    z_ds, x_ds, y_ds, sx, sy = downsample_heatmap(z_disp, time_values_disp, y_disp, max_x=900, max_y=400)

    fig1 = go.Figure(
        data=go.Heatmap(
        x=x_ds,
        y=y_ds,
        z=z_ds.astype(np.float32),
        zsmooth=False,   # keep cells crisp and faster
        colorscale=cmap_cfg["panel1"],
        zmin=clim["panel1"][0] if clim["panel1"] else None,
        zmax=clim["panel1"][1] if clim["panel1"] else None,
        colorbar=dict(title=f"{var} ({get_units_from_attrs(ds_mom[var])})")
        )
        )
    dark_gray = "#4d4d4d"
    grid_gray_major = "rgba(77,77,77,0.45)"
    grid_gray_minor = "rgba(77,77,77,0.20)"
    fig1.update_layout(
        #title=dict(text=f"Time x Range {var}", font=dict(size=25, color=dark_gray)),
        yaxis_title=f"Range ({units['range']})",
        height=450,
        margin=dict(l=20, r=20, t=80, b=20),
        font=dict(size=20, color=dark_gray)
    )
    fig1.add_annotation(
        text=f"<b>Time x Range {var}</b>",
        x=0.5,
        xref="x domain",
        y=1.01,
        yref="paper",
        xanchor="center",
        yanchor="bottom",
        showarrow=False,
        font=dict(size=25, color="#4d4d4d"),
        )

    fig1.update_xaxes(
        showgrid=True,
        gridcolor=grid_gray_major,
        gridwidth=1.2,
        tickformat="%H:%M",
        dtick=3 * 60 * 60 * 1000,
        ticks="outside",
        ticklen=9,
        tickwidth=1.6,
        tickcolor=dark_gray,
        title_font=dict(size=24, color=dark_gray),
        tickfont=dict(size=20, color=dark_gray),
        showline=True,
        linecolor="#4d4d4d",
        linewidth=1.5,
        mirror=True,
        minor=dict(
            dtick=60 * 60 * 1000,
            ticks="outside",
            ticklen=5,
            tickwidth=1.2,
            tickcolor=dark_gray,
            showgrid=True,
            gridcolor=grid_gray_minor,
            gridwidth=0.8
        )
    )

    fig1.update_yaxes(
        showgrid=True,
        gridcolor=grid_gray_major,
        gridwidth=1.2,
        ticks="outside",
        ticklen=9,
        tickwidth=1.6,
        tickcolor=dark_gray,
        title_font=dict(size=24, color=dark_gray),
        tickfont=dict(size=20, color=dark_gray),
        showline=True,
        linecolor="#4d4d4d",
        linewidth=1.5,
        mirror=True,
        tickformat=".0f",
    )

    # Colorbar text/ticks also dark gray
    fig1.data[0].colorbar.title = dict(
        text=f"{var} ({get_units_from_attrs(ds_mom[var])})",
        font=dict(size=18, color=dark_gray)
    )
    fig1.data[0].colorbar.tickfont = dict(size=16, color=dark_gray)
    fig1.add_vline(x=time_values_disp[time_idx], line_dash="dash", line_color="red")
    fig1.add_hline(y=y_disp[range_idx], line_dash="dash", line_color="red")

    # fig1.update_layout(
    #     title=f"Time x Range {var}",
    #     xaxis_title="Time",
    #     yaxis_title=f"Range ({units['range']})",
    #     height=320,
    #     margin=dict(l=20, r=20, t=40, b=20)
    # )

    st.plotly_chart(fig1, use_container_width=True, config={"displaylogo": False})

    # --- Row 2: Three columns for panels 2, 3, 4 ---
    col1, col2, col3 = st.columns(3)

    with col1:
        fig2 = go.Figure()

        # Profile line
        fig2.add_trace(
            go.Scattergl(
                x=z_disp[:, time_idx],
                y=y_disp,
                mode="lines",
                name="Profile",
                line=dict(color="#1f77b4", width=2),
            )
        )

        # Selected range marker line
        fig2.add_hline(y=y_disp[range_idx], line_dash="dash", line_color="red")

        # Match style to Panel 1 dark gray look
        dark_gray = "#4d4d4d"
        grid_gray = "rgba(77,77,77,0.35)"

        fig2.update_layout(
            #title=dict(text=f"Profile at {pd.to_datetime(time_values[time_idx]).strftime('%H:%M')}", font=dict(size=25, color=dark_gray)),
            #xaxis_title=colorbar_labels["panel1"],
            #xaxis_title=dict(title=f"{var} ({get_units_from_attrs(ds_mom[var])})"),
            xaxis_title=f"{var} ({get_units_from_attrs(ds_mom[var])})",
            yaxis_title=f"Range ({units['range']})",
            height=450,
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(size=14, color=dark_gray),
            showlegend=False,
        )

        fig2.add_annotation(
            text=f"<b>Profile at {pd.to_datetime(time_values[time_idx]).strftime('%H:%M')} </b>",
            x=0.5,
            xref="x domain",
            y=1.01,
            yref="paper",
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=25, color="#4d4d4d"),
        )

        fig2.update_xaxes(
            showgrid=True,
            gridcolor=grid_gray,
            ticks="outside",
            tickcolor=dark_gray,
            title_font=dict(size=24, color=dark_gray),
            tickfont=dict(size=20, color=dark_gray),
            showline=True,
            linecolor="#4d4d4d",
            linewidth=1.5,
            mirror=True
        )

        fig2.update_yaxes(
            showgrid=True,
            gridcolor=grid_gray,
            ticks="outside",
            tickcolor=dark_gray,
            title_font=dict(size=24, color=dark_gray),
            tickfont=dict(size=20, color=dark_gray),
            showline=True,
            linecolor="#4d4d4d",
            linewidth=1.5,
            mirror=True,
            tickformat=".0f"
        )

        st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})

    with col2:
        # Panel 3: Spectrogram (Plotly)
        vel, rng, specgram = load_spectrogram(files, ds_mom, prof_time_idx, spec_var)
        if vel is not None and rng is not None and specgram is not None:
            specgram = convert_to_db_if_linear(spec_var, ds0, specgram)

            # Match current x-limits logic from Matplotlib version
            valid_vel_mask = np.any(~np.isnan(specgram), axis=0)
            if np.any(valid_vel_mask):
                vmin_x = vel[np.argmax(valid_vel_mask)]
                vmax_x = vel[::-1][np.argmax(valid_vel_mask[::-1])]
                margin = 0.05 * (vmax_x - vmin_x)
                vmin_x -= margin
                vmax_x += margin
            else:
                vmin_x, vmax_x = np.nanmin(vel), np.nanmax(vel)

            fig3 = go.Figure(
                data=go.Heatmap(
                    x=vel,
                    y=rng,
                    z=specgram,
                    colorscale=cmap_cfg["panel3"],
                    zmin=clim["panel3"][0] if clim["panel3"] else None,
                    zmax=clim["panel3"][1] if clim["panel3"] else None,
                    colorbar=dict(title=colorbar_labels["panel3"]),
                )
            )

            dark_gray = "#4d4d4d"
            grid_gray_major = "rgba(77,77,77,0.45)"

            fig3.update_layout(
                # title=dict(
                #     text=f"Spectrogram at {pd.to_datetime(time_values[prof_time_idx]).strftime('%H:%M')}",
                #     font=dict(size=25, color=dark_gray),
                # ),
                xaxis_title=f"Velocity ({units['velocity']})",
                yaxis_title=f"Range ({units['range']})",
                height=450,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(size=20, color=dark_gray),
            )
            fig3.add_annotation(
                text=f"<b>Spectrogram at {pd.to_datetime(time_values[time_idx]).strftime('%H:%M')} </b>",
                x=0.5,
                xref="x domain",
                y=1.01,
                yref="paper",
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=25, color="#4d4d4d"),
            )

            fig3.update_xaxes(
                range=[vmin_x, vmax_x],
                showgrid=True,
                gridcolor=grid_gray_major,
                gridwidth=1.2,
                ticks="outside",
                ticklen=9,
                tickwidth=1.6,
                tickcolor=dark_gray,
                title_font=dict(size=24, color=dark_gray),
                tickfont=dict(size=20, color=dark_gray),
                showline=True,
                linecolor=dark_gray,
                linewidth=1.5,
                mirror=True,
            )

            fig3.update_yaxes(
                showgrid=True,
                gridcolor=grid_gray_major,
                gridwidth=1.2,
                ticks="outside",
                ticklen=9,
                tickwidth=1.6,
                tickcolor=dark_gray,
                title_font=dict(size=24, color=dark_gray),
                tickfont=dict(size=20, color=dark_gray),
                showline=True,
                tickformat=".0f",
                linecolor=dark_gray,
                linewidth=1.5,
                mirror=True,
            )

            # Horizontal selected-range marker
            fig3.add_hline(y=rng[range_idx], line_dash="dash", line_color="red")

            # Colorbar style
            # Colorbar text/ticks also dark gray
            fig3.data[0].colorbar.title = dict(
                text=f"{var} ({get_units_from_attrs(ds_mom[var])})",
                font=dict(size=18, color=dark_gray)
            )
            fig3.data[0].colorbar.tickfont = dict(size=16, color=dark_gray)

            st.plotly_chart(fig3, width='stretch', config={"displaylogo": False})

    with col3:
        # Panel 4: Single Spectrum at selected time/range (Plotly)
        vel, spectrum = load_spectrum(files, ds_mom, prof_time_idx, range_idx, spec_var)
        if vel is not None and spectrum is not None:
            spectrum = convert_to_db_if_linear(spec_var, ds0, spectrum)
            print(spectrum)
            #print(np.nanargmin(spectrum))
            #print(vel[np.nanargmin(spectrum)])
            #print(vel.where(~np.isnan(spectrum)))
            mask = ~np.isnan(spectrum)
            #print(vel[mask])
            spectrum = spectrum[mask]
            vel = vel[mask]
            fig4 = go.Figure()
            fig4.add_trace(
                go.Scattergl(
                    x=vel,
                    y=spectrum,
                    mode="lines",
                    name="Spectrum",
                    line=dict(color="#1f77b4", width=2),
                )
            )

            dark_gray = "#4d4d4d"
            grid_gray_major = "rgba(77,77,77,0.45)"

            fig4.update_layout(
                # title=dict(
                #     text=f"Spectrum at range {range_values[range_idx]:.1f} {units['range']}",
                #     font=dict(size=25, color=dark_gray),
                # ),
                xaxis_title=f"Velocity ({units['velocity']})",
                #yaxis_title=colorbar_labels["panel3"],
                yaxis_title=f"{var} ({get_units_from_attrs(ds_mom[var])})",
                height=450,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(size=20, color=dark_gray),
                showlegend=False,
            )

            fig4.add_annotation(
                text=f"<b>Spectrum at range {range_values[range_idx]:.1f} {units['range']}</b>",
                x=0.5,
                xref="x domain",
                y=1.01,
                yref="paper",
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=25, color="#4d4d4d"),
            )
            

            fig4.update_xaxes(
                showgrid=True,
                gridcolor=grid_gray_major,
                gridwidth=1.2,
                ticks="outside",
                ticklen=9,
                tickwidth=1.6,
                tickcolor=dark_gray,
                title_font=dict(size=24, color=dark_gray),
                tickfont=dict(size=20, color=dark_gray),
                showline=True,
                linecolor=dark_gray,
                linewidth=1.5,
                mirror=True,
            )

            fig4.update_yaxes(
                showgrid=True,
                gridcolor=grid_gray_major,
                gridwidth=1.2,
                ticks="outside",
                ticklen=9,
                tickwidth=1.6,
                tickcolor=dark_gray,
                title_font=dict(size=24, color=dark_gray),
                tickfont=dict(size=20, color=dark_gray),
                showline=True,
                linecolor=dark_gray,
                linewidth=1.5,
                mirror=True,
            )

            st.plotly_chart(fig4, width='stretch', config={"displaylogo": False})

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
        # Panel 5: Measured and Simulated Spectrum (Rain or Snow) - Plotly
        vel, rng, specgram = load_spectrogram(files, ds_mom, prof_time_idx, spec_var)
        if vel is not None and rng is not None and specgram is not None:
            specgram = convert_to_db_if_linear(spec_var, ds0, specgram)
            measured_spectrum = specgram[range_idx, :]

            fig5 = go.Figure()
            fig5.add_trace(
                go.Scattergl(
                    x=vel,
                    y=measured_spectrum,
                    mode="lines",
                    name="Measured",
                    line=dict(color="#1f77b4", width=2),
                )
            )

            center_height = float(range_values[range_idx])
            try:
                vel_bins = radar_simulator._centers_to_edges(vel)
                if simulator_type == "Rain":
                    vel_sim, sim_H, _ = radar_simulator.simulate_rain_spectrum(
                        vel_bins=vel_bins,
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
                    sim_label = "Simulated (Rain)"
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
                    sim_label = "Simulated (Snow)"

                fig5.add_trace(
                    go.Scattergl(
                        x=vel_sim,
                        y=sim_H,
                        mode="lines",
                        name=sim_label,
                        line=dict(color="#d62728", width=2, dash="dash"),
                    )
                )
            except Exception as e:
                st.warning(f"Simulation failed: {e}")

            dark_gray = "#4d4d4d"
            grid_gray_major = "rgba(77,77,77,0.45)"

            fig5.update_layout(
                # title=dict(
                #     text=f"Measured & Simulated Spectrum ({simulator_type})",
                #     font=dict(size=25, color=dark_gray),
                # ),
                xaxis_title=f"Velocity ({units['velocity']})",
                #yaxis_title=colorbar_labels["panel3"],
                yaxis_title=f"{var} ({get_units_from_attrs(ds_mom[var])})",
                height=450,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(size=20, color=dark_gray),
                legend=dict(
                            x=0.02,           # inside-left
                            y=0.98,           # inside-top
                            xanchor="left",
                            yanchor="top",
                            bgcolor="rgba(255,255,255,0.75)",
                            bordercolor=dark_gray,
                            borderwidth=1,
                            font=dict(size=16, color=dark_gray),
                        ),
            )

            fig5.add_annotation(
                text=f"<b>Measured & Simulated Spectrum ({simulator_type})</b>",
                x=0.5,
                xref="x domain",
                y=1.01,
                yref="paper",
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=25, color="#4d4d4d"),
            )

            fig5.update_xaxes(
                showgrid=True,
                gridcolor=grid_gray_major,
                gridwidth=1.2,
                ticks="outside",
                ticklen=9,
                tickwidth=1.6,
                tickcolor=dark_gray,
                title_font=dict(size=24, color=dark_gray),
                tickfont=dict(size=20, color=dark_gray),
                showline=True,
                linecolor=dark_gray,
                linewidth=1.5,
                mirror=True,
            )

            fig5.update_yaxes(
                showgrid=True,
                gridcolor=grid_gray_major,
                gridwidth=1.2,
                ticks="outside",
                ticklen=9,
                tickwidth=1.6,
                tickcolor=dark_gray,
                title_font=dict(size=24, color=dark_gray),
                tickfont=dict(size=20, color=dark_gray),
                showline=True,
                linecolor=dark_gray,
                linewidth=1.5,
                mirror=True,
            )

            st.plotly_chart(fig5, width="content", config={"displaylogo": False})
    with col6:
        # Panel 6: Simulated PSD (D vs PSD) - Plotly
        Dmax = np.linspace(1e-5, 50.0e-3, 1000)
        PSD = N0 * (Dmax ** gamma) * np.exp(-lam * Dmax)

        fig6 = go.Figure()
        fig6.add_trace(
            go.Scatter(
                x=Dmax,
                y=PSD,
                mode="lines",
                name=f"Simulated PSD ({simulator_type})",
                line=dict(color="#1f77b4", width=2),
            )
        )

        dark_gray = "#4d4d4d"
        grid_gray_major = "rgba(77,77,77,0.45)"

        fig6.update_layout(
            # title=dict(
            #     text=f"Simulated PSD ({simulator_type})",
            #     font=dict(size=25, color=dark_gray),
            # ),
            xaxis_title="Diameter D [m]",
            yaxis_title="PSD [m<sup>-3</sup> m<sup>-1</sup>]",
            height=450,
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(size=20, color=dark_gray),
            legend=dict(font=dict(size=16, color=dark_gray)),
        )

        fig6.add_annotation(
            text=f"<b>Simulated PSD ({simulator_type})</b>",
            x=0.5,
            xref="x domain",
            y=1.01,
            yref="paper",
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=25, color="#4d4d4d"),
        )

        fig6.update_xaxes(
            type="log",
            showgrid=True,
            gridcolor=grid_gray_major,
            gridwidth=1.2,
            ticks="outside",
            ticklen=9,
            tickwidth=1.6,
            tickcolor=dark_gray,
            title_font=dict(size=24, color=dark_gray),
            tickfont=dict(size=20, color=dark_gray),
            showline=True,
            linecolor=dark_gray,
            linewidth=1.5,
            mirror=True,
        )

        fig6.update_yaxes(
            type="log",
            showgrid=True,
            gridcolor=grid_gray_major,
            gridwidth=1.2,
            ticks="outside",
            ticklen=9,
            tickwidth=1.6,
            tickcolor=dark_gray,
            title_font=dict(size=24, color=dark_gray),
            tickfont=dict(size=20, color=dark_gray),
            showline=True,
            linecolor=dark_gray,
            linewidth=1.5,
            mirror=True,
        )

        st.plotly_chart(fig6, width="content", config={"displaylogo": False})
        
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
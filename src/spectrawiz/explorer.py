import streamlit as st
import xarray as xr
import numpy as np
import glob
import os
import pandas as pd
from spectrawiz import radar_simulator
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import warnings

from plotly.subplots import make_subplots

def mpl_to_plotly(cmap_name, n=255):
    """Convert a Matplotlib colormap to Plotly colorscale."""
    cmap = matplotlib.cm.get_cmap(cmap_name, n)
    colorscale = []
    for i in range(cmap.N):
        rgba = cmap(i)
        colorscale.append([
            i / (cmap.N - 1),
            f'rgb({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)})'
        ])
    return colorscale

def main():
    warnings.filterwarnings("ignore")
    print('###############################################################')
    print('new version of explorer.py loaded')
    print('###############################################################')

    st.set_page_config(layout="wide")

    # --- Title and logo using Streamlit columns ---
    col_title, col_logo = st.columns([8, 1])
    with col_title:
        st.markdown("<h1 style='margin-bottom: 0.2em;'>SpectraWiz: Interactive Radar Spectra Visualization</h1>", unsafe_allow_html=True)
    with col_logo:
        st.image("static/mim_logo_gross.png", width=160)

    # --- MIM Logo at top right using HTML/CSS only, from static folder ---
    st.markdown(
        """
        <div style="position: fixed; top: 1.5rem; right: 2.5rem; z-index: 9999;">
            <img src="/static/mim_logo_gross.png" alt="MIM Logo" style="height:54px; width:auto; box-shadow: 0 0 6px #fff; background: #fff; border-radius: 8px;"
                 onerror="this.style.display='none'; document.getElementById('logo-error').style.display='block';">
        </div>
        <div id="logo-error" style="display:none; position: fixed; top: 1.5rem; right: 2.5rem; z-index: 9999; color: red; background: #fff; padding: 6px 12px; border-radius: 8px; box-shadow: 0 0 6px #fff;">
            Logo not found: <a href='/static/mim_logo_gross.png' target='_blank'>/static/mim_logo_gross.png</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- User Inputs ---
    datapath = st.sidebar.text_input("Data directory", "processed/")
    date = st.sidebar.text_input("Date (YYYY-MM-DD)", "2025-09-10")
    pattern = st.sidebar.text_input("File pattern", "*rpg_hourly_proc.nc")

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
    xlim = {
        "panel1": None,
        "panel3": None,
    }
    ylim = {
        "panel1": None,
        "panel3": None,
    }
    cmap_cfg = {
        "panel1": "Spectral",
        "panel3": "Spectral",
    }

    # --- Sliders for selection (always visible and at the top) ---
    if "time_idx" not in st.session_state:
        st.session_state.time_idx = len(time_values) // 2
    if "range_idx" not in st.session_state:
        st.session_state.range_idx = len(range_values) // 2
    if "box_xlim" not in st.session_state:
        st.session_state.box_xlim = None
    if "box_ylim" not in st.session_state:
        st.session_state.box_ylim = None
    for _pkey in ["p2", "p3"]:
        if f"box_xlim_{_pkey}" not in st.session_state:
            st.session_state[f"box_xlim_{_pkey}"] = None
        if f"box_ylim_{_pkey}" not in st.session_state:
            st.session_state[f"box_ylim_{_pkey}"] = None

    # Custom CSS for compact step buttons
    st.sidebar.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            padding: 0.15rem 0.3rem !important;
            min-height: 1.8rem;
            font-size: 1rem;
            border-radius: 6px;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] div,
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] p {
            width: 100% !important;
            text-align: center !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("**Time**")
    t_col1, t_col2, t_col3 = st.sidebar.columns([1, 8, 1], vertical_alignment="center")
    with t_col1:
        if st.button("◂", key="time_prev", use_container_width=True):
            st.session_state.time_idx = max(0, st.session_state.time_idx - 1)
    with t_col3:
        if st.button("▸", key="time_next", use_container_width=True):
            st.session_state.time_idx = min(len(time_values) - 1, st.session_state.time_idx + 1)
    with t_col2:
        time_idx = st.select_slider(
            "Time",
            options=list(range(len(time_values))),
            value=st.session_state.time_idx,
            format_func=lambda i: pd.to_datetime(time_values[i]).strftime('%Y-%m-%d %H:%M'),
            label_visibility="collapsed",
            key="time_slider",
        )
    st.session_state.time_idx = time_idx

    st.sidebar.markdown("**Range**")
    r_col1, r_col2, r_col3 = st.sidebar.columns([1, 8, 1], vertical_alignment="center")
    with r_col1:
        if st.button("◂", key="range_prev", use_container_width=True):
            st.session_state.range_idx = max(0, st.session_state.range_idx - 1)
    with r_col3:
        if st.button("▸", key="range_next", use_container_width=True):
            st.session_state.range_idx = min(len(range_values) - 1, st.session_state.range_idx + 1)
    with r_col2:
        range_idx = st.select_slider(
            "Range",
            options=list(range(len(range_values))),
            value=st.session_state.range_idx,
            format_func=lambda i: f"{range_values[i]:.1f} {units['range']}",
            label_visibility="collapsed",
            key="range_slider",
        )
    st.session_state.range_idx = range_idx

    prof_time_idx = time_idx

    # --- Display/Units/Colorbar config dictionary (can override defaults) ---
    with st.sidebar.expander("Display Options"):
        def _parse_time(s, date_ref):
            """Accept HH:MM, HH:MM:SS, or full timestamp; prepend date_ref date if no date given."""
            s = s.strip()
            # If string contains no date part (no '-' or '/'), prepend the reference date
            if "-" not in s and "/" not in s:
                s = f"{date_ref.strftime('%Y-%m-%d')} {s}"
            return pd.Timestamp(s)

        mpl_cmaps = sorted(
            [m for m in matplotlib.colormaps if not m.endswith("_r")],
            key=str.casefold,
        )

        panel1_xlimits = st.text_input(
            "X-axis limits (e.g. 06:00,12:00 or 2025-09-10 06:00,2025-09-10 12:00)",
            "",
            key="panel1_xlimits",
        )
        panel1_ylimits = st.text_input("Y-axis limits (e.g. 0,3000)", "", key="panel1_ylimits")
        selected_cmap_panel1 = st.selectbox(
            "Colormap",
            options=mpl_cmaps,
            index=mpl_cmaps.index("Spectral"),
            key="panel1_cmap",
        )
        panel1_colorbar_limits = st.text_input(
            "Colorbar limits (e.g. 0,30)",
            "",
            key="panel1_colorbar_limits",
        )

        panel3_xlimits = st.text_input("X-axis limits (e.g. -10,10)", "", key="panel3_xlimits")
        panel3_ylimits = st.text_input("Y-axis limits (e.g. 0,3000)", "", key="panel3_ylimits")
        selected_cmap_panel3 = st.selectbox(
            "Colormap",
            options=mpl_cmaps,
            index=mpl_cmaps.index("Spectral"),
            key="panel3_cmap",
        )
        panel3_colorbar_limits = st.text_input(
            "Colorbar limits (e.g. 0,1)",
            "",
            key="panel3_colorbar_limits",
        )

        st.markdown("---")
        st.markdown("**Panel 5 (Meas. & Sim. Spectrum)**")
        panel5_xlimits = st.text_input("X-axis limits (e.g. -4,2)", "", key="panel5_xlimits")
        panel5_ylimits = st.text_input("Y-axis limits (e.g. -40,10)", "", key="panel5_ylimits")

        xlimits = {
            "panel1": panel1_xlimits,
            "panel3": panel3_xlimits,
            "panel5": panel5_xlimits,
        }
        ylimits = {
            "panel1": panel1_ylimits,
            "panel3": panel3_ylimits,
            "panel5": panel5_ylimits,
        }
        colorbar_limits = {
            "panel1": panel1_colorbar_limits,
            "panel3": panel3_colorbar_limits,
        }

        _date_ref = pd.Timestamp(time_values[0])
        for key, val in xlimits.items():
            if val:
                try:
                    a, b = [s.strip() for s in val.split(",", 1)]
                    if key == "panel1":
                        xlim[key] = (_parse_time(a, _date_ref), _parse_time(b, _date_ref))
                    else:
                        xlim[key] = (float(a), float(b))
                except Exception:
                    xlim[key] = None
            else:
                xlim[key] = None

        for key, val in ylimits.items():
            if val:
                try:
                    ymin, ymax = [float(s.strip()) for s in val.split(",")]
                    ylim[key] = (ymin, ymax)
                except Exception:
                    ylim[key] = None
            else:
                ylim[key] = None

        for key, val in colorbar_limits.items():
            if val:
                try:
                    vmin, vmax = [float(x) for x in val.split(",")]
                    clim[key] = (vmin, vmax)
                except Exception:
                    clim[key] = None
            else:
                clim[key] = None

    # --- Simulation controls in sidebar ---
    with st.sidebar.expander("Simulation Parameters"):
        simulator_type = st.radio("Hydrometeor Type in Simulation", options=["Snow", "Rain"], index=0)
        lut_path_rain = st.text_input("Rain LUT path", value="/project/meteo/work/L.Terzi/spectrawiz/scattering_luts/liquid_LUT.nc")
        lut_path_snow = st.text_input("Snow LUT path", value="/project/meteo/work/L.Terzi/spectrawiz/scattering_luts/ice_LUT_vonTerzi_dendrite.nc")
        freq_ghz = st.selectbox("Radar Frequency [GHz]", options=[9.6, 35.6, 94.0], index=2)
        gamma = st.slider("Gamma DSD shape parameter (gamma)", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
        log_lam = st.slider("log₁₀(lambda) [m⁻¹]", min_value=2.0, max_value=5.0, value=3.0, step=0.01)
        lam = 10 ** log_lam
        st.write(f"lambda = {lam:.2f} m⁻¹")
        log_N0 = st.slider("log₁₀(N0) [m⁻³ mm⁻¹]", min_value=0.0, max_value=8.0, value=4.0, step=0.05)
        N0 = 10 ** log_N0
        st.write(f"N0 = {N0:.2e} m⁻³ mm⁻¹")
        log_eps = st.slider("log₁₀(eps_diss)", min_value=-5.0, max_value=-2.0, value=-3.0, step=0.1)
        eps_diss = 10 ** log_eps
        st.write(f"eps_diss = {eps_diss:.2e}")
        uwind = st.slider("Horizontal wind [m/s]", min_value=-20.0, max_value=20.0, value=0.0, step=0.1)
        vertical_wind = st.slider("Vertical wind [m/s]", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
        noise_pow = st.slider("Noise power [dB]", min_value=-60, max_value=0, value=-40, step=1)
        nave = st.slider("Averaging (nave)", min_value=1, max_value=100, value=10, step=1)
        theta_deg = st.slider("Beam width [deg]", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
        time_int = st.slider("Integration time [s]", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        
    def get_units_from_attrs(var):
        """Get units from variable attrs, case-insensitive for 'unit' or 'units'."""
        for key in var.attrs:
            if key.lower() in ["unit", "units"]:
                return var.attrs[key]
        return ""

    # def convert_to_db_if_linear(varname, ds, values):
    #     """
    #     If units suggest linear reflectivity, convert to dB (10*log10).
    #     Only convert if not already in dB.
    #     """
    #     try:
    #         var = ds[varname]
    #         var_units = get_units_from_attrs(var)
    #     except Exception:
    #         var_units = ""
    #     print(var_units)
    #     var_units_lc = str(var_units).lower().replace(" ", "")
    #     print('var_units_lc:', var_units_lc)
    #     # If already in dB, do nothing
    #     if any(x in var_units_lc for x in ["db", "dbz", "dbm"]):
    #         return values
    #     # Typical linear reflectivity units
    #     linear_patterns = ["mm6", "mm^6", "mm6/m3", "mm^6/m^3", "mm6m-3", "mm^6m^-3", "mm6 m-3 (m s-1)-1"]
    #     if any(pat in var_units_lc for pat in linear_patterns):
    #         values = np.where(values > 0, values, np.nan)
    #         ds[varname].attrs["units"] = "dB"
    #         return 10 * np.log10(values)
    #     return values
    def convert_to_db_if_linear(varname, ds, values):
        """
        If units suggest linear reflectivity, convert to dB (10*log10).
        Returns converted values and the display unit string.
        """
        # Prefer units from the values array itself, fall back to ds
        var_units = ""
        if hasattr(values, "attrs"):
            var_units = get_units_from_attrs(values)
        if not var_units:
            try:
                var = ds[varname]
                var_units = get_units_from_attrs(var)
            except Exception:
                var_units = ""
        values_out = np.asarray(values)
        var_units_lc = str(var_units).lower().replace(" ", "")

        # If already in dB, do nothing
        if any(x in var_units_lc for x in ["db", "dbz", "dbm"]):
            return values_out, (var_units if var_units else "dB")

        # Typical linear reflectivity units
        linear_patterns = ["mm6", "mm^6", "mm6/m3", "mm^6/m^3", "mm6m-3", "mm^6m^-3", "mm6 m-3 (m s-1)-1"]
        if any(pat in var_units_lc for pat in linear_patterns):
            values_out = np.where(values_out > 0, values_out, np.nan)
            return 10 * np.log10(values_out), "dB"

        return values_out, str(var_units)
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
        nx = len(x)
        ny = len(y)
        sx = max(1, int(np.ceil(nx / max_x)))
        sy = max(1, int(np.ceil(ny / max_y)))
        z_ds = z[::sy, ::sx]
        x_ds = x[::sx]
        y_ds = y[::sy]
        z_ds = np.asarray(z_ds, dtype=np.float32)
        return z_ds, x_ds, y_ds, sx, sy

    def subset_heatmap(z, x, y, x_range=None, y_range=None):
        x_mask = np.ones(len(x), dtype=bool)
        y_mask = np.ones(len(y), dtype=bool)

        if x_range:
            x_start = pd.Timestamp(x_range[0])
            x_end = pd.Timestamp(x_range[1])
            x_mask = (x >= x_start) & (x <= x_end)
        if y_range:
            y_start, y_end = y_range
            y_min, y_max = sorted((y_start, y_end))
            y_mask = (y >= y_min) & (y <= y_max)

        if not np.any(x_mask):
            x_mask[:] = True
        if not np.any(y_mask):
            y_mask[:] = True

        z_sub = z[np.ix_(y_mask, x_mask)]
        x_sub = x[x_mask]
        y_sub = y[y_mask]
        return z_sub, x_sub, y_sub

    fontsize=14

    # --- Panel 1: Time-Height Plot ---
    z_disp = ds_mom[var].T.values
    z_disp, panel1_unit = convert_to_db_if_linear(var, ds_mom, z_disp)
    time_values_disp = pd.to_datetime(ds_mom[time_var].values)
    y_disp = ds_mom[range_var].values

    # Merge box selection with text-input limits (box selection takes priority)
    _effective_xrange = st.session_state.box_xlim or xlim["panel1"]
    _effective_yrange = st.session_state.box_ylim or ylim["panel1"]

    z_panel1, time_panel1, y_panel1 = subset_heatmap(
        z_disp,
        time_values_disp,
        y_disp,
        x_range=_effective_xrange,
        y_range=_effective_yrange,
    )

    z_ds, x_ds, y_ds, sx, sy = downsample_heatmap(
        z_panel1,
        time_panel1,
        y_panel1,
        max_x=900,
        max_y=400,
    )

    fig1 = go.Figure(
        data=go.Heatmap(
            x=x_ds,
            y=y_ds,
            z=z_ds.astype(np.float32),
            zsmooth=False,
            colorscale=mpl_to_plotly(selected_cmap_panel1),
            zmin=clim["panel1"][0] if clim["panel1"] else None,
            zmax=clim["panel1"][1] if clim["panel1"] else None,
            colorbar=dict(title=f"{var} ({panel1_unit})" if panel1_unit else var)
        )
    )
    dark_gray = "#4d4d4d"
    grid_gray_major = "rgba(77,77,77,0.45)"
    grid_gray_minor = "rgba(77,77,77,0.20)"
    fig1.update_layout(
        yaxis_title=f"Range ({units['range']})",
        height=450,
        margin=dict(l=20, r=20, t=80, b=20),
        font=dict(size=20, color=dark_gray),
        dragmode="select",
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

    # Adaptive tick spacing based on visible time range
    _vis_xlim = st.session_state.box_xlim or xlim.get("panel1")
    if _vis_xlim:
        _dt_seconds = (pd.Timestamp(_vis_xlim[1]) - pd.Timestamp(_vis_xlim[0])).total_seconds()
    else:
        _dt_seconds = (time_values_disp[-1] - time_values_disp[0]).total_seconds()
    if _dt_seconds < 600:           # < 10 min
        _major_dtick = 60 * 1000
        _minor_dtick = 15 * 1000
        _tfmt = "%H:%M:%S"
    elif _dt_seconds < 3600:        # < 1 h
        _major_dtick = 10 * 60 * 1000
        _minor_dtick = 2 * 60 * 1000
        _tfmt = "%H:%M"
    elif _dt_seconds < 3 * 3600:    # < 3 h
        _major_dtick = 30 * 60 * 1000
        _minor_dtick = 10 * 60 * 1000
        _tfmt = "%H:%M"
    elif _dt_seconds < 12 * 3600:   # < 12 h
        _major_dtick = 1 * 60 * 60 * 1000
        _minor_dtick = 15 * 60 * 1000
        _tfmt = "%H:%M"
    else:
        _major_dtick = 3 * 60 * 60 * 1000
        _minor_dtick = 60 * 60 * 1000
        _tfmt = "%H:%M"

    fig1.update_xaxes(
        autorange=True,
        showgrid=True,
        gridcolor=grid_gray_major,
        gridwidth=1.2,
        tickformat=_tfmt,
        dtick=_major_dtick,
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
            dtick=_minor_dtick,
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
        autorange=True,
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

    fig1.data[0].colorbar.title = dict(
        text=f"{var} ({panel1_unit})" if panel1_unit else var,
        font=dict(size=18, color=dark_gray)
    )
    fig1.data[0].colorbar.tickfont = dict(size=16, color=dark_gray)

    # Clip crosshairs to the visible data range
    _vline_time = time_values_disp[time_idx]
    _hline_range = y_disp[range_idx]
    if time_panel1[0] <= _vline_time <= time_panel1[-1]:
        fig1.add_vline(x=_vline_time, line_dash="dash", line_color="red")
    if y_panel1[0] <= _hline_range <= y_panel1[-1]:
        fig1.add_hline(y=_hline_range, line_dash="dash", line_color="red")

    event = st.plotly_chart(
        fig1, use_container_width=True,
        config={"displaylogo": False},
        on_select="rerun",
        key="panel1_select",
    )
    if event and event.selection and event.selection.get("box"):
        box = event.selection["box"][0]
        st.session_state.box_xlim = (min(box["x"]), max(box["x"]))
        st.session_state.box_ylim = (min(box["y"]), max(box["y"]))
        st.rerun()

    if st.session_state.box_xlim or st.session_state.box_ylim:
        if st.button("Reset zoom", key="reset_panel1_zoom"):
            st.session_state.box_xlim = None
            st.session_state.box_ylim = None
            st.rerun()

    # --- Row 2: Three columns for panels 2, 3, 4 ---
    col1, col2, col3 = st.columns(3)

    with col1:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scattergl(
                x=z_disp[:, time_idx],
                y=y_disp,
                mode="lines",
                name="Profile",
                line=dict(color="#1f77b4", width=2),
            )
        )
        fig2.add_hline(y=y_disp[range_idx], line_dash="dash", line_color="red")
        dark_gray = "#4d4d4d"
        grid_gray = "rgba(77,77,77,0.35)"
        fig2.update_layout(
            xaxis_title=f"{var} ({panel1_unit})" if panel1_unit else var,
            yaxis_title=f"Range ({units['range']})",
            height=450,
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(size=14, color=dark_gray),
            showlegend=False,
            dragmode="select",
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
            range=list(ylim["panel3"]) if ylim["panel3"] else None,
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
        if st.session_state.box_xlim_p2:
            fig2.update_xaxes(range=list(st.session_state.box_xlim_p2))
        if st.session_state.box_ylim_p2:
            fig2.update_yaxes(range=list(st.session_state.box_ylim_p2))
        ev2 = st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False}, on_select="rerun", key="panel2_select")
        if ev2 and ev2.selection and ev2.selection.get("box"):
            b = ev2.selection["box"][0]
            st.session_state.box_xlim_p2 = (min(b["x"]), max(b["x"]))
            st.session_state.box_ylim_p2 = (min(b["y"]), max(b["y"]))
            st.rerun()
        if st.session_state.box_xlim_p2 or st.session_state.box_ylim_p2:
            if st.button("Reset zoom", key="reset_p2"):
                st.session_state.box_xlim_p2 = None
                st.session_state.box_ylim_p2 = None
                st.rerun()

    with col2:
        vel, rng, specgram = load_spectrogram(files, ds_mom, prof_time_idx, spec_var)
        if vel is not None and rng is not None and specgram is not None:
            specgram, specgram_unit = convert_to_db_if_linear(spec_var, ds0, specgram)
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
                    colorscale=mpl_to_plotly(selected_cmap_panel3),
                    zmin=clim["panel3"][0] if clim["panel3"] else None,
                    zmax=clim["panel3"][1] if clim["panel3"] else None,
                    colorbar=dict(title=colorbar_labels["panel3"]),
                )
            )
            dark_gray = "#4d4d4d"
            grid_gray_major = "rgba(77,77,77,0.45)"
            fig3.update_layout(
                xaxis_title=f"Velocity ({units['velocity']})",
                yaxis_title=f"Range ({units['range']})",
                height=450,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(size=20, color=dark_gray),
                dragmode="select",
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
                range=list(xlim["panel3"]) if xlim["panel3"] else [vmin_x, vmax_x],
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
                range=list(ylim["panel3"]) if ylim["panel3"] else None,
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
            fig3.add_hline(y=rng[range_idx], line_dash="dash", line_color="red")
            fig3.data[0].colorbar.title = dict(
                text=f"{spec_var} ({specgram_unit})" if specgram_unit else spec_var,
                font=dict(size=18, color=dark_gray)
            )
            fig3.data[0].colorbar.tickfont = dict(size=16, color=dark_gray)
            if st.session_state.box_xlim_p3:
                fig3.update_xaxes(range=list(st.session_state.box_xlim_p3))
            if st.session_state.box_ylim_p3:
                fig3.update_yaxes(range=list(st.session_state.box_ylim_p3))
            ev3 = st.plotly_chart(fig3, width='stretch', config={"displaylogo": False}, on_select="rerun", key="panel3_select")
            if ev3 and ev3.selection and ev3.selection.get("box"):
                b = ev3.selection["box"][0]
                st.session_state.box_xlim_p3 = (min(b["x"]), max(b["x"]))
                st.session_state.box_ylim_p3 = (min(b["y"]), max(b["y"]))
                st.rerun()
            if st.session_state.box_xlim_p3 or st.session_state.box_ylim_p3:
                if st.button("Reset zoom", key="reset_p3"):
                    st.session_state.box_xlim_p3 = None
                    st.session_state.box_ylim_p3 = None
                    st.rerun()

    with col3:
        vel, spectrum = load_spectrum(files, ds_mom, prof_time_idx, range_idx, spec_var)
        if vel is not None and spectrum is not None:
            #print(spectrum)
            #print(spectrum.attrs)
            spectrum, spectrum_unit = convert_to_db_if_linear(spec_var, ds0, spectrum)
            mask = ~np.isnan(spectrum)
            spectrum = spectrum[mask]
            valid_vel = vel[mask]
            fig4 = go.Figure()
            fig4.add_trace(
                go.Scattergl(
                    x=valid_vel,
                    y=spectrum,
                    mode="lines",
                    name="Spectrum",
                    line=dict(color="#1f77b4", width=2),
                )
            )
            dark_gray = "#4d4d4d"
            grid_gray_major = "rgba(77,77,77,0.45)"
            fig4.update_layout(
                xaxis_title=f"Velocity ({units['velocity']})",
                yaxis_title=f"{spec_var} ({spectrum_unit})" if spectrum_unit else spec_var,
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

    # --- Row 3: Two columns for panels 5, 6 ---
    col5, col6 = st.columns(2)

    with col5:
        vel, rng, specgram = load_spectrogram(files, ds_mom, prof_time_idx, spec_var)
        if vel is not None and rng is not None and specgram is not None:
            specgram, _ = convert_to_db_if_linear(spec_var, ds0, specgram)
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
            vel_bins = radar_simulator._centers_to_edges(vel)
            if simulator_type == "Rain":
                lutPath = lut_path_rain
                sim_label = "Simulated (Rain)"
            else:
                lutPath = lut_path_snow
                sim_label = "Simulated (Snow)"
            vel_sim, sim_H, PSD, D, precip_rate = radar_simulator.simulate_spectrum(
                vel_bins=vel_bins,
                center_height=center_height,
                eps_diss=eps_diss,
                noise_pow=noise_pow,
                nave=nave,
                theta_deg=theta_deg,
                uwind=uwind,
                time_int=time_int,
                lut_path=lutPath,
                N0=N0,
                gamma=gamma,
                lam=lam,
                vertical_wind=vertical_wind,
                freq_ghz = freq_ghz,
            )
            print(precip_rate)
            fig5.add_trace(
                go.Scattergl(
                    x=vel_sim,
                    y=sim_H,
                    mode="lines",
                    name=sim_label,
                    line=dict(color="#d62728", width=2, dash="dash"),
                )
            )
            dark_gray = "#4d4d4d"
            grid_gray_major = "rgba(77,77,77,0.45)"
            fig5.update_layout(
                xaxis_title=f"Velocity ({units['velocity']})",
                yaxis_title=f"{spec_var} ({spectrum_unit})" if spectrum_unit else spec_var,
                height=450,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(size=20, color=dark_gray),
                legend=dict(
                    x=0.02,
                    y=0.98,
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
                tickformat=".0f",
                linecolor=dark_gray,
                linewidth=1.5,
                mirror=True,
            )
            if xlimits.get("panel5"):
                try:
                    xlo, xhi = [float(v) for v in xlimits["panel5"].split(",")]
                    fig5.update_xaxes(range=[xlo, xhi])
                except Exception:
                    pass
            if ylimits.get("panel5"):
                try:
                    ylo, yhi = [float(v) for v in ylimits["panel5"].split(",")]
                    fig5.update_yaxes(range=[ylo, yhi])
                except Exception:
                    pass
            st.plotly_chart(fig5, width="content", config={"displaylogo": False})

    with col6:
        fig6 = go.Figure()
        fig6.add_trace(
            go.Scatter(
                x=D,
                y=PSD,
                mode="lines",
                name=f"Simulated PSD ({simulator_type})",
                line=dict(color="#1f77b4", width=2),
            )
        )
        dark_gray = "#4d4d4d"
        grid_gray_major = "rgba(77,77,77,0.45)"
        fig6.update_layout(
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
        fig6.add_annotation(
            text=f"R = {precip_rate:.2e} mm/h",
            x=0.98,
            xref="paper",
            y=0.97,
            yref="paper",
            xanchor="right",
            yanchor="top",
            showarrow=False,
            font=dict(size=20, color="#4d4d4d"),
        )
        fig6.update_xaxes(
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
            exponentformat="e",
        )
        fig6.update_yaxes(
            type="log",
            showgrid=True,
            nticks=5,
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

    # --- Save all panels as one PNG ---

    save_path = st.sidebar.text_input("Save PNG as...", value="all_panels.png")
    save_all = st.sidebar.button("Save all panels as one PNG")
    fontsize=18
    if save_all:
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1,0.35])

        # Panel 1: Time x Range (spans all columns)
        ax1 = fig.add_subplot(gs[0, :])
        im1 = ax1.pcolormesh(time_values_disp, y_disp, z_disp, cmap=selected_cmap_panel1)
        ax1.set_title("Time x Range",fontsize=fontsize)
        ax1.set_ylabel("Range (m)",fontsize=fontsize)
        cbar = plt.colorbar(im1, ax=ax1, pad=0.01, aspect=30)
        cbar.set_label(f"{var} ({get_units_from_attrs(ds_mom[var])})", fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize-2)
        ax1.tick_params(labelsize=fontsize-2)
        ax1.axvline(time_values_disp[time_idx], color='r', linestyle='--')
        ax1.axhline(y_disp[range_idx], color='r', linestyle='--')
        ax1.grid()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        default_length = plt.rcParams['xtick.major.size']
        ax1.tick_params(axis='x', which='major', length=default_length, width=1.5)
        ax1.tick_params(axis='x', which='minor', length=default_length, width=1.5)

        # Panel 2: Profile
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(z_disp[:, time_idx], y_disp)
        ax2.set_title(f"Profile at {pd.to_datetime(time_values[time_idx]).strftime('%H:%M')}",fontsize=fontsize)
        ax2.set_xlabel(var,fontsize=fontsize)
        ax2.set_ylabel("Range (m)",fontsize=fontsize)
        ax2.grid()
        ax2.axhline(y_disp[range_idx], color='r', linestyle='--')
        ax2.tick_params(labelsize=fontsize-2)

        # Panel 3: Spectrogram
        if 'vel' in locals() and 'rng' in locals() and 'specgram' in locals() and vel is not None and rng is not None and specgram is not None:
            ax3 = fig.add_subplot(gs[1, 1])
            im3 = ax3.pcolormesh(vel, rng, specgram, cmap=selected_cmap_panel3, shading='nearest')
            ax3.set_title(f"Spectrogram at {pd.to_datetime(time_values[time_idx]).strftime('%H:%M')}",fontsize=fontsize)
            ax3.set_xlabel("Velocity (m/s)",fontsize=fontsize)
            ax3.set_ylabel("Range (m)",fontsize=fontsize)
            ax3.grid()
            cbar = plt.colorbar(im3, ax=ax3,pad = 0.01, aspect=30)
            cbar.set_label(f"{spec_var} ({get_units_from_attrs(ds_mom[var])})", fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize-2)
            ax3.axhline(rng[range_idx], color='r', linestyle='--')
            ax3.tick_params(labelsize=fontsize-2)
            ax3.set_xlim(vmin_x, vmax_x)

        # Panel 4: Spectrum
        if 'valid_vel' in locals() and 'spectrum' in locals() and valid_vel is not None and spectrum is not None:
            ax4 = fig.add_subplot(gs[1, 2])
            ax4.plot(valid_vel, spectrum)
            ax4.set_title(f"Spectrum at range {range_values[range_idx]:.1f} {units['range']}",fontsize=fontsize)
            ax4.set_xlabel("Velocity (m/s)",fontsize=fontsize)
            ax4.set_ylabel(var,fontsize=fontsize)
            ax4.tick_params(labelsize=fontsize-2)
            ax4.grid()

        # Panel 5: Measured & Simulated Spectrum
        if 'vel' in locals() and 'measured_spectrum' in locals() and 'vel_sim' in locals() and 'sim_H' in locals():
            ax5 = fig.add_subplot(gs[2, 0])
            ax5.plot(vel, measured_spectrum, label="Measured")
            ax5.plot(vel_sim, sim_H, label="Simulated", linestyle='--')
            ax5.set_title(f"Measured & Simulated Spectrum ({simulator_type})",fontsize=fontsize)
            ax5.set_xlabel("Velocity (m/s)",fontsize=fontsize)
            ax5.set_ylabel(var,fontsize=fontsize)
            ax5.legend()
            ax5.tick_params(labelsize=fontsize-2)
            ax5.grid()

        # Panel 6: Simulated PSD
        if 'D' in locals() and 'PSD' in locals():
            ax6 = fig.add_subplot(gs[2, 1])
            ax6.plot(D, PSD)
            ax6.set_title("Simulated PSD",fontsize=fontsize)
            ax6.set_xlabel("Diameter D [m]",fontsize=fontsize)
            ax6.set_ylabel("PSD",fontsize=fontsize)
            ax6.set_xscale("log")
            ax6.set_yscale("log")
            ax6.tick_params(labelsize=fontsize-2)
            ax6.grid()

        # Hide unused axes (bottom right)
        ax_unused = fig.add_subplot(gs[2, 2])
        ax_unused.axis('off')
        ax_unused = fig.add_subplot(gs[1, 3])
        ax_unused.axis('off')
        ax_unused = fig.add_subplot(gs[2, 3])
        ax_unused.axis('off')


        plt.tight_layout()
        fig.savefig(save_path, dpi=200,bbox_inches='tight')
        plt.close(fig)
        st.success(f"All panels saved as {os.path.abspath(save_path)}")

    st.markdown("""
    ---
    **Tips:**
    - Use the sliders to select time/range for all panels.
    - Panel 1 shows the time-range variable (e.g. reflectivity) with the selected point highlighted.
    - Panel 2 shows the vertical profile at the selected time.
    - Panel 3 shows the spectrogram at the selected time, with the selected range highlighted.
    - Panel 4 shows the spectrum at the selected time and range.
    - Panel 5 compares the measured spectrum with the simulated spectrum based on the selected parameters. Use the controls in the "Simulation Parameters" sidebar to adjust the gamma DSD parameters and see how the simulated spectrum changes.
    - Panel 6 shows the simulated PSD based on the gamma DSD parameters.
                
    - The Display Options allow you to customize the colormaps and colorbar limits for panels 1 and 3.
    - Panel 1 allows you to select a specific time-range region by dragging a box. Click "Reset zoom" to clear the selection. Alternatively you can set the axis limits manually using the text inputs in the sidebar Display Options.
    - Panel 2 and 3 also allow box selection to zoom in on specific velocity or range intervals. Use the "Reset zoom" buttons to clear those selections.
    
    """)

if __name__ == "__main__":
    main()
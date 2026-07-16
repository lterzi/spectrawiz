from __future__ import annotations

"""
mrr.py

Self-contained MRR-PRO (Metek) backend for spectrawiz.

This backend re-implements, in one place and without depending on the external ERUO package,
the same three stages ERUO normally runs as separate scripts:
  1) preprocessing  - estimate the interference mask, border correction and a smoothed median
                       noise floor from the dataset itself;
  2) processing      - per-spectrum de-aliasing, noise removal and moment computation;
  3) postprocessing  - removal of remaining interference lines / isolated artifacts from the
                       processed (time, range) moments.

Unlike the original ERUO scripts, all three stages run in-memory from a single input
xr.Dataset (one raw MRR-PRO file, or one hour merged across files, exactly like the other
backends receive), and the preprocessing products are estimated from that same dataset's own
time extent rather than from a separate, campaign-wide pass.

The de-aliasing logic includes the fixes developed against this exact instrument's data:
choosing the Nyquist-fold duplicate closest to a reference velocity (preferring the
manufacturer's own VEL field when present, since it is computed independently per time step),
picking the "main" line by peak power rather than range span, requiring a substantial range
overlap before treating two lines as duplicates of each other, and rescuing real signal flagged
for reconstruction via nearby time steps (with the rescued peak's full width restored) when no
valid range neighbor is available within the same spectrum.
"""

import copy
import warnings
from typing import Any

import numpy as np
import scipy.ndimage
import scipy.signal
import astropy.convolution
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans, convolve
import xarray as xr

from .base import RadarBackend
from ..processing_common import add_standard_variable_attrs, as_vel_ref, finalize_metadata

# ----------------------------------------------------------------------------------------------
# Constants (instrument-fixed, and the processing thresholds validated against this MRR-PRO).
# ----------------------------------------------------------------------------------------------

# Fixed physical constants (MRR-PRO).
F_S = 500000.0          # Sampling rate [Hz]
LAM = 0.01238           # Wavelength [m]
K2 = 0.92                # |K^2| dielectric factor of water
CONST_Z_CALC = ((10.0 ** 18) * (LAM ** 4) * K2) / (np.pi ** 5)
CALIB_CONST_FACTOR = 1.0e20

# Preprocessing (interference mask / border correction estimation).
QUANTILE_PREPROCESSING = 0.5
MAX_GRADIENT_MULTIPLIER_INTER_FIT = 3.0
CHOSEN_DEGREE_FIT_INTER_FIT = 4
MIN_LEN_SLICES_INTERF_FIT = 3
PROMINENCE_INTERFERENCE_REMOVAL_RAW_SPECTRUM = 0.2
MAX_FRACTION_OF_NAN_AT_RANGE = 0.9
NUM_ITERATIONS_INTERF_MASK_DILATION = 3
MARGIN_L_BORD_CORR = 3
MARGIN_R_BORD_CORR = 3

# Spectrum reconstruction (interference replacement).
RECONSTRUCT_SPECTRUM = True
MARGIN_SMALL_INTERF_DETECTION = 2
MAX_NUM_PEAKS_IN_MARGIN_SMALL_INTERF = 5
FRACTION_VEL_LINE_INTERFERENCE = 0.8
ADIACIENTIA_WIDTH = 5
EXCEPTIONAL_ANOMALY_THRESHOLD = 5.0
HORIZONTAL_TOL = 5
MIN_WIN_RECONSTRUCTION = 8
KERNEL_SCALE_FACTOR = 3.0
NUM_BOTTOM_GATES_TO_SKIP_IN_RECONSTRUCTION = 15
MIN_PROMINENCE_THRESHOLD_RECONSTRUCTED = 1.0
RESCUE_VIA_TIME_NEIGHBORS = True
MAX_TIME_WINDOW_RESCUE = 5

# Spectrum processing (peak/line finding, de-aliasing, noise removal, moments).
PROMINENCE_THRESHOLD = 0.2
RELATIVE_PROMINENCE_THRESHOLD = 0.25
MAX_NUM_PEAKS_AT_R = 6
WINDOW_R = 5.0
WINDOW_V = 10.0
MIN_NUM_PEAKS_IN_LINE = 3
VEL_TOL = 1.0
DA_THRESHOLD = 1.0e-3
OVERLAP_FRACTION_THRESHOLD_DUPLICATE = 0.5
NOISE_STD_FACTOR = 3.0
CORRECT_NOISE_LVL = True
NOISE_CORR_WINDOW = 5.0
MAX_DIFF_NOISE_LVL = 0.2
REMOVE_ISOLATED_PEAK_SPECTRUM = True
DEALIAS_DEFAULT_REF_FRAC = 0.5

# Postprocessing (cleanup of the processed (time, range) moments).
MIN_SNR_POSTPROC = -20.0
REMOVE_INTERF_POSTPROC = True
MIN_TIME_FRACTION_INTERF_POSTPROC = 0.2
WINDOW_POSTPROCESS_T = 40
WINDOW_POSTPROCESS_R = 40
MIN_HALF_FRACTION = 0.2
MIN_RATIO_H_V = 2.0
MIN_INTERF_FLAG = 20
REMOVE_NOISE_POSTPROC = True
MIN_SLICE_LENGTH_NOISE_REMOVAL = 3
MIN_NUM_PIXEL_NOISE_REMOVAL = 4

# Transfer function reconstruction.
RECONSTRUCT_TRANSFER_FUNCTION = True
TRANSFER_FUNCTION_MAX_VALUE = 9.0e9


# ----------------------------------------------------------------------------------------------
# Small shared helpers
# ----------------------------------------------------------------------------------------------

def _slice_at_nan(a: np.ndarray) -> list[list]:
    """Split a 1D array into [slice, values] pairs of contiguous non-NaN runs."""
    return [[s, a[s]] for s in np.ma.clump_unmasked(np.ma.masked_invalid(a))]


def _contiguous_flagged_run(flagged_1d: np.ndarray, pos: int) -> tuple[int, int]:
    """
    Index range (inclusive) of the contiguous run of True values in "flagged_1d" containing
    "pos". Used so that rescuing a peak restores its full natural width, not just its single
    strongest bin (a real Doppler peak is several bins wide).
    """
    left = pos
    while left > 0 and flagged_1d[left - 1]:
        left -= 1
    right = pos
    while right < flagged_1d.shape[0] - 1 and flagged_1d[right + 1]:
        right += 1
    return left, right


# ----------------------------------------------------------------------------------------------
# 1) PREPROCESSING: estimate interference mask, border correction, smoothed median noise floor.
# ----------------------------------------------------------------------------------------------

def _fit_reconstructed_median(spectra_q: np.ndarray, min_len_slice: int,
                              max_gradient_multiplier: float, chosen_degree_fit: int):
    """
    Fits a smooth ("interference-free") version of the per-range median of "spectra_q".

    Returns (reconstructed_median_line, min_r_interf), or (None, None) if there are not enough
    acceptable range gates to fit (e.g. far too short a dataset).
    """
    median_line = np.nanmedian(spectra_q, axis=1)
    median_line_grad = np.gradient(median_line)

    neg_grad = median_line_grad[median_line_grad < 0.0]
    if not neg_grad.size:
        return None, None
    min_r_interf = np.nanmin(np.where(median_line_grad < np.nanmedian(neg_grad)))

    grad_thresh = -max_gradient_multiplier * np.abs(np.nanmedian(median_line_grad[min_r_interf:]))
    condition_above_peak = np.logical_and(
        np.logical_and(median_line_grad > grad_thresh, median_line_grad < 0.0),
        np.arange(median_line_grad.shape[0]) > min_r_interf,
    )
    if not np.sum(condition_above_peak):
        return None, None

    accepted_r_raw = np.full(median_line_grad.shape, np.nan)
    accepted_r_raw[condition_above_peak] = median_line[condition_above_peak]
    accepted_r = np.full(median_line_grad.shape, np.nan)
    for sl, sl_values in _slice_at_nan(accepted_r_raw):
        if sl_values.shape[0] > min_len_slice:
            accepted_r[sl] = sl_values

    x = np.arange(median_line.shape[0])[np.isfinite(accepted_r)]
    y = accepted_r[np.isfinite(accepted_r)]
    if x.size <= chosen_degree_fit:
        return None, None
    x_fit = np.arange(median_line.shape[0])[min_r_interf:]

    r_fit_params = np.polyfit(x, y, chosen_degree_fit, full=False)
    r_fit_function = np.poly1d(r_fit_params)
    fitted_r = np.full(median_line.shape, np.nan)
    fitted_r[min_r_interf:] = r_fit_function(x_fit)

    reconstructed_median_line = np.full(median_line.shape, np.nan)
    reconstructed_median_line[:min_r_interf] = median_line[:min_r_interf]
    reconstructed_median_line[min_r_interf:] = np.min(
        np.stack([fitted_r, median_line]), axis=0
    )[min_r_interf:]

    return reconstructed_median_line, r_fit_function


def estimate_preprocessing_products(
    spectra_q: np.ndarray,
    *,
    max_gradient_multiplier: float = MAX_GRADIENT_MULTIPLIER_INTER_FIT,
    chosen_degree_fit: int = CHOSEN_DEGREE_FIT_INTER_FIT,
    min_len_slice: int = MIN_LEN_SLICES_INTERF_FIT,
    threshold_prominence_interference: float = PROMINENCE_INTERFERENCE_REMOVAL_RAW_SPECTRUM,
    max_fraction_of_nan_at_range: float = MAX_FRACTION_OF_NAN_AT_RANGE,
    num_iterations_interf_mask_dilation: int = NUM_ITERATIONS_INTERF_MASK_DILATION,
    margin_l_bord_corr: int = MARGIN_L_BORD_CORR,
    margin_r_bord_corr: int = MARGIN_R_BORD_CORR,
):
    """
    Estimates the interference mask, border correction and smoothed median noise floor from a
    single (range, velocity) quantile of the spectrum (e.g. the median across the time steps in
    one input dataset).

    Returns (interference_mask, border_corr, reconstructed_median_line) or, if there is not
    enough information to fit a reliable median (e.g. an unusually short dataset), a tuple of
    all-False/all-zero/raw-median fallbacks so that processing can still proceed.
    """
    m = spectra_q.shape[1]

    reconstructed_median_line, _ = _fit_reconstructed_median(
        spectra_q, min_len_slice, max_gradient_multiplier, chosen_degree_fit
    )
    if reconstructed_median_line is None:
        # Not enough range gates/contrast to fit: fall back to a flat, unmasked baseline rather
        # than failing outright.
        flat_median = np.nanmedian(spectra_q, axis=1)
        return (
            np.zeros(spectra_q.shape, dtype=bool),
            np.zeros(spectra_q.shape, dtype=float),
            flat_median,
        )

    diff_from_median = spectra_q - np.tile(reconstructed_median_line, (m, 1)).T

    # Preliminary mask, used only to exclude isolated bumps from the border-correction estimate.
    interf_mask_nonfull = np.zeros(diff_from_median.shape, dtype=bool)
    interf_mask_nonfull[diff_from_median > threshold_prominence_interference] = True
    interf_mask_nonfull[
        np.sum(interf_mask_nonfull, axis=1) > m - margin_l_bord_corr - margin_r_bord_corr, :
    ] = False
    interf_mask_nonfull[0:margin_l_bord_corr, :] = False
    interf_mask_nonfull[-margin_r_bord_corr:, :] = False
    interf_mask_nonfull = scipy.ndimage.binary_erosion(interf_mask_nonfull).astype(bool)
    interf_mask_nonfull = scipy.ndimage.binary_dilation(interf_mask_nonfull).astype(bool)

    # Border correction: difference between the (isolated-bump-masked) median and the quantile.
    quantile_for_border_corr = copy.deepcopy(spectra_q)
    quantile_for_border_corr[interf_mask_nonfull] = np.nan
    median_line_for_border_corr = np.nanmedian(quantile_for_border_corr, axis=1)
    border_corr = np.tile(median_line_for_border_corr, (m, 1)).T - quantile_for_border_corr
    border_corr[np.isnan(border_corr)] = 0.0
    border_corr[border_corr < 0.0] = 0.0
    num_significant_corrections = np.sum(border_corr, axis=1) > 2 * threshold_prominence_interference
    border_corr[num_significant_corrections > 6] = 0.0

    # Re-fit the median including the border correction, so the final interference mask also
    # accounts for the border-corrected values at the Doppler extremes.
    corrected_spectra_q = spectra_q + border_corr
    corrected_spectra_q_masked = corrected_spectra_q.copy()
    corrected_spectra_q_masked[interf_mask_nonfull] = np.nan

    corrected_reconstructed_median_line, corrected_fit_function = _fit_reconstructed_median(
        corrected_spectra_q_masked, min_len_slice, max_gradient_multiplier, chosen_degree_fit
    )
    if corrected_reconstructed_median_line is None:
        corrected_reconstructed_median_line = reconstructed_median_line

    last_spectra_q = spectra_q + border_corr
    corrected_diff_from_median = last_spectra_q - np.tile(
        corrected_reconstructed_median_line, (m, 1)
    ).T

    interf_mask = np.zeros(corrected_diff_from_median.shape, dtype=bool)
    interf_mask[corrected_diff_from_median > threshold_prominence_interference] = True
    interf_mask[np.sum(interf_mask, axis=1) > max_fraction_of_nan_at_range * m, :] = True
    interference_mask = scipy.ndimage.binary_dilation(
        interf_mask, iterations=num_iterations_interf_mask_dilation
    ).astype(bool)
    interference_mask[num_significant_corrections > 6] = True

    return interference_mask, border_corr, corrected_reconstructed_median_line


# ----------------------------------------------------------------------------------------------
# 2a) RECONSTRUCTION: replace interference-flagged regions with a modeled value.
# ----------------------------------------------------------------------------------------------

def define_reficiendo(spectrum_3d: np.ndarray, median_line_tiled: np.ndarray,
                      interference_mask_2d: np.ndarray):
    """Identifies the region of the spectrum, at each time step, to be reconstructed."""
    num_t = spectrum_3d.shape[0]
    r_idx = np.arange(spectrum_3d.shape[1])

    interference_mask_3d = np.tile(interference_mask_2d, (num_t, 1, 1))
    anomaly_3d = np.array(spectrum_3d - median_line_tiled)

    reficiendo_3d_raw = np.logical_and(
        interference_mask_3d, anomaly_3d > MIN_PROMINENCE_THRESHOLD_RECONSTRUCTED
    )
    reficiendo_3d_v2 = np.zeros(reficiendo_3d_raw.shape, dtype=bool)

    for i_t in range(reficiendo_3d_v2.shape[0]):
        label, num_features = scipy.ndimage.label(reficiendo_3d_raw[i_t, :, :])
        if not num_features:
            continue
        for i_feat in range(1, num_features + 1):
            curr_masked = label == i_feat
            curr_adiacentia = np.logical_xor(
                curr_masked,
                scipy.ndimage.binary_dilation(curr_masked, iterations=MARGIN_SMALL_INTERF_DETECTION),
            )
            curr_adiacentia[np.logical_xor(curr_masked, reficiendo_3d_raw[i_t, :, :])] = False
            if (
                np.sum(anomaly_3d[i_t, :, :][curr_adiacentia] > MIN_PROMINENCE_THRESHOLD_RECONSTRUCTED)
                < MAX_NUM_PEAKS_IN_MARGIN_SMALL_INTERF
            ):
                reficiendo_3d_v2[i_t, :, :] += curr_masked
            else:
                num_masked_in_range = np.sum(curr_masked, axis=1)
                affected_by_line_interf = r_idx[
                    num_masked_in_range > FRACTION_VEL_LINE_INTERFERENCE * spectrum_3d.shape[2]
                ]
                reficiendo_3d_v2[i_t, affected_by_line_interf, :] += curr_masked[affected_by_line_interf]

    reficiendo_3d = reficiendo_3d_v2 > 0
    gates_reficiendi = np.sum(reficiendo_3d, axis=2) > 0

    # Rescue: keep a flagged peak if nearby, unflagged range gates show a consistent position
    # (real, continuous precipitation looks similar at adjacent ranges).
    for i_t in range(reficiendo_3d.shape[0]):
        for i_r in r_idx[gates_reficiendi[i_t, :]]:
            curr_anom_max = np.nanmax(anomaly_3d[i_t, i_r, :])
            if curr_anom_max <= EXCEPTIONAL_ANOMALY_THRESHOLD:
                continue

            valid_below = np.where(np.logical_and(~gates_reficiendi[i_t, :], r_idx < i_r))[0]
            valid_above = np.where(np.logical_and(~gates_reficiendi[i_t, :], r_idx > i_r))[0]

            closest_valid_below = valid_below[0 - min(valid_below.shape[0], ADIACIENTIA_WIDTH):]
            closest_valid_above = valid_above[0:min(valid_above.shape[0], ADIACIENTIA_WIDTH)]

            if closest_valid_below.shape[0] > 3:
                closest_valid_below = closest_valid_below[:-1][np.diff(closest_valid_below) < 2]
                valid_anom_below = anomaly_3d[i_t, closest_valid_below, :]
                max_anom_below = np.nanmedian(np.nanmax(valid_anom_below, axis=1))
                max_anom_pos_below = np.nanmedian(np.nanargmax(valid_anom_below, axis=1))
            else:
                max_anom_below = 0.0
                max_anom_pos_below = -999

            if closest_valid_above.shape[0] > 3:
                closest_valid_above = closest_valid_above[1:][np.diff(closest_valid_above) < 2]
                valid_anom_above = anomaly_3d[i_t, closest_valid_above, :]
                max_anom_above = np.nanmedian(np.nanmax(valid_anom_above, axis=1))
                max_anom_pos_above = np.nanmedian(np.nanargmax(valid_anom_above, axis=1))
            else:
                max_anom_above = 0.0
                max_anom_pos_above = -999

            if max_anom_below > EXCEPTIONAL_ANOMALY_THRESHOLD or max_anom_above > EXCEPTIONAL_ANOMALY_THRESHOLD:
                curr_anom_max_pos = np.nanargmax(anomaly_3d[i_t, i_r, :])
                if (
                    np.abs(curr_anom_max_pos - max_anom_pos_below) < HORIZONTAL_TOL
                    or np.abs(curr_anom_max_pos - max_anom_pos_above) < HORIZONTAL_TOL
                ):
                    reficiendo_3d[i_t, i_r, curr_anom_max_pos] = False

    return anomaly_3d, reficiendo_3d


def rescue_via_time_neighbors(
    anomaly_3d: np.ndarray, reficiendo_3d: np.ndarray,
    max_time_window: int = MAX_TIME_WINDOW_RESCUE, horizontal_tol: int = HORIZONTAL_TOL,
    exceptional_anomaly_threshold: float = EXCEPTIONAL_ANOMALY_THRESHOLD,
) -> np.ndarray:
    """
    Rescues real signal flagged for reconstruction when no valid range neighbor is available to
    compare against (e.g. the whole profile flagged at once), by comparing the same range gate
    at nearby time steps instead. Restores the full contiguous flagged run around the rescued
    peak, not just its single strongest bin, so the real peak's natural width is preserved.
    """
    num_t, N, _m = anomaly_3d.shape
    reficiendo_3d_rescued = reficiendo_3d.copy()
    r_idx = np.arange(N)
    gates_reficiendi = np.sum(reficiendo_3d, axis=2) > 0

    for i_t in range(num_t):
        for i_r in r_idx[gates_reficiendi[i_t, :]]:
            curr_anom_max = np.nanmax(anomaly_3d[i_t, i_r, :])
            if curr_anom_max <= exceptional_anomaly_threshold:
                continue
            curr_anom_max_pos = np.nanargmax(anomaly_3d[i_t, i_r, :])

            for dt in range(1, max_time_window + 1):
                rescued = False
                for t_other in (i_t - dt, i_t + dt):
                    if t_other < 0 or t_other >= num_t or gates_reficiendi[t_other, i_r]:
                        continue
                    other_anom = anomaly_3d[t_other, i_r, :]
                    if np.nanmax(other_anom) <= exceptional_anomaly_threshold:
                        continue
                    other_pos = np.nanargmax(other_anom)
                    if np.abs(curr_anom_max_pos - other_pos) < horizontal_tol:
                        left, right = _contiguous_flagged_run(
                            reficiendo_3d_rescued[i_t, i_r, :], curr_anom_max_pos
                        )
                        reficiendo_3d_rescued[i_t, i_r, left:right + 1] = False
                        rescued = True
                        break
                if rescued:
                    break

    return reficiendo_3d_rescued


def reconstruct_anomaly(anomaly: np.ndarray, reficiendo: np.ndarray) -> np.ndarray:
    """Reconstructs the flagged region of a single spectrum's anomaly via 2D interpolation."""
    has_at_least_one_masked = ~np.any(reficiendo, axis=1)
    masked_tmp = np.ma.masked_array(has_at_least_one_masked, mask=has_at_least_one_masked)
    sections_masked = np.ma.clump_unmasked(masked_tmp)
    longest_masked_section = MIN_WIN_RECONSTRUCTION
    for sl in sections_masked:
        if masked_tmp[sl].shape[0] > longest_masked_section:
            longest_masked_section = masked_tmp[sl].shape[0]

    y_std = int(np.ceil(longest_masked_section / KERNEL_SCALE_FACTOR))

    valid_spectrum_extend = np.logical_or(
        np.sum(np.isfinite(anomaly), axis=1) > 0, np.sum(reficiendo, axis=1) > 0
    )
    valid_spectrum_extend[0:NUM_BOTTOM_GATES_TO_SKIP_IN_RECONSTRUCTION] = False

    valid_anomaly = anomaly[valid_spectrum_extend, :]
    valid_anomaly[reficiendo[valid_spectrum_extend, :]] = np.nan

    vertical_dim_reconstr = np.sum(valid_spectrum_extend) + (2 * longest_masked_section)
    img_for_reconstruction = np.zeros((vertical_dim_reconstr, anomaly.shape[1]))
    img_for_reconstruction[longest_masked_section:-longest_masked_section, :] = valid_anomaly

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        to_fill_bot = np.nanmean(np.stack([valid_anomaly[0, :], valid_anomaly[1, :]], axis=1), axis=1)
        to_fill_top = np.nanmean(np.stack([valid_anomaly[-1, :], valid_anomaly[-2, :]], axis=1), axis=1)
    for i in range(longest_masked_section):
        img_for_reconstruction[i, :] = to_fill_bot
        img_for_reconstruction[-i, :] = to_fill_top

    kernel = Gaussian2DKernel(x_stddev=1, y_stddev=y_std)
    fixed_img = interpolate_replace_nans(img_for_reconstruction, kernel, convolve=convolve, boundary="wrap")

    fixed_anomaly = copy.deepcopy(anomaly)
    fixed_anomaly[valid_spectrum_extend, :] = fixed_img[longest_masked_section:-longest_masked_section, :]
    return fixed_anomaly


# ----------------------------------------------------------------------------------------------
# 2b) PROCESSING: per-spectrum peak/line finding, de-aliasing, noise removal, moments.
# ----------------------------------------------------------------------------------------------

def reconstruct_transfer_function(
    transfer_function: np.ndarray,
    max_value: float = TRANSFER_FUNCTION_MAX_VALUE,
) -> np.ndarray:
    """
    Stretches the transfer function to cover range gates where it exceeds max_value.

    Some MRR-PRO files have an abrupt cutoff in the transfer function at upper range gates
    (values >> 1, normally the transfer function is in [0, 1]).  ERUO's fix: resample the
    acceptable portion to cover the full range and rescale so the peak is unchanged.
    If no values exceed max_value the transfer function is returned as-is.
    """
    if not np.any(transfer_function > max_value):
        return transfer_function
    cond_acceptable = transfer_function < max_value
    new_tf = scipy.signal.resample(transfer_function[cond_acceptable], transfer_function.shape[0])
    new_tf *= np.nanmax(transfer_function[cond_acceptable]) / np.nanmax(new_tf)
    return new_tf


def compute_additional_mrr_parameters(N: int, m: int, T_i: float, d_r: float) -> dict[str, Any]:
    f_ny = F_S / (2.0 * N)
    v_ny = (LAM * F_S) / (4.0 * N)
    d_v = (LAM * F_S) / (4.0 * N * m)
    v_0 = np.arange(0.0, d_v * m, d_v)
    return {"f_ny": f_ny, "v_ny": v_ny, "d_v": d_v, "v_0": v_0}


def repeat_spectra(all_spectra: np.ndarray, transfer_function: np.ndarray):
    """Tiles the spectrum x3 along velocity (for de-aliasing) and converts dB -> linear."""
    spectrum_before = np.full(all_spectra.shape, np.nan)
    spectrum_after = np.full(all_spectra.shape, np.nan)
    spectrum_before[:, :-1, :] = all_spectra[:, 1:, :]
    spectrum_after[:, 1:, :] = all_spectra[:, :-1, :]

    tiled_spectra = np.concatenate([spectrum_before, all_spectra, spectrum_after], axis=2)
    all_spectra_x3_lin = np.power(10.0, tiled_spectra / 10.0)

    m_x3 = all_spectra_x3_lin.shape[2]
    transfer_function_x3 = np.tile(transfer_function, (m_x3, 1)).T
    return all_spectra_x3_lin, transfer_function_x3


def find_raw_peaks(spec: np.ndarray, N: int, m: int, max_num_peaks_at_r: int = MAX_NUM_PEAKS_AT_R):
    r_idx_peaks_list, v_idx_peaks_list = [], []
    v_l_idx_peaks_list, v_r_idx_peaks_list = [], []

    for i_r in range(N):
        peaks, properties = scipy.signal.find_peaks(spec[i_r, :], prominence=PROMINENCE_THRESHOLD, height=0.0)
        if not len(peaks):
            continue
        if len(peaks) > max_num_peaks_at_r:
            peak_order = np.argsort(properties["peak_heights"])
            peaks = peaks[peak_order][-max_num_peaks_at_r:]
            for k in properties.keys():
                properties[k] = properties[k][peak_order][-max_num_peaks_at_r:]

        accepted = properties["prominences"] > RELATIVE_PROMINENCE_THRESHOLD * np.max(properties["prominences"])
        r_idx_peaks_list.append(np.ones(np.sum(accepted)) * i_r)
        v_idx_peaks_list.append(peaks[accepted])
        v_l_idx_peaks_list.append(properties["left_bases"][accepted])
        v_r_idx_peaks_list.append(properties["right_bases"][accepted])

    if not len(r_idx_peaks_list):
        return [], [], [], [], []

    r_idx_peaks = np.concatenate(r_idx_peaks_list).astype(int)
    v_idx_peaks = np.concatenate(v_idx_peaks_list).astype(int)
    v_l_idx_peaks = np.concatenate(v_l_idx_peaks_list).astype(int)
    v_r_idx_peaks = np.concatenate(v_r_idx_peaks_list).astype(int)
    idx_peaks = np.arange(r_idx_peaks.shape[0], dtype=int)
    return r_idx_peaks, v_idx_peaks, v_l_idx_peaks, v_r_idx_peaks, idx_peaks


def find_raw_lines(spec, v_0_3, r, r_idx_peaks, v_idx_peaks, idx_peaks):
    lines = [[]]
    for i_peak in idx_peaks:
        curr_r = r_idx_peaks[i_peak]
        curr_v = v_idx_peaks[i_peak]
        elegible = np.logical_and(
            np.logical_and(np.abs(curr_r - r_idx_peaks) < WINDOW_R, np.abs(curr_v - v_idx_peaks) < WINDOW_V),
            idx_peaks > i_peak,
        )
        if not np.sum(elegible):
            continue
        elegible_r = r_idx_peaks[elegible]
        elegible_v = v_idx_peaks[elegible]
        elegible_idx = idx_peaks[elegible]
        distance2 = (1 + WINDOW_V ** 2) * np.square(curr_r - elegible_r) + np.square(curr_v - elegible_v)
        closest_idx = elegible_idx[np.argmin(distance2)]

        for l in lines:
            if i_peak in l:
                l.append(closest_idx)
                break
        else:
            lines.append([i_peak, closest_idx])

    line_v_idx, line_r_idx = [], []
    line_v, line_r, line_pow_lin = [], [], []
    line_min_r, line_max_r = [], []
    line_median_v, line_median_pow_lin = [], []
    lines_array = []

    for l in lines:
        if len(l) < MIN_NUM_PEAKS_IN_LINE:
            continue
        l_array = np.array(l, dtype=int)
        lines_array.append(l_array)
        line_v_idx.append(v_idx_peaks[l_array])
        line_r_idx.append(r_idx_peaks[l_array])
        line_v.append(v_0_3[v_idx_peaks[l_array]])
        line_r.append(r[r_idx_peaks[l_array]])
        line_pow_lin.append(spec[r_idx_peaks[l_array], v_idx_peaks[l_array]])
        line_min_r.append(np.nanmin(line_r[-1]))
        line_max_r.append(np.nanmax(line_r[-1]))
        idx_half_line_v = int(np.floor(len(line_v[-1]) / 2.0))
        line_median_v.append(np.nanmedian(line_v[-1][idx_half_line_v:]))
        line_median_pow_lin.append(np.nanmedian(line_pow_lin[-1]))

    return (lines_array, line_v_idx, line_r_idx, line_v, line_r, line_pow_lin, line_min_r,
            line_max_r, line_median_v, line_median_pow_lin)


def exclude_duplicate_lines(v_ny, lines_array, line_v_idx, line_r_idx, line_v, line_r,
                            line_pow_lin, line_min_r, line_max_r, line_median_v, prev_v_ref=0.0):
    """
    Removes peaks repeated at approximately v_ny, choosing the one in the line closest to
    "prev_v_ref". A conflicting candidate only counts towards that choice if it covers at least
    OVERLAP_FRACTION_THRESHOLD_DUPLICATE of the line being resolved: a small fragment that only
    coincidentally sits at a v_ny offset should not be able to "win" over the true (much larger)
    duplicate just by chance being closer to the reference.
    """
    array_min_r = np.array(line_min_r)
    array_max_r = np.array(line_max_r)
    array_median_v = np.array(line_median_v)

    matrix_min_r_1, matrix_min_r_2 = np.meshgrid(array_min_r, array_min_r)
    matrix_max_r_1, matrix_max_r_2 = np.meshgrid(array_max_r, array_max_r)
    matrix_median_v_1, _ = np.meshgrid(array_median_v, array_median_v)

    cond_r = np.logical_and(matrix_min_r_1 <= matrix_max_r_2, matrix_min_r_2 <= matrix_max_r_1)

    cond_v = np.zeros(cond_r.shape, dtype=bool)
    overlap_size = np.zeros(cond_r.shape, dtype=int)
    for i in range(len(lines_array) - 1):
        for j in range(i + 1, len(lines_array)):
            lines_intersect, comm1, comm2 = np.intersect1d(
                line_r_idx[i], line_r_idx[j], return_indices=True, assume_unique=False
            )
            if not len(lines_intersect):
                continue
            diff = np.abs(np.nanmedian(line_v[i][comm1] - line_v[j][comm2]))
            if (
                np.isclose(diff, v_ny, atol=VEL_TOL)
                or np.isclose(diff, 2.0 * v_ny, atol=VEL_TOL)
                or np.isclose(diff, 3.0 * v_ny, atol=VEL_TOL)
            ):
                cond_v[i, j] = cond_v[j, i] = True
                overlap_size[i, j] = overlap_size[j, i] = len(lines_intersect)

    cond = np.logical_not(np.logical_and(cond_r, cond_v))
    cond[
        np.logical_and(
            np.tile(np.any(np.logical_not(cond), axis=0), (cond.shape[0], 1)),
            np.identity(cond.shape[0], dtype=bool),
        )
    ] = False

    line_lengths = np.array([len(arr) for arr in line_r_idx])
    np.fill_diagonal(overlap_size, line_lengths)
    overlap_fraction = overlap_size / line_lengths[:, None]
    cond_for_resolution = np.logical_or(cond, overlap_fraction < OVERLAP_FRACTION_THRESHOLD_DUPLICATE)

    v_investigated = np.ma.masked_array(np.abs(matrix_median_v_1 - prev_v_ref), mask=cond_for_resolution)

    no_conflict = np.all(cond, axis=1)
    idx_no_conflict = np.arange(no_conflict.shape[0])[no_conflict]

    idx_conflict_all = np.argmin(v_investigated, axis=1)[np.logical_not(no_conflict)]
    y_conflict_all = np.arange(no_conflict.shape[0])[np.logical_not(no_conflict)]
    idx_conflict = np.unique(np.intersect1d(idx_conflict_all, y_conflict_all))

    accepted_lines, accepted_lines_v_idx, accepted_lines_r_idx = [], [], []
    accepted_lines_v, accepted_lines_r = [], []
    accepted_lines_min_r, accepted_lines_max_r = [], []
    accepted_lines_v_med, accepted_lines_pow_lin_max = [], []

    def _append_whole(curr_idx):
        accepted_lines.append(lines_array[curr_idx])
        accepted_lines_v_idx.append(line_v_idx[curr_idx])
        accepted_lines_r_idx.append(line_r_idx[curr_idx])
        accepted_lines_v.append(line_v[curr_idx])
        accepted_lines_r.append(line_r[curr_idx])
        accepted_lines_min_r.append(line_min_r[curr_idx])
        accepted_lines_max_r.append(line_max_r[curr_idx])
        idx_half = int(np.floor(len(line_v[curr_idx]) / 2.0))
        accepted_lines_v_med.append(np.nanmedian(line_v[curr_idx][idx_half:]))
        accepted_lines_pow_lin_max.append(np.nanmax(line_pow_lin[curr_idx]))

    for curr_idx in idx_no_conflict:
        _append_whole(curr_idx)

    for curr_idx in idx_conflict:
        curr_best_line_idx = np.argmin(v_investigated[curr_idx, :])
        if curr_best_line_idx == curr_idx:
            _append_whole(curr_idx)
            continue

        r_idx_to_keep = np.setdiff1d(line_r_idx[curr_idx], line_r_idx[curr_best_line_idx], assume_unique=True)
        if not len(r_idx_to_keep):
            continue
        mask_valid = np.isin(line_r_idx[curr_idx], r_idx_to_keep)
        accepted_lines.append(lines_array[curr_idx][mask_valid])
        accepted_lines_v_idx.append(line_v_idx[curr_idx][mask_valid])
        accepted_lines_r_idx.append(line_r_idx[curr_idx][mask_valid])
        accepted_lines_v.append(line_v[curr_idx][mask_valid])
        accepted_lines_r.append(line_r[curr_idx][mask_valid])
        accepted_lines_min_r.append(np.min(line_r[curr_idx][mask_valid]))
        accepted_lines_max_r.append(np.max(line_r[curr_idx][mask_valid]))
        idx_half = int(np.floor(len(line_v[curr_idx][mask_valid]) / 2.0))
        accepted_lines_v_med.append(np.nanmedian(line_v[curr_idx][mask_valid][idx_half:]))
        accepted_lines_pow_lin_max.append(np.nanmax(line_pow_lin[curr_idx][mask_valid]))

    return (accepted_lines, accepted_lines_v_idx, accepted_lines_r_idx, accepted_lines_v,
            accepted_lines_r, accepted_lines_min_r, accepted_lines_max_r,
            np.array(accepted_lines_v_med), np.array(accepted_lines_pow_lin_max))


def exclude_lines_far_from_main_one(v_ny, accepted_lines, accepted_lines_v_idx, accepted_lines_r_idx,
                                    accepted_lines_v, accepted_lines_r, accepted_lines_min_r,
                                    accepted_lines_max_r, accepted_lines_v_med_array,
                                    accepted_lines_pow_lin_max_array):
    """Excludes lines too far from the one with the highest peak power (not the largest span: a
    weak, noise-level line can easily span more range gates than the genuinely strong echo)."""
    idx_main_line = np.argmax(accepted_lines_pow_lin_max_array)
    dist_v_from_main_line = np.abs(accepted_lines_v_med_array - accepted_lines_v_med_array[idx_main_line])
    accepted_idx_dist = dist_v_from_main_line < v_ny

    out_lines, out_v_idx, out_r_idx, out_v, out_r = [], [], [], [], []
    for i_idx in np.arange(len(accepted_lines))[accepted_idx_dist]:
        out_lines.append(accepted_lines[i_idx])
        out_v_idx.append(accepted_lines_v_idx[i_idx])
        out_r_idx.append(accepted_lines_r_idx[i_idx])
        out_v.append(accepted_lines_v[i_idx])
        out_r.append(accepted_lines_r[i_idx])
    return out_lines, out_v_idx, out_r_idx, out_v, out_r


def extract_spectrum_around_peaks(spec, m, r_idx_peaks, v_idx_peaks, accepted_lines_v2):
    mask_spec = np.ones(spec.shape, dtype=bool)
    peak_spectrum_masked_dic: dict[int, list[int]] = {}
    indexes_v = np.arange(spec.shape[1], dtype=int)

    for l in accepted_lines_v2:
        curr_peak_r = r_idx_peaks[l]
        curr_peak_v = v_idx_peaks[l]
        mask_spec[curr_peak_r, curr_peak_v] = False
        for i_r_idx, r_idx in enumerate(curr_peak_r):
            peak_spectrum_masked_dic.setdefault(r_idx, []).append(v_idx_peaks[l][i_r_idx])

    for i_r in np.where(
        np.logical_and(np.sum(np.logical_not(mask_spec), axis=1) < m, np.sum(np.logical_not(mask_spec), axis=1) > 0)
    )[0]:
        num_gates_to_add = m - np.sum(1 - mask_spec[i_r, :])
        while num_gates_to_add > 0:
            erosion = scipy.ndimage.binary_erosion(mask_spec[i_r, :], border_value=1)
            candidates_to_add = indexes_v[np.logical_xor(erosion, mask_spec[i_r, :])]
            to_add = candidates_to_add[np.argmax(spec[i_r, :][candidates_to_add])]
            mask_spec[i_r, to_add] = False
            num_gates_to_add = m - np.sum(1 - mask_spec[i_r, :])

    masked_spectrum = np.ma.masked_array(spec, mask=mask_spec)
    return masked_spectrum, peak_spectrum_masked_dic


def compute_noise_lvl_std(r, masked_spectrum, peak_spectrum_masked_dic):
    mask_spec = masked_spectrum.mask
    indexes_v = np.arange(mask_spec.shape[1], dtype=int)
    noise_mask = np.ones(mask_spec.shape, dtype=bool)
    noise_lvl = np.zeros(r.shape[0])
    noise_std = np.zeros(r.shape[0])

    for i_r in np.arange(r.shape[0], dtype=int)[~np.all(mask_spec, axis=1)]:
        curr_spec = masked_spectrum[i_r, :]
        unmasked_part = ~curr_spec.mask
        x = curr_spec[unmasked_part]
        idx_array = indexes_v[unmasked_part]

        curr_valid_peaks = np.intersect1d(peak_spectrum_masked_dic[i_r], indexes_v[~mask_spec[i_r, :]])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*partition.*will ignore the.*mask", category=UserWarning)
            curr_valid_peaks = curr_valid_peaks[np.argsort(masked_spectrum[i_r, :][curr_valid_peaks])]

        curr_mask_sum = np.zeros(idx_array.shape, dtype=bool)
        for peak in curr_valid_peaks:
            mask = idx_array == peak
            old_mean = np.mean(x)
            new_mean = np.mean(np.ma.masked_array(x, mask=mask))
            while np.sum(~mask):
                old_mean = new_mean
                candidates = np.logical_xor(mask, scipy.ndimage.binary_dilation(mask))
                idx = np.where(candidates)[0][np.argmax(x[candidates])]
                mask[idx] = True
                new_mean = np.mean(np.ma.masked_array(x, mask=mask))
                if old_mean - new_mean < DA_THRESHOLD:
                    break
            curr_mask_sum += mask

        noise_mask[i_r, unmasked_part] = np.logical_not(curr_mask_sum)
        noise = np.ma.masked_array(x.data, mask=curr_mask_sum)
        signal = np.ma.masked_array(x.data, mask=~curr_mask_sum)
        if (~noise.mask).sum():
            noise_lvl[i_r] = np.nanmean(noise)
            noise_std[i_r] = np.nanstd(noise)
        else:
            noise_lvl[i_r] = np.nanmin(signal)

    return noise_mask, noise_lvl, noise_std


def correct_noise_lvl(noise_lvl_raw, standard_noise_lvl, noise_corr_window, max_diff):
    kernel = astropy.convolution.Box1DKernel(width=noise_corr_window)
    condition_lvl = np.logical_and(
        noise_lvl_raw > 0.0, np.abs(np.subtract(noise_lvl_raw, standard_noise_lvl)) < max_diff
    )
    noise_lvl_tmp = np.full(noise_lvl_raw.shape, np.nan)
    noise_lvl_tmp[condition_lvl] = noise_lvl_raw[condition_lvl]
    noise_lvl_tmp[np.logical_not(condition_lvl)] = standard_noise_lvl[np.logical_not(condition_lvl)]

    noise_lvl = astropy.convolution.convolve(
        noise_lvl_tmp, kernel, boundary="fill", fill_value=0.0, nan_treatment="interpolate", preserve_nan=True
    )
    return noise_lvl


def convert_spectrum_to_reflectivity(raw_spec, noise_lvl, noise_std, d_r, transfer_function,
                                     calibration_constant, noise_std_factor=0.0,
                                     remove_isolated_peals=True):
    if noise_std_factor > 0.0:
        noise_lvl = noise_lvl + (noise_std * noise_std_factor)

    spec_out = raw_spec - noise_lvl[:, None]
    spec_out.mask[spec_out < 0] = True
    if noise_std_factor > 0.0:
        spec_out = spec_out + (noise_std * noise_std_factor)[:, None]

    if remove_isolated_peals:
        img = spec_out.mask == False  # noqa: E712
        eroded_img = scipy.ndimage.binary_erosion(img)
        label, num_features = scipy.ndimage.label(img)
        for i_feat in range(1, num_features + 1):
            curr_region = label == i_feat
            if not np.sum(eroded_img[curr_region]):
                spec_out.mask[curr_region] = True

    N = raw_spec.shape[0]
    m_x3 = raw_spec.shape[1]
    n_square_mat = np.square(np.tile(np.arange(1, N + 1), (m_x3, 1)).T)

    with np.errstate(divide="ignore", invalid="ignore"):
        spec_out = np.divide(spec_out * calibration_constant, transfer_function) * n_square_mat * d_r
        noise_lvl_out = (
            np.divide(noise_lvl * calibration_constant, transfer_function[:, 0]) * n_square_mat[:, 0] * d_r
        )
        noise_std_out = (
            np.divide(noise_std * calibration_constant, transfer_function[:, 0]) * n_square_mat[:, 0] * d_r
        )
    noise_floor = int(m_x3 / 3) * noise_lvl_out

    return spec_out, noise_lvl_out, noise_std_out, noise_floor


def compute_spectra_parameters(spec_refined, vel_array, noise_floor):
    power = np.nansum(spec_refined, axis=1)
    z = CONST_Z_CALC * power
    noise_floor_z = CONST_Z_CALC * noise_floor
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = 10 * np.log10(power / noise_floor)

    weights = spec_refined / power[:, None]
    with np.errstate(divide="ignore"):
        m1_dop = np.sum(vel_array * weights, axis=1)
        m2_dop = np.sqrt(np.sum(weights * (vel_array - m1_dop[:, None]) ** 2, axis=1))

    return {"z": z, "m1_dop": m1_dop, "m2_dop": m2_dop, "noise_floor_z": noise_floor_z, "snr": snr}


def convert_spectrum_parameters_to_dBZ(noise_masked_spectrum, noise_lvl, spectrum_params):
    with np.errstate(divide="ignore", invalid="ignore"):
        spectrum_reflectivity = 10.0 * np.log10(noise_masked_spectrum * CONST_Z_CALC)
    output_dic = {
        "spectrum_reflectivity": spectrum_reflectivity,
        "Zea": 10.0 * np.log10(spectrum_params["z"]),
        "VEL": spectrum_params["m1_dop"],
        "WIDTH": spectrum_params["m2_dop"],
        "SNR": spectrum_params["snr"],
        "noise_level": 10.0 * np.log10(noise_lvl),
        "noise_floor": 10.0 * np.log10(spectrum_params["noise_floor_z"]),
    }
    return output_dic


def process_single_spectrum(spec, v_0_3, r, m, v_ny, d_r, transfer_function_x3, calibration_constant,
                            standard_noise_lvl, prev_v_ref=0.0, external_v_ref=np.nan):
    """
    Processes one spectrum: peak/line finding, de-aliasing, noise removal, moment computation.

    Reference-velocity priority for de-aliasing: external_v_ref (e.g. the manufacturer's own
    de-aliased VEL for this exact time step - independent per spectrum, so it cannot inherit or
    self-reinforce a de-aliasing error from a neighboring spectrum) > prev_v_ref (carried over
    from the previous spectrum) > DEALIAS_DEFAULT_REF_FRAC * v_ny.

    Returns (output_dic, new_v_ref): new_v_ref is np.nan whenever no signal was found.
    """
    N = r.shape[0]
    r_idx_peaks, v_idx_peaks, v_l_idx_peaks, v_r_idx_peaks, idx_peaks = find_raw_peaks(spec, N, m)
    if not len(r_idx_peaks):
        return {}, np.nan

    (lines_array, line_v_idx, line_r_idx, line_v, line_r, line_pow_lin, line_min_r, line_max_r,
     line_median_v, line_median_pow_lin) = find_raw_lines(spec, v_0_3, r, r_idx_peaks, v_idx_peaks, idx_peaks)
    if not len(line_median_v):
        return {}, np.nan

    if np.isfinite(external_v_ref):
        dealias_ref_v = external_v_ref
    elif np.isfinite(prev_v_ref):
        dealias_ref_v = prev_v_ref
    else:
        dealias_ref_v = DEALIAS_DEFAULT_REF_FRAC * v_ny

    (accepted_lines, accepted_lines_v_idx, accepted_lines_r_idx, accepted_lines_v, accepted_lines_r,
     accepted_lines_min_r, accepted_lines_max_r, accepted_lines_v_med_array,
     accepted_lines_pow_lin_max_array) = exclude_duplicate_lines(
        v_ny, lines_array, line_v_idx, line_r_idx, line_v, line_r, line_pow_lin, line_min_r,
        line_max_r, line_median_v, prev_v_ref=dealias_ref_v
    )

    # A single echo split into several disjoint range segments gets de-aliased independently
    # above; nothing forces them to agree on the same v_ny branch. Re-resolve using the
    # strongest (highest peak power) segment's own velocity as the reference.
    if len(accepted_lines) > 1:
        anchor_idx = np.argmax(accepted_lines_pow_lin_max_array)
        anchor_v = accepted_lines_v_med_array[anchor_idx]
        (accepted_lines, accepted_lines_v_idx, accepted_lines_r_idx, accepted_lines_v, accepted_lines_r,
         accepted_lines_min_r, accepted_lines_max_r, accepted_lines_v_med_array,
         accepted_lines_pow_lin_max_array) = exclude_duplicate_lines(
            v_ny, lines_array, line_v_idx, line_r_idx, line_v, line_r, line_pow_lin, line_min_r,
            line_max_r, line_median_v, prev_v_ref=anchor_v
        )

    accepted_lines_v2, _, _, _, _ = exclude_lines_far_from_main_one(
        v_ny, accepted_lines, accepted_lines_v_idx, accepted_lines_r_idx, accepted_lines_v,
        accepted_lines_r, accepted_lines_min_r, accepted_lines_max_r, accepted_lines_v_med_array,
        accepted_lines_pow_lin_max_array,
    )

    masked_spectrum, peak_spectrum_masked_dic = extract_spectrum_around_peaks(
        spec, m, r_idx_peaks, v_idx_peaks, accepted_lines_v2
    )

    noise_mask, noise_lvl_raw, noise_std = compute_noise_lvl_std(r, masked_spectrum, peak_spectrum_masked_dic)
    noise_masked_spectrum = np.ma.masked_array(spec, mask=noise_mask)

    if CORRECT_NOISE_LVL:
        noise_lvl = correct_noise_lvl(noise_lvl_raw, standard_noise_lvl, NOISE_CORR_WINDOW, MAX_DIFF_NOISE_LVL)
    else:
        noise_lvl = noise_lvl_raw

    noise_masked_spectrum_cal, noise_lvl_cal, noise_std_cal, noise_floor_cal = convert_spectrum_to_reflectivity(
        noise_masked_spectrum, noise_lvl, noise_std, d_r, transfer_function_x3, calibration_constant,
        noise_std_factor=NOISE_STD_FACTOR, remove_isolated_peals=REMOVE_ISOLATED_PEAK_SPECTRUM,
    )

    spectrum_params = compute_spectra_parameters(noise_masked_spectrum_cal, v_0_3, noise_lvl_cal)
    spectrum_params_dBZ = convert_spectrum_parameters_to_dBZ(noise_masked_spectrum_cal, noise_floor_cal, spectrum_params)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*partition.*will ignore the.*mask", category=UserWarning)
        new_v_ref = np.nanmedian(spectrum_params_dBZ["VEL"])
    return spectrum_params_dBZ, new_v_ref


# ----------------------------------------------------------------------------------------------
# 3) POSTPROCESSING: cleanup of the processed (time, range) moments.
# ----------------------------------------------------------------------------------------------

def identify_interference_lines(zea: np.ndarray) -> np.ndarray:
    """Flags suspiciously elongated (in time) features in the processed reflectivity matrix."""
    if zea.shape[0] == 0 or zea.shape[1] == 0:
        return np.zeros(zea.shape, dtype=float)

    window_t = min(WINDOW_POSTPROCESS_T, zea.shape[0])
    window_r = min(WINDOW_POSTPROCESS_R, zea.shape[1])
    min_t_len = MIN_HALF_FRACTION * window_t
    hw_t = int(window_t / 2.0)
    hw_r = int(window_r / 2.0)

    investigated_r_idx = np.arange(zea.shape[1])[
        np.sum(np.isfinite(zea), axis=0) > MIN_TIME_FRACTION_INTERF_POSTPROC * zea.shape[0]
    ]

    t_min_array = np.max(
        np.stack([np.arange(zea.shape[0], dtype=int) - hw_t, np.zeros(zea.shape[0], dtype=int)], axis=1), axis=1
    )
    t_max_array = t_min_array + window_t
    t_min_array[t_max_array > zea.shape[0]] = zea.shape[0] - window_t
    t_max_array[t_max_array > zea.shape[0]] = zea.shape[0]

    r_low_array = np.max(
        np.stack([investigated_r_idx - hw_r, np.zeros(investigated_r_idx.shape, dtype=int)], axis=1), axis=1
    )
    r_top_array = r_low_array + window_r
    r_low_array[r_top_array > zea.shape[1]] = zea.shape[1] - window_r
    r_top_array[r_top_array > zea.shape[1]] = zea.shape[1]

    interf_flag = np.zeros(zea.shape, dtype=float)
    kernel_std = max(window_t / 8.0, 1.0)
    sus_weights = window_t * np.array(astropy.convolution.Gaussian1DKernel(kernel_std, x_size=window_t))

    for i_t in range(zea.shape[0]):
        for i_r in range(investigated_r_idx.shape[0]):
            t_valid = np.sum(np.isfinite(zea[t_min_array[i_t]:t_max_array[i_t], investigated_r_idx[i_r]]))
            if t_valid <= min_t_len:
                continue
            r_valid = max(1, np.sum(np.isfinite(zea[i_t, r_low_array[i_r]:r_top_array[i_r]])))
            if (t_valid / r_valid) <= MIN_RATIO_H_V:
                continue
            closest_to_min_t = t_max_array[i_t] - i_t - hw_t
            if not closest_to_min_t:
                interf_flag[t_min_array[i_t]:t_max_array[i_t], investigated_r_idx[i_r]] += sus_weights
            elif closest_to_min_t > 0:
                interf_flag[t_min_array[i_t]:t_max_array[i_t] - closest_to_min_t, investigated_r_idx[i_r]] += \
                    sus_weights[closest_to_min_t:]
            else:
                interf_flag[t_min_array[i_t] - closest_to_min_t:t_max_array[i_t], investigated_r_idx[i_r]] += \
                    sus_weights[:closest_to_min_t]

    return interf_flag


def identify_isolated_artifacts_2d(zea_post: np.ndarray, to_remove: np.ndarray) -> np.ndarray:
    label, num_features = scipy.ndimage.label(np.isfinite(zea_post))
    for i_feat in range(1, num_features + 1):
        curr_region = label == i_feat
        if np.sum(curr_region) < MIN_NUM_PIXEL_NOISE_REMOVAL:
            to_remove[curr_region] = True
    return to_remove


def postprocess_moments(zea, vel, s_w, snr, noise_level, noise_floor):
    """Cleans up the processed (time, range) moments: SNR threshold, interference-line removal,
    and removal of small isolated artifacts. Mirrors ERUO's postprocess_file, operating directly
    on arrays instead of a file."""
    to_remove = np.zeros(zea.shape, dtype=bool)
    zea_post = np.full(zea.shape, np.nan)

    to_remove[snr < MIN_SNR_POSTPROC] = True
    zea_post[np.logical_not(to_remove)] = zea[np.logical_not(to_remove)]

    if REMOVE_INTERF_POSTPROC:
        interf_flag = identify_interference_lines(zea_post)
        to_remove[interf_flag > MIN_INTERF_FLAG] = True
        zea_post[to_remove] = np.nan

    if REMOVE_NOISE_POSTPROC:
        to_remove = identify_isolated_artifacts_2d(zea_post, to_remove)

    Zea_post = copy.deepcopy(zea)
    VEL_post = copy.deepcopy(vel)
    WIDTH_post = copy.deepcopy(s_w)
    SNR_post = copy.deepcopy(snr)
    noise_level_post = copy.deepcopy(noise_level)
    noise_floor_post = copy.deepcopy(noise_floor)

    Zea_post[to_remove] = np.nan
    VEL_post[to_remove] = np.nan
    WIDTH_post[to_remove] = np.nan
    SNR_post[to_remove] = np.nan
    noise_level_post[to_remove] = np.nan
    noise_floor_post[to_remove] = np.nan

    return Zea_post, VEL_post, WIDTH_post, SNR_post, noise_level_post, noise_floor_post, to_remove


# ----------------------------------------------------------------------------------------------
# Backend
# ----------------------------------------------------------------------------------------------

def _seconds_between(t0, t1) -> float:
    """Time difference in seconds between two values of a (possibly datetime64) time coordinate."""
    diff = t1 - t0
    if np.issubdtype(np.asarray(diff).dtype, np.timedelta64):
        return float(np.asarray(diff).astype("timedelta64[ns]").astype(float) / 1e9)
    return float(diff)


class MRRBackend(RadarBackend):
    """Backend implementation for Metek MRR-PRO raw spectra (preprocessing + processing +
    postprocessing folded into a single, self-contained pass)."""

    name = "mrr"

    @staticmethod
    def can_handle(ds: xr.Dataset) -> bool:
        return "spectrum_raw" in ds.variables

    @staticmethod
    def compute_preprocessing_spectra_q(
        datasets: list[xr.Dataset],
    ) -> np.ndarray | None:
        """
        Pre-compute the campaign-wide spectra_q (median spectrum) from a list of raw datasets.

        Call this once before processing multiple files and pass the result as
        preprocessing_spectra_q to process() to avoid repeating the expensive median
        computation on every call.

        Returns None if no usable spectra are found.
        """
        if not datasets:
            return None

        first = datasets[0]
        range_len = first.sizes["range"]
        non_time_dims_0 = [d for d in first["spectrum_raw"].dims if d != "time"]
        range_like_0 = next((d for d in non_time_dims_0 if first.sizes[d] == range_len), None)
        if range_like_0 is None:
            return None
        spec_dim_0 = next(d for d in non_time_dims_0 if d != range_like_0)
        expected_shape = (range_len, first.sizes[spec_dim_0])

        arrays = []
        for ds in datasets:
            try:
                spec_da = ds["spectrum_raw"]
                non_time_dims = [d for d in spec_da.dims if d != "time"]
                range_like = next((d for d in non_time_dims if ds.sizes[d] == range_len), None)
                if range_like is None:
                    continue
                spec_dim = next(d for d in non_time_dims if d != range_like)
                if range_like != "range":
                    spec_da = spec_da.rename({range_like: "range"})
                arr = np.asarray(
                    spec_da.transpose("time", "range", spec_dim).values, dtype=float
                )
                if arr.shape[1:] == expected_shape:
                    arrays.append(arr)
            except Exception:
                continue

        if not arrays:
            return None
        return np.nanmedian(np.concatenate(arrays, axis=0), axis=0)

    def process(
        self,
        ds: xr.Dataset,
        *,
        velRef=None,
        include_moments: bool = True,
        include_ldr: bool = True,
        preprocessing_datasets: list[xr.Dataset] | None = None,
        preprocessing_spectra_q: np.ndarray | None = None,
        **_,
    ) -> xr.Dataset:
        # --- Read raw inputs ---------------------------------------------------------------
        # MRR-PRO files commonly name the spectrum's range-like dim differently from the
        # actual "range" coordinate (e.g. "n_spectra"); match by size rather than guessing
        # from name/position, since that is unambiguous.
        range_len = ds.sizes["range"]
        non_time_dims = [d for d in ds["spectrum_raw"].dims if d != "time"]
        range_like_dim = next((d for d in non_time_dims if ds.sizes[d] == range_len), None)
        if range_like_dim is None:
            raise ValueError("Could not match spectrum_raw's range-like dimension to 'range'.")
        spec_dim = next(d for d in non_time_dims if d != range_like_dim)

        spec_da = ds["spectrum_raw"]
        if range_like_dim != "range":
            spec_da = spec_da.rename({range_like_dim: "range"})

        all_spectra_raw = np.asarray(spec_da.transpose("time", "range", spec_dim).values, dtype=float)
        r = np.asarray(ds["range"].values, dtype=float)
        t = ds["time"].values
        transfer_function = np.asarray(ds["transfer_function"].values, dtype=float)
        if RECONSTRUCT_TRANSFER_FUNCTION:
            transfer_function = reconstruct_transfer_function(transfer_function)
        calibration_constant = float(np.asarray(ds["calibration_constant"].values)) / CALIB_CONST_FACTOR
        manufacturer_vel = np.asarray(ds["VEL"].values, dtype=float) if "VEL" in ds.variables else None

        num_t, N, m = all_spectra_raw.shape
        diffs = [
            _seconds_between(t[i], t[i + 1]) for i in range(min(len(t) - 1, 50))
        ]
        T_i = float(np.round(np.median(diffs))) if diffs else 10.0
        d_r = float(np.round(np.median(np.diff(r))))

        info_dic = compute_additional_mrr_parameters(N, m, T_i, d_r)
        v_ny = info_dic["v_ny"]
        v_0 = info_dic["v_0"]

        # --- 1) Preprocessing: estimate interference mask / border correction / median ------
        # Priority: pre-computed spectra_q (fast path, computed once outside this call) >
        # preprocessing_datasets (compute here, slow if called repeatedly) > current file only.
        if preprocessing_spectra_q is not None:
            spectra_q = preprocessing_spectra_q
        elif preprocessing_datasets:
            extra_arrays = []
            for extra_ds in preprocessing_datasets:
                extra_spec_da = extra_ds["spectrum_raw"]
                non_time_dims_extra = [d for d in extra_spec_da.dims if d != "time"]
                range_like_extra = next(
                    (d for d in non_time_dims_extra if extra_ds.sizes[d] == range_len), None
                )
                if range_like_extra is None:
                    continue
                spec_dim_extra = next(d for d in non_time_dims_extra if d != range_like_extra)
                if range_like_extra != "range":
                    extra_spec_da = extra_spec_da.rename({range_like_extra: "range"})
                extra_arr = np.asarray(
                    extra_spec_da.transpose("time", "range", spec_dim_extra).values, dtype=float
                )
                if extra_arr.shape[1:] == all_spectra_raw.shape[1:]:
                    extra_arrays.append(extra_arr)
            spectra_for_preprocessing = (
                np.concatenate([all_spectra_raw] + extra_arrays, axis=0) if extra_arrays
                else all_spectra_raw
            )
            spectra_q = np.nanmedian(spectra_for_preprocessing, axis=0)
        else:
            spectra_q = np.nanmedian(all_spectra_raw, axis=0)

        interference_mask, border_correction, smooth_median_spec = estimate_preprocessing_products(spectra_q)

        # --- 2) Processing -------------------------------------------------------------------
        all_spectra_raw = all_spectra_raw + np.tile(border_correction, (num_t, 1, 1))

        if RECONSTRUCT_SPECTRUM:
            median_line_tiled = np.moveaxis(np.tile(smooth_median_spec, (num_t, m, 1)), 1, 2)
            anomaly_3d, reficiendo_3d = define_reficiendo(all_spectra_raw, median_line_tiled, interference_mask)
            if RESCUE_VIA_TIME_NEIGHBORS:
                reficiendo_3d = rescue_via_time_neighbors(anomaly_3d, reficiendo_3d)

            all_anomalies = np.full(anomaly_3d.shape, np.nan)
            for i_t in range(num_t):
                all_anomalies[i_t, :, :] = reconstruct_anomaly(anomaly_3d[i_t, :, :], reficiendo_3d[i_t, :, :])
            all_spectra = median_line_tiled + all_anomalies
        else:
            all_spectra = all_spectra_raw

        all_spectra_x3_lin, transfer_function_x3 = repeat_spectra(all_spectra, transfer_function)

        v_0_3 = np.tile(v_0, 3)
        v_0_3[0:m] -= v_ny
        v_0_3[2 * m:] += v_ny

        smooth_median_spec_lin = np.power(10.0, smooth_median_spec / 10.0)

        out_varnames = ["spectrum_reflectivity", "Zea", "VEL", "WIDTH", "SNR", "noise_level", "noise_floor"]
        empty_var_dic = {out_varnames[0]: np.full((N, 3 * m), np.nan, dtype="float32")}
        for varname in out_varnames[1:]:
            empty_var_dic[varname] = np.full(N, np.nan, dtype="float32")

        new_vars_dic: dict[str, list] = {}
        prev_v_ref = np.nan
        for i_t in range(num_t):
            external_v_ref = np.nanmedian(manufacturer_vel[i_t, :]) if manufacturer_vel is not None else np.nan

            output_dic, prev_v_ref = process_single_spectrum(
                all_spectra_x3_lin[i_t, :, :], v_0_3, r, m, v_ny, d_r, transfer_function_x3,
                calibration_constant, smooth_median_spec_lin, prev_v_ref=prev_v_ref, external_v_ref=external_v_ref,
            )
            if not len(output_dic.keys()):
                output_dic = empty_var_dic
            for k in output_dic.keys():
                new_vars_dic.setdefault(k, []).append(output_dic[k])

        concatenated: dict[str, np.ndarray] = {}
        for k, v in new_vars_dic.items():
            arr = np.ma.stack(v)
            arr[arr.mask] = np.nan
            concatenated[k] = np.asarray(arr)

        # --- 3) Postprocessing ----------------------------------------------------------------
        Zea_post, VEL_post, WIDTH_post, SNR_post, noise_level_post, noise_floor_post, _ = postprocess_moments(
            concatenated["Zea"], concatenated["VEL"], concatenated["WIDTH"], concatenated["SNR"],
            concatenated["noise_level"], concatenated["noise_floor"],
        )

        # --- Assemble output dataset -----------------------------------------------------------
        out = xr.Dataset(
            data_vars={
                "sZe": (("time", "range", "Vel"), concatenated["spectrum_reflectivity"].astype("float32")),
                "Zea": (("time", "range"), Zea_post.astype("float32")),
                "VEL": (("time", "range"), VEL_post.astype("float32")),
                "WIDTH": (("time", "range"), WIDTH_post.astype("float32")),
                "SNR": (("time", "range"), SNR_post.astype("float32")),
                "noise_level": (("time", "range"), noise_level_post.astype("float32")),
                "noise_floor": (("time", "range"), noise_floor_post.astype("float32")),
            },
            coords={"time": t, "range": r, "Vel": v_0_3.astype("float32")},
        )

        vr = as_vel_ref(velRef)
        if vr is not None:
            out = out.reindex({"Vel": vr}, method="nearest", tolerance=0.05)

        out = out.sortby("time")
        out = add_standard_variable_attrs(out)
        out["sZe"].attrs.update({"long_name": "MRR-PRO Doppler spectral reflectivity", "units": "dBZ/bin"})
        out["Zea"].attrs.update({"long_name": "MRR-PRO equivalent reflectivity factor", "units": "dBZ"})
        out["VEL"].attrs.update({"long_name": "MRR-PRO mean Doppler velocity", "units": "m s-1"})
        out["WIDTH"].attrs.update({"long_name": "MRR-PRO Doppler spectral width", "units": "m s-1"})
        out["SNR"].attrs.update({"long_name": "MRR-PRO signal-to-noise ratio", "units": "dB"})
        out["noise_level"].attrs.update({"long_name": "MRR-PRO estimated noise level", "units": "dB"})
        out["noise_floor"].attrs.update({"long_name": "MRR-PRO estimated noise floor", "units": "dBZ"})

        return finalize_metadata(out, backend_name=self.name, include_moments=include_moments, include_ldr=False)

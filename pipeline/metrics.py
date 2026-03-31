"""
Clinical metrics:
- MRD1 (Margin Reflex Distance 1) in mm
- PHUL (Peak Height of Upper Lid) + nasal/temporal position
- Temporal/Nasal ocular surface area ratio (T/N)
- Symmetry % via normalized curve overlap
  (Aytogan & Ayıntap, BMC Ophthalmology 2021, DOI:10.1186/s12886-021-02208-7)

No "operated vs reference" concept — both eyes measured equally.
Symmetry is computed by mirroring the lower-MRD1 eye and comparing to the other.
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple
from dataclasses import dataclass


@dataclass
class EyeMetrics:
    mrd1_mm:         float
    phul_mm:         float
    phul_position:   str     # "nasal" | "central" | "temporal"
    phul_offset_mm:  float   # + = temporal, - = nasal from pupil center
    temporal_area:   float   # px²
    nasal_area:      float   # px²
    tn_ratio:        float
    px_per_mm:       float


@dataclass
class SymmetryResult:
    symmetry_pct:   float
    lower_mrd1_side: str          # "right" or "left" (more ptotic eye)
    ref_curve:      np.ndarray    # normalized reference (higher MRD1)
    flipped_curve:  np.ndarray    # normalized + flipped lower-MRD1 eye


# ── Individual metric functions ───────────────────────────────────────────────

def compute_mrd1(pupil_center, lid_curve, px_per_mm) -> float:
    """
    MRD1: vertical distance from pupil center to upper lid at pupil x.
    Positive = lid above pupil center (normal ≈ 3-4 mm).
    Negative = lid covers pupil (ptosis).
    """
    idx = np.argmin(np.abs(lid_curve[:, 0] - pupil_center[0]))
    lid_y = lid_curve[idx, 1]
    return (pupil_center[1] - lid_y) / px_per_mm


def compute_phul(pupil_center, lid_curve, px_per_mm, side) -> Tuple[float, str, float]:
    """
    PHUL: height of the highest lid point above pupil center.
    Returns (phul_mm, position_label, offset_mm_from_pupil_center).
    """
    apex_idx = np.argmin(lid_curve[:, 1])
    apex     = lid_curve[apex_idx]

    phul_mm   = (pupil_center[1] - apex[1]) / px_per_mm
    offset_mm = (apex[0] - pupil_center[0]) / px_per_mm  # + = image-right

    THRESH = 0.5
    if abs(offset_mm) <= THRESH:
        position = "central"
    elif side == "right":
        position = "temporal" if offset_mm > 0 else "nasal"
    else:
        position = "temporal" if offset_mm < 0 else "nasal"

    return phul_mm, position, offset_mm


def compute_tn_ratio(pupil_center, lid_curve, lower_boundary_y, side) -> Tuple[float, float, float]:
    """
    Temporal and nasal ocular surface areas split at the pupil vertical axis.
    """
    cx = pupil_center[0]
    sorted_c = lid_curve[np.argsort(lid_curve[:, 0])]

    if side == "right":
        t_pts = sorted_c[sorted_c[:, 0] >= cx]
        n_pts = sorted_c[sorted_c[:, 0] <  cx]
    else:
        t_pts = sorted_c[sorted_c[:, 0] <= cx]
        n_pts = sorted_c[sorted_c[:, 0] >  cx]

    def area(pts):
        if len(pts) < 2:
            return 0.0
        h = np.maximum(lower_boundary_y - pts[:, 1], 0)
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        return float(_trapz(h, pts[:, 0]))

    t_a = area(t_pts)
    n_a = area(n_pts)
    return t_a, n_a, (t_a / n_a if n_a > 1e-6 else float("inf"))


# ── Symmetry ──────────────────────────────────────────────────────────────────

def compute_symmetry(
    right_curve, left_curve,
    right_pupil, left_pupil,
    right_ppmm, left_ppmm,
    right_mrd1, left_mrd1,
) -> SymmetryResult:
    """
    Normalize both curves to pupil-centered mm coordinates.
    Mirror the lower-MRD1 (more ptotic) eye's curve horizontally.
    Compute overlap % (Aytogan 2021 method).
    """
    lower_side = "right" if right_mrd1 <= left_mrd1 else "left"

    def normalize(curve, pupil, ppmm):
        c = curve.astype(float).copy()
        c[:, 0] = (c[:, 0] - pupil[0]) / ppmm   # mm from pupil center, + = image-right
        c[:, 1] = (pupil[1] - c[:, 1]) / ppmm   # mm above pupil, + = up
        return c

    ref_curve  = normalize(
        right_curve if lower_side == "left" else left_curve,
        right_pupil if lower_side == "left" else left_pupil,
        right_ppmm  if lower_side == "left" else left_ppmm,
    )
    flip_curve = normalize(
        left_curve  if lower_side == "left" else right_curve,
        left_pupil  if lower_side == "left" else right_pupil,
        left_ppmm   if lower_side == "left" else right_ppmm,
    )

    # Mirror the ptotic eye so temporal/nasal align with the reference
    flip_curve[:, 0] = -flip_curve[:, 0]

    # Common x range
    x_min = max(ref_curve[:, 0].min(), flip_curve[:, 0].min())
    x_max = min(ref_curve[:, 0].max(), flip_curve[:, 0].max())

    if x_max <= x_min:
        return SymmetryResult(0.0, lower_side, ref_curve, flip_curve)

    x_common = np.linspace(x_min, x_max, 500)

    def interp(curve, x_new):
        s = curve[np.argsort(curve[:, 0])]
        _, uid = np.unique(s[:, 0], return_index=True)
        s = s[uid]
        if len(s) < 2:
            return np.zeros_like(x_new)
        return interp1d(s[:, 0], s[:, 1],
                        bounds_error=False, fill_value="extrapolate")(x_new)

    y_ref  = interp(ref_curve,  x_common)
    y_flip = interp(flip_curve, x_common)

    mean_ref = np.mean(np.abs(y_ref))
    if mean_ref < 1e-6:
        sym_pct = 0.0
    else:
        sym_pct = max(0.0, min(100.0,
                     (1.0 - np.mean(np.abs(y_ref - y_flip)) / mean_ref) * 100.0))

    return SymmetryResult(sym_pct, lower_side, ref_curve, flip_curve)


# ── Combined entry point ──────────────────────────────────────────────────────

def compute_all_metrics(
    right_curve, left_curve,
    right_eye, left_eye,
) -> Tuple[EyeMetrics, EyeMetrics, SymmetryResult]:
    """Compute all metrics for both eyes. No operated-side concept."""

    r_lower = right_eye.pupil_center[1] + right_eye.iris_radius_px
    l_lower = left_eye.pupil_center[1]  + left_eye.iris_radius_px

    r_mrd1 = compute_mrd1(right_eye.pupil_center, right_curve, right_eye.px_per_mm)
    r_phul, r_pos, r_off = compute_phul(
        right_eye.pupil_center, right_curve, right_eye.px_per_mm, "right")
    r_t, r_n, r_tn = compute_tn_ratio(
        right_eye.pupil_center, right_curve, r_lower, "right")
    right_m = EyeMetrics(r_mrd1, r_phul, r_pos, r_off, r_t, r_n, r_tn, right_eye.px_per_mm)

    l_mrd1 = compute_mrd1(left_eye.pupil_center, left_curve, left_eye.px_per_mm)
    l_phul, l_pos, l_off = compute_phul(
        left_eye.pupil_center, left_curve, left_eye.px_per_mm, "left")
    l_t, l_n, l_tn = compute_tn_ratio(
        left_eye.pupil_center, left_curve, l_lower, "left")
    left_m = EyeMetrics(l_mrd1, l_phul, l_pos, l_off, l_t, l_n, l_tn, left_eye.px_per_mm)

    symmetry = compute_symmetry(
        right_curve, left_curve,
        right_eye.pupil_center, left_eye.pupil_center,
        right_eye.px_per_mm, left_eye.px_per_mm,
        r_mrd1, l_mrd1,
    )

    return right_m, left_m, symmetry

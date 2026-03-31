"""
Upper eyelid contour — sclera-anchored MediaPipe correction.

Strategy
--------
1. Find the topmost TRUE SCLERA pixels in the nasal and temporal strips
   flanking the iris.  Sclera is white (V>150, S<30 in HSV) and clearly
   distinguishable from orange skin (S>30) and dark iris (V<150).
   → Topmost sclera per strip  =  upper lid margin at that x position.

2. MediaPipe gives the correct SHAPE of the eyelid arc (7 lid points),
   but the absolute height is often wrong.  Compute a per-column shift
   that maps MediaPipe at the nasal / temporal iris edges onto the sclera
   anchors.  Interpolate that shift linearly across x.

3. Apply the shift to every MediaPipe lid point  →  corrected control pts.

4. Outlier-reject → Savitzky-Golay → Bezier 1000 pts.

Fallback: raw MediaPipe (if no sclera anchors found).
"""

import numpy as np
import cv2
from scipy.signal import savgol_filter
from scipy.special import comb
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from typing import Tuple, Optional


# ── Bezier ────────────────────────────────────────────────────────────────────
def _bernstein(n, i, t):
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier_curve(ctrl: np.ndarray, n_points: int = 1000) -> np.ndarray:
    n = len(ctrl) - 1
    t = np.linspace(0, 1, n_points)
    curve = np.zeros((n_points, 2))
    for i, pt in enumerate(ctrl):
        curve += np.outer(_bernstein(n, i, t), pt)
    return curve


# ── Sclera-strip top-of-lid detection ────────────────────────────────────────
def _sclera_strip_lid_y(
    sclera_ok: np.ndarray,       # boolean mask (h, w)
    x_lo: int, x_hi: int,
    cy: float,
    px_per_mm: float,
    search_above_mm: float = 8.0,
    min_pixels: int = 4,
) -> Optional[float]:
    """
    For each column in [x_lo, x_hi], find the topmost (min y) sclera pixel
    in the window [cy - search_above_mm*ppmm, cy).
    Returns median over all qualifying columns.
    Requires at least min_pixels sclera pixels per column.
    """
    h = sclera_ok.shape[0]
    y_search_top = int(np.clip(cy - search_above_mm * px_per_mm, 0, h - 1))
    y_search_bot = int(np.clip(cy, 0, h - 1))

    ys = []
    for x in range(x_lo, x_hi + 1):
        col  = sclera_ok[y_search_top:y_search_bot, x]
        rows = np.where(col)[0]
        if len(rows) >= min_pixels:
            ys.append(float(y_search_top + rows.min()))
    return float(np.median(ys)) if len(ys) >= 3 else None


# ── Sclera-anchored MediaPipe correction ─────────────────────────────────────
def _sclera_anchored_ctrl(
    image_bgr: np.ndarray,
    cx: float, cy: float, r: float,
    px_per_mm: float,
    upper_lid_margin: np.ndarray,   # MediaPipe 7 pts (x, y)
    n_ctrl: int = 9,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Returns (control_points, was_corrected).
    control_points: (n_ctrl, 2) sampled from the corrected lid arc.
    was_corrected:  True if at least one sclera anchor was found.
    """
    h, w = image_bgr.shape[:2]
    hsv  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    S    = hsv[:, :, 1]
    V    = hsv[:, :, 2]

    # True sclera mask: very white (V>150, S<30)
    sclera_ok = (V > 150) & (S < 30)

    # ── Sclera strip widths ───────────────────────────────────────────────
    # Nasal  : cx - 1.6r  →  cx - 0.8r
    # Temporal: cx + 0.8r  →  cx + 1.6r
    n_lo = int(np.clip(cx - 1.6 * r, 0, w - 1))
    n_hi = int(np.clip(cx - 0.8 * r, 0, w - 1))
    t_lo = int(np.clip(cx + 0.8 * r, 0, w - 1))
    t_hi = int(np.clip(cx + 1.6 * r, 0, w - 1))

    nasal_y    = _sclera_strip_lid_y(sclera_ok, n_lo, n_hi, cy, px_per_mm)
    temporal_y = _sclera_strip_lid_y(sclera_ok, t_lo, t_hi, cy, px_per_mm)

    if nasal_y is None and temporal_y is None:
        return None, False

    # ── MediaPipe interpolator ────────────────────────────────────────────
    mp = upper_lid_margin[np.argsort(upper_lid_margin[:, 0])].astype(float)
    if len(mp) < 2:
        return None, False

    f_mp = interp1d(mp[:, 0], mp[:, 1],
                    kind="linear", bounds_error=False,
                    fill_value=(float(mp[0, 1]), float(mp[-1, 1])))

    # Anchor x positions: nasal and temporal iris edges
    x_nasal    = cx - r
    x_temporal = cx + r

    mp_y_nasal    = float(f_mp(x_nasal))
    mp_y_temporal = float(f_mp(x_temporal))

    # Compute shifts at each anchor
    shift_nasal    = (nasal_y    - mp_y_nasal)    if nasal_y    is not None else None
    shift_temporal = (temporal_y - mp_y_temporal) if temporal_y is not None else None

    # If only one anchor, use it for both
    if shift_nasal    is None: shift_nasal    = shift_temporal
    if shift_temporal is None: shift_temporal = shift_nasal

    # ── Apply x-varying shift to every MediaPipe point ───────────────────
    corrected = []
    x_span = max(1.0, x_temporal - x_nasal)
    for x_mp, y_mp in mp:
        t     = np.clip((x_mp - x_nasal) / x_span, 0.0, 1.0)
        shift = shift_nasal * (1.0 - t) + shift_temporal * t
        y_new = float(y_mp + shift)
        # Keep strictly above pupil centre
        y_new = min(y_new, float(cy) - 1.0)
        corrected.append([float(x_mp), y_new])

    pts = np.array(corrected)
    pts = pts[np.argsort(pts[:, 0])]

    # Sample n_ctrl evenly-spaced control points
    idx = np.round(np.linspace(0, len(pts) - 1, n_ctrl)).astype(int)
    return pts[idx], True


# ── Smooth + Bezier ───────────────────────────────────────────────────────────
def _smooth_and_bezier(ctrl: np.ndarray):
    ctrl = ctrl[np.argsort(ctrl[:, 0])]
    y    = ctrl[:, 1].copy()
    med  = float(np.median(y))
    std  = float(np.std(y)) + 1.0
    bad  = (y > med + 1.5 * std) | (y < med - 1.5 * std)
    if bad.any() and not bad.all():
        x_ok, y_ok = ctrl[~bad, 0], ctrl[~bad, 1]
        if len(x_ok) >= 2:
            f = interp1d(x_ok, y_ok, kind="linear", fill_value="extrapolate")
            y[bad] = f(ctrl[bad, 0])
        ctrl = np.column_stack([ctrl[:, 0], y])

    n  = len(ctrl)
    wl = min(n - (0 if n % 2 else 1), 7)
    wl = max(5, wl) if wl % 2 else max(5, wl - 1)
    if n >= wl:
        sy   = savgol_filter(ctrl[:, 1], window_length=wl, polyorder=2)
        ctrl = np.column_stack([ctrl[:, 0], sy])

    curve  = bezier_curve(ctrl, n_points=1000)
    apex_y = float(np.min(curve[:, 1]))
    return curve, ctrl, apex_y


# ── Main extraction ───────────────────────────────────────────────────────────
def extract_upper_lid_contour(
    image_bgr: np.ndarray,
    pupil_center: np.ndarray,
    iris_radius_px: float,
    px_per_mm: float,
    medial_canthus: np.ndarray,
    lateral_canthus: np.ndarray,
    upper_lid_margin: np.ndarray,
    side: str = "right",
) -> Tuple[np.ndarray, np.ndarray, float]:

    cx   = float(pupil_center[0])
    cy   = float(pupil_center[1])
    r    = float(iris_radius_px)

    # ── Primary: sclera-anchored MediaPipe correction ─────────────────────
    ctrl, corrected = _sclera_anchored_ctrl(
        image_bgr, cx, cy, r, float(px_per_mm), upper_lid_margin,
    )

    # ── Fallback: raw MediaPipe ───────────────────────────────────────────
    if ctrl is None:
        mp = upper_lid_margin[np.argsort(upper_lid_margin[:, 0])].astype(float)
        ctrl = mp

    return _smooth_and_bezier(ctrl)

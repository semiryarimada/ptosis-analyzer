"""
PDF report & visualizations — style matches Aytogan & Ayıntap 2021.

Per-eye panel layout (2 × 2 matplotlib grid):
  Row 0 (black bg): normalized curves in mm, crosshairs, pupil dot
  Row 1 (photo):    color eye crop, curve overlaid, crosshairs, iris circle
"""

import io
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy.interpolate import interp1d

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from pipeline.metrics import EyeMetrics, SymmetryResult


# ── Color scheme ─────────────────────────────────────────────────────────────
_GREEN  = "#00CC44"    # normal / higher-MRD1 eye
_YELLOW = "#FFD000"    # ptotic / lower-MRD1 eye


def _eye_colors(r_metrics: EyeMetrics, l_metrics: EyeMetrics):
    """Return (right_color, left_color) — ptotic eye gets yellow."""
    if r_metrics.mrd1_mm <= l_metrics.mrd1_mm:
        return _YELLOW, _GREEN
    return _GREEN, _YELLOW


# ── Normalize a curve to mm-centered on pupil ─────────────────────────────────
def _normalize(curve: np.ndarray, eye) -> np.ndarray:
    c = curve.astype(float).copy()
    c[:, 0] = (c[:, 0] - eye.pupil_center[0]) / eye.px_per_mm
    c[:, 1] = (eye.pupil_center[1] - c[:, 1]) / eye.px_per_mm   # + = up
    return c


# ── Eye region crop (color) ────────────────────────────────────────────────────
def _eye_crop(image_rgb: np.ndarray, eye, pad_x_r=2.8, pad_top_r=2.2, pad_bot_r=1.5):
    """Return (crop_rgb, x1, y1) so callers can re-use offsets."""
    h, w = image_rgb.shape[:2]
    cx, cy = eye.pupil_center
    r = eye.iris_radius_px
    x1 = max(0, int(cx - pad_x_r * r))
    x2 = min(w, int(cx + pad_x_r * r))
    y1 = max(0, int(cy - pad_top_r * r))
    y2 = min(h, int(cy + pad_bot_r * r))
    crop_rgb = image_rgb[y1:y2, x1:x2]
    return crop_rgb, x1, y1


# ── Paper-style 2×2 figure ────────────────────────────────────────────────────
def build_paper_figure(
    image_rgb: np.ndarray,
    r_curve: np.ndarray,
    l_curve: np.ndarray,
    r_eye, l_eye,
    r_metrics: EyeMetrics,
    l_metrics: EyeMetrics,
    r_ctrl: np.ndarray = None,   # control points (optional, shown on ptotic side)
    l_ctrl: np.ndarray = None,
) -> io.BytesIO:
    """
    2 columns × 2 rows matching the paper figures:
      Top row   : normalized lid curves on black background + crosshairs
      Bottom row: color eye crop with curve, crosshairs, iris circle
    """
    r_color, l_color = _eye_colors(r_metrics, l_metrics)
    r_is_ptotic = r_color == _YELLOW

    fig = plt.figure(figsize=(12, 8), facecolor="#111111")
    gs  = fig.add_gridspec(2, 2, hspace=0.08, wspace=0.06,
                           top=0.93, bottom=0.06, left=0.05, right=0.97,
                           height_ratios=[1, 1.4])

    eyes   = [(r_eye, r_curve, r_color, r_ctrl, "Right Eye",  r_metrics, r_is_ptotic),
              (l_eye, l_curve, l_color, l_ctrl, "Left Eye",  l_metrics, not r_is_ptotic)]

    for col, (eye, curve, color, ctrl, label, metrics, is_ptotic) in enumerate(eyes):
        norm = _normalize(curve, eye)

        # ── Row 0: normalized curve on black ─────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])
        ax0.set_facecolor("black")

        ax0.plot(norm[:, 0], norm[:, 1], color=color, lw=2.0, zorder=3)

        # Control point polygon (shown on ptotic eye, like Fig 2 in paper)
        if is_ptotic and ctrl is not None:
            nc = _normalize(ctrl, eye)
            ax0.plot(nc[:, 0], nc[:, 1], color="red", lw=0.8,
                     marker="o", markersize=4, markerfacecolor="none",
                     markeredgecolor="red", zorder=4, alpha=0.8)

        ax0.axvline(0, color="white", lw=0.7, ls="--", alpha=0.5, zorder=2)
        ax0.axhline(0, color="white", lw=0.7, ls="--", alpha=0.5, zorder=2)
        # Light reflex marker
        ax0.plot(0, 0, marker=(4,1,45), color="white",      ms=10, zorder=5,
                 markeredgewidth=1.5, markerfacecolor="white")
        ax0.plot(0, 0, "o",             color="dodgerblue", ms=4,  zorder=6)

        # MRD1 annotation
        mrd1_mm = metrics.mrd1_mm
        ax0.annotate(f"MRD1 = {mrd1_mm:.1f} mm",
                     xy=(0, 0), xytext=(0.05, 0.12),
                     textcoords="axes fraction",
                     color="white", fontsize=8, zorder=6)

        ax0.set_title(label, color="white", fontsize=10, pad=4)
        ax0.tick_params(colors="white", labelsize=7)
        ax0.xaxis.label.set_color("white")
        ax0.yaxis.label.set_color("white")
        ax0.set_xlabel("mm from pupil center", fontsize=8, color="white")
        ax0.set_ylabel("mm above pupil", fontsize=8, color="white")
        for sp in ax0.spines.values():
            sp.set_edgecolor("#555555")
        ax0.grid(True, alpha=0.12, color="white")

        # ── Row 1: eye photo crop ─────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])
        ax1.set_facecolor("black")

        crop_rgb, x1, y1 = _eye_crop(image_rgb, eye)
        ch, cw = crop_rgb.shape[:2]
        ax1.imshow(crop_rgb, origin="upper",
                   extent=[x1, x1 + cw, y1 + ch, y1],
                   aspect="auto")

        # Curve
        ax1.plot(curve[:, 0], curve[:, 1], color=color, lw=2.2, zorder=4)

        # Control points (ptotic eye only)
        if is_ptotic and ctrl is not None:
            ax1.scatter(ctrl[:, 0], ctrl[:, 1],
                        color="red", s=18, zorder=5, marker="o")

        cx, cy = eye.pupil_center
        r = eye.iris_radius_px

        # Crosshairs
        ax1.axvline(cx, color="white", lw=0.8, ls="--", alpha=0.6, zorder=3)
        ax1.axhline(cy, color="white", lw=0.8, ls="--", alpha=0.6, zorder=3)

        # Iris circle
        circ = mpatches.Circle((cx, cy), r, color="dodgerblue",
                                fill=False, lw=1.5, zorder=5)
        ax1.add_patch(circ)

        # Light reflex marker — bright star cross
        ax1.plot(cx, cy, marker=(4, 1, 45), color="white",   ms=10, zorder=7,
                 markeredgewidth=1.5, markerfacecolor="white")
        ax1.plot(cx, cy, "o",                color="dodgerblue", ms=4,  zorder=8)

        # MRD1 vertical line segment
        lid_y_at_cx = _lid_y_at_x(curve, cx)
        ax1.annotate("", xy=(cx + r * 0.3, lid_y_at_cx),
                     xytext=(cx + r * 0.3, cy),
                     arrowprops=dict(arrowstyle="<->", color="white", lw=1.0))
        ax1.text(cx + r * 0.4, (lid_y_at_cx + cy) / 2,
                 f"MRD1\n{mrd1_mm:.1f}mm",
                 color="white", fontsize=7, va="center", zorder=7)

        ax1.set_xlim(x1, x1 + cw)
        ax1.set_ylim(y1 + ch, y1)
        ax1.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=130, facecolor="#111111")
    plt.close(fig)
    buf.seek(0)
    return buf


def _lid_y_at_x(curve: np.ndarray, x: float) -> float:
    """Return y on the curve closest to x."""
    idx = np.argmin(np.abs(curve[:, 0] - x))
    return float(curve[idx, 1])


# ── Symmetry plot (paper style) ───────────────────────────────────────────────
def build_symmetry_plot(symmetry: SymmetryResult) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#111111")
    ax.set_facecolor("black")

    ref  = symmetry.ref_curve[np.argsort(symmetry.ref_curve[:, 0])]
    flip = symmetry.flipped_curve[np.argsort(symmetry.flipped_curve[:, 0])]

    ax.plot(ref[:, 0],  ref[:, 1],  color=_GREEN,  lw=2,
            label="Higher-MRD1 eye (reference)")
    ax.plot(flip[:, 0], flip[:, 1], color=_YELLOW, lw=2, ls="--",
            label=f"Lower-MRD1 eye (mirrored) — {symmetry.lower_mrd1_side}")

    x_min = max(ref[:, 0].min(), flip[:, 0].min())
    x_max = min(ref[:, 0].max(), flip[:, 0].max())
    if x_max > x_min:
        xc = np.linspace(x_min, x_max, 500)
        def si(c, x):
            _, u = np.unique(c[:, 0], return_index=True)
            c = c[u]
            return interp1d(c[:, 0], c[:, 1], bounds_error=False,
                            fill_value="extrapolate")(x) if len(c) >= 2 else np.zeros_like(x)
        yr = si(ref, xc); yf = si(flip, xc)
        ax.fill_between(xc, np.minimum(yr, yf), np.maximum(yr, yf),
                        alpha=0.3, color="gray", label="Discrepancy")

    ax.axvline(0, color="white", lw=0.8, ls=":", alpha=0.5)
    ax.axhline(0, color="white", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel("Horizontal distance from pupil center (mm)", color="white", fontsize=9)
    ax.set_ylabel("Lid height above pupil (mm)", color="white", fontsize=9)
    ax.set_title(f"Lid Curve Symmetry: {symmetry.symmetry_pct:.1f}%",
                 color="white", fontsize=11)
    ax.legend(fontsize=8, facecolor="#222222", labelcolor="white",
              framealpha=0.8, edgecolor="#555555")
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#555555")
    ax.grid(True, alpha=0.15, color="white")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=120, facecolor="#111111")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Simple color overlay (for PDF page 2) ─────────────────────────────────────
def build_contour_overlay(image_rgb, right_curve, left_curve,
                          right_eye, left_eye,
                          r_metrics=None, l_metrics=None) -> np.ndarray:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    rc  = (0, 200, 60)   if (r_metrics is None or r_metrics.mrd1_mm > (l_metrics.mrd1_mm if l_metrics else 999)) else (0, 200, 220)
    lc  = (0, 200, 220)  if (r_metrics is None or r_metrics.mrd1_mm > (l_metrics.mrd1_mm if l_metrics else 999)) else (0, 200, 60)

    for curve, color in [(right_curve, (0, 200, 60)), (left_curve, (0, 160, 255))]:
        pts = curve.astype(np.int32)
        for i in range(len(pts) - 1):
            cv2.line(bgr, tuple(pts[i]), tuple(pts[i + 1]), color, 2)

    for eye in [right_eye, left_eye]:
        cx, cy = int(eye.pupil_center[0]), int(eye.pupil_center[1])
        cv2.circle(bgr, (cx, cy), int(eye.iris_radius_px), (200, 200, 0), 1)
        cv2.circle(bgr, (cx, cy), 3, (0, 220, 255), -1)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── PDF generation ────────────────────────────────────────────────────────────
def _np_to_rl(img_rgb, max_w_cm, max_h_cm):
    pil = Image.fromarray(img_rgb.astype(np.uint8))
    buf = io.BytesIO(); pil.save(buf, "JPEG", quality=90); buf.seek(0)
    ow, oh = pil.size
    scale = min(max_w_cm * cm / ow, max_h_cm * cm / oh)
    return RLImage(buf, width=ow * scale, height=oh * scale)


def _buf_to_rl(buf: io.BytesIO, max_w_cm, max_h_cm):
    img = Image.open(buf)
    ow, oh = img.size
    scale = min(max_w_cm * cm / ow, max_h_cm * cm / oh)
    buf.seek(0)
    return RLImage(buf, width=ow * scale, height=oh * scale)


def generate_pdf(
    image_rgb, right_curve, left_curve, right_eye, left_eye,
    right_metrics: EyeMetrics, left_metrics: EyeMetrics,
    symmetry: SymmetryResult,
    confidence: float, low_confidence: bool, warning_msg: str,
    r_ctrl=None, l_ctrl=None,
) -> io.BytesIO:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=1.5*cm, leftMargin=1.5*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    title_s  = ParagraphStyle("T", parent=styles["Title"], fontSize=15, spaceAfter=4)
    head_s   = ParagraphStyle("H", parent=styles["Heading2"], fontSize=11, spaceAfter=3)
    norm_s   = styles["Normal"]
    warn_s   = ParagraphStyle("W", parent=norm_s, textColor=colors.orangered, fontSize=9)
    ref_s    = ParagraphStyle("R", parent=norm_s, fontSize=7, textColor=colors.grey)

    def f(v, d=2):
        if v is None or (isinstance(v, float) and v != v): return "N/A"
        return f"{v:.{d}f}"

    story = []
    story.append(Paragraph("Ptosis Eyelid Analysis Report", title_s))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(f"Landmark confidence: {confidence:.0%}", norm_s))
    if low_confidence and warning_msg:
        story.append(Paragraph(f"⚠ {warning_msg}", warn_s))
    story.append(Spacer(1, 0.3*cm))

    # Paper-style figure
    story.append(Paragraph("1. Eyelid Contour Analysis (paper-style)", head_s))
    pfig = build_paper_figure(image_rgb, right_curve, left_curve,
                              right_eye, left_eye,
                              right_metrics, left_metrics,
                              r_ctrl, l_ctrl)
    story.append(_buf_to_rl(pfig, 17, 11))
    story.append(Spacer(1, 0.3*cm))

    # Symmetry
    story.append(Paragraph("2. Symmetry Curve Overlap", head_s))
    story.append(_buf_to_rl(build_symmetry_plot(symmetry), 16, 8))
    story.append(Spacer(1, 0.3*cm))

    # Metrics table
    story.append(Paragraph("3. Clinical Metrics", head_s))
    lower = symmetry.lower_mrd1_side.capitalize()
    tdata = [
        ["Metric", "Right Eye", "Left Eye", "Normal Range"],
        ["MRD1 (mm)",         f(right_metrics.mrd1_mm),        f(left_metrics.mrd1_mm),        "3.0 – 4.5"],
        ["PHUL (mm)",         f(right_metrics.phul_mm),         f(left_metrics.phul_mm),         "3.5 – 5.0"],
        ["PHUL position",     right_metrics.phul_position.capitalize(),
                              left_metrics.phul_position.capitalize(),     "Temporal"],
        ["PHUL offset (mm)",  f(right_metrics.phul_offset_mm),  f(left_metrics.phul_offset_mm),  ""],
        ["T/N ratio",         f(right_metrics.tn_ratio),        f(left_metrics.tn_ratio),        "~1.0"],
        ["Symmetry %",        "",  f(symmetry.symmetry_pct, 1) + "%",  "> 85%"],
        ["MRD1 diff (mm)",    f(abs(right_metrics.mrd1_mm - left_metrics.mrd1_mm)), "", "< 1.0"],
        ["Calib (px/mm)",     f(right_metrics.px_per_mm, 1),   f(left_metrics.px_per_mm, 1),    ""],
    ]
    t = Table(tdata, colWidths=[4.5*cm, 3.5*cm, 3.5*cm, 3*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR",     (0,0),(-1,0),  colors.white),
        ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("ALIGN",         (1,0),(-1,-1), "CENTER"),
        ("ALIGN",         (0,0),(0,-1),  "LEFT"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.grey),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(
        "Symmetry method: Aytogan H, Ayıntap E. BMC Ophthalmology 2021;21:360. "
        "DOI:10.1186/s12886-021-02208-7. Iris calibration: 11.7 mm standard diameter.", ref_s))

    doc.build(story)
    buf.seek(0)
    return buf

"""
Ptosis Eyelid Analyzer — Streamlit UI (two-stage: adjust → calculate)
Click-on-image point repositioning.
"""

import os
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from pipeline.landmark import detect_landmarks
from pipeline.contour  import extract_upper_lid_contour, _smooth_and_bezier
from pipeline.metrics  import compute_all_metrics
from pipeline.report   import generate_pdf, build_paper_figure, build_symmetry_plot

st.set_page_config(page_title="Ptosis Eyelid Analyzer", page_icon="👁", layout="wide")
st.title("👁 Ptosis Eyelid Analyzer")
st.markdown("---")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _eye_crop_bgr(image_bgr, eye, pad_x=2.8, pad_t=2.4, pad_b=1.8):
    h, w = image_bgr.shape[:2]
    cx, cy = eye.pupil_center
    r = eye.iris_radius_px
    x1 = max(0, int(cx - pad_x * r));  x2 = min(w, int(cx + pad_x * r))
    y1 = max(0, int(cy - pad_t * r));  y2 = min(h, int(cy + pad_b * r))
    return image_bgr[y1:y2, x1:x2].copy(), x1, y1


def render_eye_preview(image_bgr, eye, pts_world,
                       color_rgb=(0, 220, 80), selected_idx=0,
                       target_width=430):
    """
    Returns a PIL image of the eye crop with:
      - iris circle (yellow)
      - Bezier curve (colored)
      - all control points numbered; selected one highlighted in orange
    """
    crop_bgr, ox, oy = _eye_crop_bgr(image_bgr, eye)
    h_orig, w_orig = crop_bgr.shape[:2]

    scale = target_width / max(w_orig, 1)
    dw, dh = int(w_orig * scale), int(h_orig * scale)

    disp = cv2.resize(crop_bgr, (dw, dh), interpolation=cv2.INTER_LINEAR)
    rgb  = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

    # Iris circle
    cx_d = int((float(eye.pupil_center[0]) - ox) * scale)
    cy_d = int((float(eye.pupil_center[1]) - oy) * scale)
    r_d  = int(float(eye.iris_radius_px)    * scale)
    cv2.circle(rgb, (cx_d, cy_d), r_d, (255, 255, 0), 2)

    # Bezier curve
    try:
        curve, _, _ = _smooth_and_bezier(pts_world)
        for i in range(len(curve) - 1):
            p1 = (int((curve[i,   0] - ox) * scale),
                  int((curve[i,   1] - oy) * scale))
            p2 = (int((curve[i+1, 0] - ox) * scale),
                  int((curve[i+1, 1] - oy) * scale))
            cv2.line(rgb, p1, p2, color_rgb, 2)
    except Exception:
        pass

    # Control points
    for i, (px, py) in enumerate(pts_world):
        lx = int((px - ox) * scale)
        ly = int((py - oy) * scale)
        is_sel = (i == selected_idx)
        dot_color = (255, 165, 0) if is_sel else (255, 50, 50)
        radius    = 9 if is_sel else 6
        cv2.circle(rgb, (lx, ly), radius, dot_color, -1)
        cv2.circle(rgb, (lx, ly), radius, (255, 255, 255), 1)
        cv2.putText(rgb, str(i + 1), (lx + 11, ly - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return Image.fromarray(rgb), scale, ox, oy


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown(
        "Both eyelids measured independently.\n\n"
        "**Iris calibration:** 11.7 mm\n\n"
        "**Reference:** Aytogan H, Ayıntap E. *BMC Ophthalmology* 2021;21:360"
    )
    st.markdown("---")
    st.markdown("**Normal ranges**\n- MRD1: 3.0–4.5 mm\n- PHUL: 3.5–5.0 mm\n- T/N: ~1.0\n- Symmetry: > 85%")

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload face photograph (JPG/PNG)",
                             type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Upload a frontal face photograph to begin.")
    st.stop()

fb = np.frombuffer(uploaded.read(), np.uint8)
image_bgr = cv2.imdecode(fb, cv2.IMREAD_COLOR)
if image_bgr is None:
    st.error("Could not decode image."); st.stop()
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

file_key = uploaded.name + str(uploaded.size)
if st.session_state.get("file_key") != file_key:
    for k in ["lm", "r_pts", "l_pts", "r_sel", "l_sel", "r_img_v", "l_img_v", "stage"]:
        st.session_state.pop(k, None)
    st.session_state["file_key"] = file_key

st.image(image_rgb, caption="Uploaded photo", width=420)
st.markdown("---")

# ── Stage 0: Detect ───────────────────────────────────────────────────────────
if "lm" not in st.session_state:
    if st.button("🔍 Detect Landmarks", type="primary", use_container_width=True):
        with st.spinner("Detecting landmarks…"):
            lm = detect_landmarks(image_bgr)
        if lm.right_eye is None and lm.left_eye is None:
            st.error("Could not detect either eye."); st.stop()
        if lm.right_eye is None or lm.left_eye is None:
            st.warning("One eye not detected — symmetry analysis unavailable."); st.stop()

        with st.spinner("Extracting initial lid contours…"):
            _, r_ctrl, _ = extract_upper_lid_contour(
                image_bgr,
                lm.right_eye.pupil_center, lm.right_eye.iris_radius_px,
                lm.right_eye.px_per_mm,
                lm.right_eye.medial_canthus, lm.right_eye.lateral_canthus,
                lm.right_eye.upper_lid_margin, side="right",
            )
            _, l_ctrl, _ = extract_upper_lid_contour(
                image_bgr,
                lm.left_eye.pupil_center, lm.left_eye.iris_radius_px,
                lm.left_eye.px_per_mm,
                lm.left_eye.medial_canthus, lm.left_eye.lateral_canthus,
                lm.left_eye.upper_lid_margin, side="left",
            )

        st.session_state.update({
            "lm":    lm,
            "r_pts": r_ctrl.tolist(),
            "l_pts": l_ctrl.tolist(),
            "r_sel": 0,
            "l_sel": 0,
            "stage": "adjust",
        })
        st.rerun()
    st.stop()

# ── Stage 1: Adjust lid control points ───────────────────────────────────────
lm    = st.session_state["lm"]
r_pts = np.array(st.session_state["r_pts"])
l_pts = np.array(st.session_state["l_pts"])

st.subheader("Step 1 — Adjust Lid Control Points")
st.markdown(
    "1. **Select** which point to move using the buttons below each eye.\n"
    "2. **Click** on the image where the point should go (on the lid margin).\n"
    "3. Repeat for all points, then press **Calculate Metrics**."
)

col_r, col_l = st.columns(2)

# ── Right eye ─────────────────────────────────────────────────────────────────
with col_r:
    st.markdown("**Right Eye** *(lateral → medial)*")

    r_sel = st.session_state.get("r_sel", 0)

    # Point selector buttons
    btn_cols = st.columns(len(r_pts))
    for i in range(len(r_pts)):
        with btn_cols[i]:
            label = f"**{i+1}**" if i == r_sel else str(i + 1)
            if st.button(label, key=f"r_btn_{i}",
                         use_container_width=True,
                         type="primary" if i == r_sel else "secondary"):
                st.session_state["r_sel"] = i
                st.rerun()

    # Preview + click
    pil_r, sc_r, ox_r, oy_r = render_eye_preview(
        image_bgr, lm.right_eye, r_pts,
        color_rgb=(0, 220, 80), selected_idx=r_sel,
    )
    r_img_v = st.session_state.get("r_img_v", 0)
    coords_r = streamlit_image_coordinates(pil_r, key=f"re_img_{r_img_v}")

    if coords_r is not None:
        new_r = r_pts.copy()
        new_r[r_sel, 0] = coords_r["x"] / sc_r + ox_r
        new_r[r_sel, 1] = coords_r["y"] / sc_r + oy_r
        st.session_state["r_pts"] = new_r.tolist()
        st.session_state["r_img_v"] = r_img_v + 1
        st.rerun()

# ── Left eye ──────────────────────────────────────────────────────────────────
with col_l:
    st.markdown("**Left Eye** *(lateral → medial)*")

    l_sel = st.session_state.get("l_sel", 0)

    btn_cols2 = st.columns(len(l_pts))
    for i in range(len(l_pts)):
        with btn_cols2[i]:
            label = f"**{i+1}**" if i == l_sel else str(i + 1)
            if st.button(label, key=f"l_btn_{i}",
                         use_container_width=True,
                         type="primary" if i == l_sel else "secondary"):
                st.session_state["l_sel"] = i
                st.rerun()

    pil_l, sc_l, ox_l, oy_l = render_eye_preview(
        image_bgr, lm.left_eye, l_pts,
        color_rgb=(0, 180, 255), selected_idx=l_sel,
    )
    l_img_v = st.session_state.get("l_img_v", 0)
    coords_l = streamlit_image_coordinates(pil_l, key=f"le_img_{l_img_v}")

    if coords_l is not None:
        new_l = l_pts.copy()
        new_l[l_sel, 0] = coords_l["x"] / sc_l + ox_l
        new_l[l_sel, 1] = coords_l["y"] / sc_l + oy_l
        st.session_state["l_pts"] = new_l.tolist()
        st.session_state["l_img_v"] = l_img_v + 1
        st.rerun()

# ── Action buttons ────────────────────────────────────────────────────────────
st.markdown("---")
btn_col1, btn_col2 = st.columns([3, 1])
with btn_col1:
    calc = st.button("✅ Calculate Metrics", type="primary", use_container_width=True)
with btn_col2:
    if st.button("🔄 Re-detect", use_container_width=True):
        for k in ["lm", "r_pts", "l_pts", "r_sel", "l_sel", "r_img_v", "l_img_v", "stage"]:
            st.session_state.pop(k, None)
        st.rerun()

if not calc:
    st.stop()

# ── Stage 2: Calculate ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 2 — Results")

with st.spinner("Computing metrics…"):
    r_curve, r_ctrl_final, _ = _smooth_and_bezier(r_pts)
    l_curve, l_ctrl_final, _ = _smooth_and_bezier(l_pts)
    r_m, l_m, sym = compute_all_metrics(r_curve, l_curve, lm.right_eye, lm.left_eye)

pfig_buf = build_paper_figure(
    image_rgb, r_curve, l_curve,
    lm.right_eye, lm.left_eye,
    r_m, l_m,
    r_ctrl=r_ctrl_final, l_ctrl=l_ctrl_final,
)
st.image(pfig_buf.read(), use_container_width=True)

st.markdown("---")
st.subheader("Clinical Metrics")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Right eye**")
    st.metric("MRD1", f"{r_m.mrd1_mm:.2f} mm",
              delta="⚠ low" if r_m.mrd1_mm < 2.0 else None, delta_color="inverse")
    st.metric("PHUL", f"{r_m.phul_mm:.2f} mm")
    st.metric("PHUL position", r_m.phul_position.capitalize())
    st.metric("T/N ratio", f"{r_m.tn_ratio:.2f}")
with c2:
    st.markdown("**Left eye**")
    st.metric("MRD1", f"{l_m.mrd1_mm:.2f} mm",
              delta="⚠ low" if l_m.mrd1_mm < 2.0 else None, delta_color="inverse")
    st.metric("PHUL", f"{l_m.phul_mm:.2f} mm")
    st.metric("PHUL position", l_m.phul_position.capitalize())
    st.metric("T/N ratio", f"{l_m.tn_ratio:.2f}")
with c3:
    st.markdown("**Symmetry**")
    st.metric("Symmetry %", f"{sym.symmetry_pct:.1f}%",
              delta="⚠ low" if sym.symmetry_pct < 85 else None, delta_color="inverse")
    st.metric("MRD1 diff", f"{abs(r_m.mrd1_mm - l_m.mrd1_mm):.2f} mm")
    st.metric("Confidence", f"{lm.confidence:.0%}")
    st.caption(f"More ptotic: **{sym.lower_mrd1_side.capitalize()}** eye")

st.markdown("---")
st.subheader("Symmetry Curve Overlap")
sym_buf = build_symmetry_plot(sym)
st.image(sym_buf.read(), use_container_width=True)

st.markdown("---")
st.subheader("Interpretation")
notes = []
for side, m in [("Right", r_m), ("Left", l_m)]:
    if m.mrd1_mm < 2.0:
        notes.append(f"**{side} MRD1 = {m.mrd1_mm:.2f} mm** — below 2 mm (ptosis)")
    elif m.mrd1_mm < 3.0:
        notes.append(f"**{side} MRD1 = {m.mrd1_mm:.2f} mm** — borderline (normal ≥ 3 mm)")
if abs(r_m.mrd1_mm - l_m.mrd1_mm) > 1.0:
    notes.append(f"MRD1 asymmetry = {abs(r_m.mrd1_mm - l_m.mrd1_mm):.2f} mm (clinically significant > 1 mm)")
if sym.symmetry_pct < 85:
    notes.append(f"Symmetry {sym.symmetry_pct:.1f}% — below 85% threshold")
if not notes:
    notes.append("All measured values appear within normal range.")
for n in notes:
    st.markdown(f"- {n}")

st.markdown("---")
with st.spinner("Generating PDF…"):
    pdf = generate_pdf(
        image_rgb, r_curve, l_curve, lm.right_eye, lm.left_eye,
        r_m, l_m, sym,
        lm.confidence, lm.low_confidence, lm.warning_msg,
        r_ctrl=r_ctrl_final, l_ctrl=l_ctrl_final,
    )
st.download_button("📄 Download PDF Report", data=pdf.getvalue(),
                   file_name="ptosis_analysis.pdf", mime="application/pdf",
                   use_container_width=True)

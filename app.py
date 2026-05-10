import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image as PILImage
from streamlit_image_coordinates import streamlit_image_coordinates

from processor import analyze_rois

st.set_page_config(page_title="WB 条带自动定量", page_icon="🧬", layout="wide")
st.title("🧬 Western Blot 条带灰度值自动定量")

mode = st.radio(
    "分析模式",
    ["单膜分析", "双膜对比（目的蛋白/内参 分开跑）", "单膜对比（目的蛋白+内参 同一张膜）"],
    horizontal=True,
)
st.divider()

with st.sidebar:
    st.header("参数设置")
    radius = st.slider("背景去除半径（rolling ball）", 10, 150, 50, 5)
    st.divider()
    st.markdown(
        "**画框说明**\n\n"
        "① 点击条带**左上角**\n\n"
        "② 点击条带**右下角** → 框生成\n\n"
        "重复以上步骤画多个框\n\n"
        "点「🗑 清空」重新开始"
    )


# ── helpers ───────────────────────────────────────────────────────────────────
def load_image(uploaded) -> np.ndarray | None:
    if uploaded is None:
        return None
    data = np.frombuffer(uploaded.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def box_selector(img_bgr: np.ndarray, key: str,
                 label: str = "", color: tuple = (0, 220, 80)):
    """Two-click-per-box ROI selector using streamlit_image_coordinates.

    streamlit_image_coordinates with use_column_width=True returns coordinates
    in the ORIGINAL image pixel space (the library scales internally via
    img.naturalWidth / img.offsetWidth).  No manual scaling is applied.
    """
    ck_boxes = f"{key}_boxes"
    ck_pt1   = f"{key}_pt1"
    ck_prev  = f"{key}_prev"
    for k, d in [(ck_boxes, []), (ck_pt1, None), (ck_prev, None)]:
        if k not in st.session_state:
            st.session_state[k] = d

    boxes = st.session_state[ck_boxes]
    pt1   = st.session_state[ck_pt1]

    if label:
        st.markdown(f"**{label}**")

    col_info, col_btn = st.columns([5, 1])
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑 清空", key=f"{key}_clr"):
            st.session_state[ck_boxes] = []
            st.session_state[ck_pt1]   = None
            st.session_state[ck_prev]  = None
            st.rerun()
    with col_info:
        if pt1 is None:
            if boxes:
                st.success(f"✅ 已画 {len(boxes)} 个框，继续点左上角加框")
            else:
                st.info("① 点击条带**左上角**")
        else:
            st.warning("② 点击条带**右下角**（左上角已记录 ✓）")

    # Draw overlay on the original-size image
    h, w = img_bgr.shape[:2]
    s = max(1.0, w / 800)                      # scale drawing elements
    lw = max(2, int(s * 2))
    fs = max(0.5, s * 0.55)
    overlay = img_bgr.copy()
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, lw)
        cv2.putText(overlay, f"L{i+1}", (x0 + int(4*s), y0 + int(22*s)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, max(1, lw-1))
    if pt1 is not None:
        ms = max(24, int(s * 28))
        cv2.drawMarker(overlay, pt1, (0, 200, 255), cv2.MARKER_CROSS, ms, lw)

    # Pass ORIGINAL image — library maps click back to original pixel coords
    img_pil = PILImage.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    clicked = streamlit_image_coordinates(img_pil, key=f"{key}_canvas",
                                          use_column_width=True)

    if clicked and clicked != st.session_state[ck_prev]:
        cx = max(0, min(w - 1, int(clicked["x"])))
        cy = max(0, min(h - 1, int(clicked["y"])))
        if pt1 is None:
            st.session_state[ck_pt1]  = (cx, cy)
            st.session_state[ck_prev] = clicked
        else:
            x0, x1 = sorted([pt1[0], cx])
            y0, y1 = sorted([pt1[1], cy])
            if x1 - x0 > 4 and y1 - y0 > 4:
                st.session_state[ck_boxes].append((x0, y0, x1, y1))
            st.session_state[ck_pt1]  = None
            st.session_state[ck_prev] = clicked
        st.rerun()

    return sorted(boxes, key=lambda r: r[0])


def show_annotated(img_bgr, roi_groups):
    ann = img_bgr.copy()
    for rois, color, prefix in roi_groups:
        for i, (x0, y0, x1, y1) in enumerate(rois):
            cv2.rectangle(ann, (x0, y0), (x1, y1), color, 2)
            cv2.putText(ann, f"{prefix}{i+1}", (x0+4, y0+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**原始图片**")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with c2:
        st.markdown("**框选区域**")
        st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)


def ratio_table(df_t, df_r):
    for col in ("Lane", "IntDen"):
        if col not in df_t.columns or col not in df_r.columns:
            st.error(f"结果表缺少列 '{col}'")
            return None
    t = df_t[["Lane","IntDen"]].rename(columns={"IntDen":"Target_IntDen"})
    r = df_r[["Lane","IntDen"]].rename(columns={"IntDen":"Ref_IntDen"})
    m = t.merge(r, on="Lane", how="inner")
    if m.empty:
        st.error("目的蛋白与内参泳道数量不一致")
        return None
    m["Ratio"] = (m["Target_IntDen"] / m["Ref_IntDen"]).round(4)
    m["Normalized_Ratio"] = (m["Ratio"] / m["Ratio"].iloc[0]).round(4)
    return m


def excel_download(sheets, base):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    buf.seek(0)
    st.download_button("⬇️ 下载 Excel", data=buf,
        file_name=base + "_WB.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def show_quant(df, uploaded, extra_sheets=None):
    cols = [c for c in ["Lane","Band","Area","Mean","Min","Max","IntDen","RawIntDen"]
            if c in df.columns]
    st.subheader(f"定量结果（{len(df)} 个条带）")
    st.dataframe(df[cols], use_container_width=True, hide_index=True)
    sheets = {"定量结果": df[cols]}
    if extra_sheets:
        sheets.update(extra_sheets)
    base = uploaded.name.rsplit(".",1)[0] if uploaded else "wb"
    excel_download(sheets, base)
    if len(df) > 1:
        st.bar_chart(df.groupby("Lane")["IntDen"].sum())


# ══════════════════════════════════════════════════════════════════════════════
if mode == "单膜分析":
    uploaded = st.file_uploader("上传 WB 图片",
                                type=["jpg","jpeg","png","tif","tiff","bmp"])
    img = load_image(uploaded)
    if img is None:
        st.info("请上传图片后开始分析。")
        st.stop()

    rois = box_selector(img, key="s1")
    if not rois:
        st.stop()

    with st.spinner("分析中…"):
        _, df, _ = analyze_rois(img, rois, radius=radius)
    st.divider()
    show_annotated(img, [(rois, (0,220,80), "L")])
    st.divider()
    show_quant(df, uploaded)


elif mode == "双膜对比（目的蛋白/内参 分开跑）":
    c1, c2 = st.columns(2)
    with c1:
        up_t = st.file_uploader("🎯 目的蛋白图片",
                                type=["jpg","jpeg","png","tif","tiff","bmp"], key="t2")
    with c2:
        up_r = st.file_uploader("⚖️ 内参图片",
                                type=["jpg","jpeg","png","tif","tiff","bmp"], key="r2")
    img_t, img_r = load_image(up_t), load_image(up_r)
    if img_t is None or img_r is None:
        st.info("请分别上传两张图片。")
        st.stop()

    tab_t, tab_r = st.tabs(["🎯 目的蛋白", "⚖️ 内参"])
    with tab_t:
        rois_t = box_selector(img_t, key="t2s", label="目的蛋白：点击画框",
                              color=(0,220,80))
    with tab_r:
        rois_r = box_selector(img_r, key="r2s", label="内参：点击画框",
                              color=(0,170,255))
    if not rois_t or not rois_r:
        st.stop()

    with st.spinner("分析中…"):
        _, df_t, _ = analyze_rois(img_t, rois_t, radius=radius)
        _, df_r, _ = analyze_rois(img_r, rois_r, radius=radius)

    st.divider()
    st.subheader("目的蛋白")
    show_annotated(img_t, [(rois_t, (0,220,80), "T")])
    st.subheader("内参")
    show_annotated(img_r, [(rois_r, (0,170,255), "R")])
    st.divider()

    merged = ratio_table(df_t, df_r)
    if merged is not None:
        st.subheader("对比结果")
        st.dataframe(merged, use_container_width=True, hide_index=True)
        st.caption("Normalized_Ratio = 以 Lane 1 为 1 归一化")
        st.bar_chart(merged.set_index("Lane")["Normalized_Ratio"])
        cols = ["Lane","Band","Area","Mean","IntDen","RawIntDen"]
        base = up_t.name.rsplit(".",1)[0] if up_t else "wb"
        excel_download({
            "目的蛋白": df_t[[c for c in cols if c in df_t.columns]],
            "内参":     df_r[[c for c in cols if c in df_r.columns]],
            "对比结果": merged,
        }, base)


else:
    uploaded = st.file_uploader("上传 WB 图片（含两条带）",
                                type=["jpg","jpeg","png","tif","tiff","bmp"])
    img = load_image(uploaded)
    if img is None:
        st.info("请上传图片后开始分析。")
        st.stop()

    tab_tgt, tab_ref = st.tabs(["🎯 目的蛋白", "⚖️ 内参"])
    with tab_tgt:
        rois_t = box_selector(img, key="s3t", label="目的蛋白条带",
                              color=(0,220,80))
    with tab_ref:
        rois_r = box_selector(img, key="s3r", label="内参条带",
                              color=(0,170,255))
    if not rois_t or not rois_r:
        st.stop()

    with st.spinner("分析中…"):
        _, df_t, _ = analyze_rois(img, rois_t, radius=radius)
        _, df_r, _ = analyze_rois(img, rois_r, radius=radius)

    st.divider()
    show_annotated(img, [(rois_t,(0,220,80),"T"), (rois_r,(0,170,255),"R")])
    st.divider()

    cols = ["Lane","Band","Area","Mean","Min","Max","IntDen","RawIntDen"]
    merged = ratio_table(df_t, df_r)
    if merged is not None:
        st.subheader("对比结果（目的蛋白 / 内参）")
        st.dataframe(merged, use_container_width=True, hide_index=True)
        st.caption("Normalized_Ratio = 以 Lane 1 为 1 归一化")
        st.bar_chart(merged.set_index("Lane")["Normalized_Ratio"])
        base = uploaded.name.rsplit(".",1)[0] if uploaded else "wb"
        excel_download({
            "目的蛋白": df_t[[c for c in cols if c in df_t.columns]],
            "内参":     df_r[[c for c in cols if c in df_r.columns]],
            "对比结果": merged,
        }, base)

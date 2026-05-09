import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image as PILImage

from processor import analyze_rois

st.set_page_config(page_title="WB 条带自动定量", page_icon="🧬", layout="wide")
st.title("🧬 Western Blot 条带灰度值自动定量")

# ── 分析模式 ──────────────────────────────────────────────────────────────────
mode = st.radio(
    "分析模式",
    ["单膜分析", "双膜对比（目的蛋白/内参 分开跑）", "单膜对比（目的蛋白+内参 同一张膜）"],
    horizontal=True,
)
st.divider()

# ── 侧边栏 ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("参数设置")
    radius = st.slider("背景去除半径（rolling ball）", 10, 150, 50, 5)
    st.divider()
    st.markdown(
        "**使用说明**\n\n"
        "1. 上传图片\n"
        "2. 用滑块拖出绿色矩形框，框住所有条带\n"
        "3. 填写泳道数量，自动等分\n"
        "4. 多个蛋白各自单独框选\n\n"
        "滑块单位为像素，实时预览图中绿框"
    )


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def load_image(uploaded) -> np.ndarray | None:
    if uploaded is None:
        return None
    data = np.frombuffer(uploaded.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def slider_roi(img_bgr: np.ndarray, key: str, label: str = "",
               color: tuple = (0, 220, 80)) -> list[tuple[int, int, int, int]]:
    """Slider-based ROI selector.

    User drags 4 sliders to define a bounding rectangle covering all lanes,
    sets the number of lanes, and the box is split into equal columns.
    Returns list of (x0, y0, x1, y1) per lane, sorted left-to-right.
    """
    h, w = img_bgr.shape[:2]

    if label:
        st.markdown(f"**{label}**")

    # ── Sliders ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x0 = st.slider("左边界", 0, w - 1, 0,           key=f"{key}_x0")
    with c2:
        x1 = st.slider("右边界", 0, w - 1, w - 1,       key=f"{key}_x1")
    with c3:
        y0 = st.slider("上边界", 0, h - 1, h // 4,      key=f"{key}_y0")
    with c4:
        y1 = st.slider("下边界", 0, h - 1, h * 3 // 4,  key=f"{key}_y1")

    c5, c6 = st.columns(2)
    with c5:
        n = st.number_input("泳道数量", 1, 30, 6, 1, key=f"{key}_n")
    with c6:
        gap_pct = st.slider("泳道间隙 %", 0, 30, 10, 1, key=f"{key}_gap")

    # Normalise: ensure x0 < x1, y0 < y1
    lx, rx = (x0, x1) if x0 < x1 else (x1, x0)
    ty, by = (y0, y1) if y0 < y1 else (y1, y0)
    if rx - lx < 4 or by - ty < 4:
        st.warning("请调整滑块，使左边界 < 右边界，且上边界 < 下边界")
        return []

    # ── Build lane boxes ───────────────────────────────────────────────────
    total = rx - lx
    sp    = total / n
    half  = max(2, int(sp * (1 - gap_pct / 100) / 2))
    rois  = []
    for i in range(int(n)):
        cx  = int(lx + i * sp + sp / 2)
        bx0 = max(0, cx - half)
        bx1 = min(w, cx + half)
        rois.append((bx0, ty, bx1, by))

    # ── Overlay preview ────────────────────────────────────────────────────
    overlay = img_bgr.copy()
    for i, (bx0, by0, bx1, by1) in enumerate(rois):
        cv2.rectangle(overlay, (bx0, by0), (bx1, by1), color, 2)
        cv2.putText(overlay, f"L{i + 1}", (bx0 + 4, by0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    box_w = rois[0][2] - rois[0][0] if rois else 0
    box_h = by - ty
    st.caption(f"每框 {box_w} × {box_h} px  ·  共 {int(n)} 条泳道  ·  间隙 {gap_pct}%")
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

    return rois


def show_annotated(img_bgr: np.ndarray,
                   roi_groups: list[tuple[list, tuple, str]]):
    ann = img_bgr.copy()
    for rois, color, prefix in roi_groups:
        for i, (x0, y0, x1, y1) in enumerate(rois):
            cv2.rectangle(ann, (x0, y0), (x1, y1), color, 2)
            cv2.putText(ann, f"{prefix}{i + 1}", (x0 + 4, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**原始图片**")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with c2:
        st.markdown("**框选区域**")
        st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)


def ratio_table(df_t: pd.DataFrame, df_r: pd.DataFrame) -> pd.DataFrame | None:
    for col in ("Lane", "IntDen"):
        if col not in df_t.columns or col not in df_r.columns:
            st.error(f"结果表缺少列 '{col}'，请检查框选区域是否正确。")
            return None
    t = df_t[["Lane", "IntDen"]].rename(columns={"IntDen": "Target_IntDen"})
    r = df_r[["Lane", "IntDen"]].rename(columns={"IntDen": "Ref_IntDen"})
    m = t.merge(r, on="Lane", how="inner")
    if m.empty:
        st.error("目的蛋白与内参泳道数量不一致，无法配对。")
        return None
    m["Ratio"] = (m["Target_IntDen"] / m["Ref_IntDen"]).round(4)
    m["Normalized_Ratio"] = (m["Ratio"] / m["Ratio"].iloc[0]).round(4)
    return m


def excel_download(sheets: dict, base: str):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    buf.seek(0)
    st.download_button("⬇️  下载 Excel 结果", data=buf,
        file_name=base + "_WB_quantification.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def show_quant(df, uploaded, extra_sheets=None):
    cols = [c for c in ["Lane", "Band", "Area", "Mean", "Min", "Max", "IntDen", "RawIntDen"]
            if c in df.columns]
    st.subheader(f"定量结果（共 {len(df)} 个框）")
    st.dataframe(df[cols], use_container_width=True, hide_index=True)
    sheets = {"定量结果": df[cols]}
    if extra_sheets:
        sheets.update(extra_sheets)
    base = uploaded.name.rsplit(".", 1)[0] if uploaded else "wb"
    excel_download(sheets, base)
    if len(df) > 1:
        st.subheader("IntDen 柱状图")
        st.bar_chart(df.groupby("Lane")["IntDen"].sum())


# ══════════════════════════════════════════════════════════════════════════════
# 模式 1：单膜分析
# ══════════════════════════════════════════════════════════════════════════════
if mode == "单膜分析":
    uploaded = st.file_uploader("上传 WB 图片",
                                type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])
    img = load_image(uploaded)
    if img is None:
        st.info("请上传图片后开始分析。")
        st.stop()

    rois = slider_roi(img, key="s1", label="拖动滑块框住条带区域，设置泳道数")
    if not rois:
        st.stop()

    if st.button("▶️ 开始分析", key="s1_run", type="primary"):
        with st.spinner("分析中…"):
            _, df, _ = analyze_rois(img, rois, radius=radius)
        st.divider()
        show_annotated(img, [(rois, (0, 220, 80), "L")])
        st.divider()
        show_quant(df, uploaded)


# ══════════════════════════════════════════════════════════════════════════════
# 模式 2：双膜对比
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "双膜对比（目的蛋白/内参 分开跑）":
    c1, c2 = st.columns(2)
    with c1:
        up_t = st.file_uploader("🎯 目的蛋白图片",
                                type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"], key="t2")
    with c2:
        up_r = st.file_uploader("⚖️ 内参图片",
                                type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"], key="r2")

    img_t = load_image(up_t)
    img_r = load_image(up_r)
    if img_t is None or img_r is None:
        st.info("请分别上传目的蛋白图片和内参图片。")
        st.stop()

    tab_t, tab_r = st.tabs(["🎯 目的蛋白", "⚖️ 内参"])
    with tab_t:
        rois_t = slider_roi(img_t, key="t2s", label="目的蛋白：框住条带区域",
                            color=(0, 220, 80))
    with tab_r:
        rois_r = slider_roi(img_r, key="r2s", label="内参：框住条带区域",
                            color=(0, 170, 255))

    if not rois_t or not rois_r:
        st.stop()

    if st.button("▶️ 开始分析", key="t2_run", type="primary"):
        with st.spinner("分析中…"):
            _, df_t, _ = analyze_rois(img_t, rois_t, radius=radius)
            _, df_r, _ = analyze_rois(img_r, rois_r, radius=radius)

        st.divider()
        st.subheader("目的蛋白")
        show_annotated(img_t, [(rois_t, (0, 220, 80), "T")])
        st.subheader("内参")
        show_annotated(img_r, [(rois_r, (0, 170, 255), "R")])
        st.divider()

        merged = ratio_table(df_t, df_r)
        if merged is not None:
            st.subheader("对比结果")
            st.dataframe(merged, use_container_width=True, hide_index=True)
            st.caption("Normalized_Ratio = 以 Lane 1 的 Ratio 为 1 归一化")
            st.bar_chart(merged.set_index("Lane")["Normalized_Ratio"])
            cols = ["Lane", "Band", "Area", "Mean", "IntDen", "RawIntDen"]
            base = up_t.name.rsplit(".", 1)[0] if up_t else "wb"
            excel_download({
                "目的蛋白": df_t[[c for c in cols if c in df_t.columns]],
                "内参":     df_r[[c for c in cols if c in df_r.columns]],
                "对比结果": merged,
            }, base)


# ══════════════════════════════════════════════════════════════════════════════
# 模式 3：单膜对比（同一张膜）
# ══════════════════════════════════════════════════════════════════════════════
else:
    uploaded = st.file_uploader("上传 WB 图片（含目的蛋白和内参两条带）",
                                type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])
    img = load_image(uploaded)
    if img is None:
        st.info("请上传图片后开始分析。")
        st.stop()

    tab_tgt, tab_ref = st.tabs(["🎯 目的蛋白", "⚖️ 内参"])
    with tab_tgt:
        rois_t = slider_roi(img, key="s3t", label="目的蛋白条带区域",
                            color=(0, 220, 80))
    with tab_ref:
        rois_r = slider_roi(img, key="s3r", label="内参条带区域",
                            color=(0, 170, 255))

    if not rois_t or not rois_r:
        st.stop()

    if st.button("▶️ 开始分析", key="s3_run", type="primary"):
        with st.spinner("分析中…"):
            _, df_t, _ = analyze_rois(img, rois_t, radius=radius)
            _, df_r, _ = analyze_rois(img, rois_r, radius=radius)

        st.divider()
        show_annotated(img, [
            (rois_t, (0, 220, 80),  "T"),
            (rois_r, (0, 170, 255), "R"),
        ])
        st.divider()

        cols = ["Lane", "Band", "Area", "Mean", "Min", "Max", "IntDen", "RawIntDen"]
        if df_t.empty or df_r.empty:
            st.warning("框选区域为空，请重新调整滑块。")
        else:
            merged = ratio_table(df_t, df_r)
            if merged is not None:
                st.subheader("对比结果（目的蛋白 / 内参）")
                st.dataframe(merged, use_container_width=True, hide_index=True)
                st.caption("Normalized_Ratio = 以 Lane 1 的 Ratio 为 1 归一化")
                st.bar_chart(merged.set_index("Lane")["Normalized_Ratio"])
                base = uploaded.name.rsplit(".", 1)[0] if uploaded else "wb"
                excel_download({
                    "目的蛋白": df_t[[c for c in cols if c in df_t.columns]],
                    "内参":     df_r[[c for c in cols if c in df_r.columns]],
                    "对比结果": merged,
                }, base)

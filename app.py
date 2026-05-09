import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image as PILImage

from processor import analyze, analyze_rois, preprocess

st.set_page_config(page_title="WB 条带自动定量", page_icon="🧬", layout="wide")
st.title("🧬 Western Blot 条带灰度值自动定量")

# ── 模式选择 ──────────────────────────────────────────────────────────────────
col_m1, col_m2 = st.columns(2)
with col_m1:
    mode = st.radio(
        "分析模式",
        ["单膜分析", "双膜对比（目的蛋白/内参 分开跑）", "单膜对比（目的蛋白+内参 同一张膜）"],
    )
with col_m2:
    lane_mode = st.radio(
        "泳道选择方式",
        ["🤖 自动检测", "✏️ 手动框选（滑块定位）"],
        help="手动模式：拖动滑块框住泳道区域，指定数量后自动等分。适用于自动检测不准的情况。",
    )
manual = lane_mode.startswith("✏️")
st.divider()

# ── 侧边栏 ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("参数设置")
    auto_crop = st.checkbox("自动裁剪到胶体区域", value=True,
        help="适用于 ChemiDoc 等暗背景图片，排除黑色边框干扰。")
    radius = st.slider("背景去除半径（rolling ball）", 10, 150, 50, 5)

    if not manual:
        auto_lanes = st.checkbox("自动检测泳道数", value=True)
        n_lanes: int | None = None
        if not auto_lanes:
            n_lanes = st.number_input("指定泳道数量", 1, 30, 6, 1)
        sensitivity = st.slider("检测灵敏度", 0.05, 0.80, 0.25, 0.05)
        st.markdown("**排除 Marker 泳道**")
        skip_first = st.checkbox("跳过最左侧泳道（左侧 marker）", value=False)
        skip_last  = st.checkbox("跳过最右侧泳道（右侧 marker）", value=False)
    else:
        skip_first = skip_last = False

    if mode == "单膜对比（目的蛋白+内参 同一张膜）" and not manual:
        st.divider()
        target_pos = st.radio("目的蛋白条带位置",
            ["上方条带（分子量较大）", "下方条带（分子量较小）"])
    else:
        target_pos = "上方条带（分子量较大）"

    st.divider()
    st.markdown(
        "**使用说明**\n\n"
        "- 手动框选：滑块圈定泳道区域，自动等分\n"
        "- 自动检测：调整灵敏度参数\n"
        "- 暗背景图片请勾选『自动裁剪』"
    )


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def load_image(uploaded) -> np.ndarray | None:
    if uploaded is None:
        return None
    data = np.frombuffer(uploaded.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def run_auto(img_bgr, n_bands=1):
    return analyze(
        img_bgr, radius=radius,
        n_lanes=None if auto_lanes else int(n_lanes),
        sensitivity=sensitivity,
        n_bands_per_lane=n_bands,
        auto_crop=auto_crop,
        skip_first_lane=skip_first,
        skip_last_lane=skip_last,
    )

def slider_selector(img_bgr: np.ndarray, key: str, label: str = "") -> list[tuple[int,int,int,int]]:
    """Slider-based manual lane selection. Returns list of (x0,y0,x1,y1) ROIs."""
    h, w = img_bgr.shape[:2]
    if label:
        st.markdown(f"**{label}**")

    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.number_input("泳道数量", 1, 30, 6, 1, key=f"{key}_n")
        x_range = st.slider("水平范围（覆盖所有泳道）", 0, w, (max(0, w//10), min(w, w*9//10)),
                            key=f"{key}_x")
        y_range = st.slider("垂直范围（框住条带行）", 0, h, (max(0, h//5), min(h, h*4//5)),
                            key=f"{key}_y")
        st.caption(f"图片尺寸：{w} × {h} px")

    x0_g, x1_g = x_range
    y0_g, y1_g = y_range
    lane_w = max(1, (x1_g - x0_g) // n)
    rois = [(x0_g + i * lane_w, y0_g, x0_g + (i + 1) * lane_w, y1_g) for i in range(n)]

    # Live preview
    preview = img_bgr.copy()
    for i, (rx0, ry0, rx1, ry1) in enumerate(rois):
        cv2.rectangle(preview, (rx0, ry0), (rx1, ry1), (0, 200, 80), 2)
        cv2.putText(preview, f"L{i+1}", (rx0 + 4, ry0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 80), 2)
    with c2:
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)

    return rois

def show_result_images(img_bgr, annotated_bgr, enhanced, bbox, note=""):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ann_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    disp    = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gx0, gy0, gx1, gy1 = bbox
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**原始图片** {note}")
        st.image(img_rgb, use_container_width=True)
    with c2:
        st.markdown(f"**预处理结果** {note}")
        st.image(disp, use_container_width=True, clamp=True)
    with c3:
        st.markdown(f"**检测结果** {note}")
        if not manual and auto_crop:
            st.caption(f"蓝框=胶体区域  绿框=条带  x={gx0}-{gx1}, y={gy0}-{gy1}")
        st.image(ann_rgb, use_container_width=True)

def ratio_table(df_t: pd.DataFrame, df_r: pd.DataFrame) -> pd.DataFrame:
    t = df_t[["Lane", "IntDen"]].rename(columns={"IntDen": "Target_IntDen"})
    r = df_r[["Lane", "IntDen"]].rename(columns={"IntDen": "Ref_IntDen"})
    m = t.merge(r, on="Lane", how="inner")
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

def show_single_result(df, uploaded, extra_sheets=None):
    cols = ["Lane", "Band", "Area", "Mean", "Min", "Max", "IntDen", "RawIntDen"]
    st.subheader(f"定量结果（共 {len(df)} 个条带）")
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
    uploaded = st.file_uploader("上传 WB 图片", type=["jpg","jpeg","png","tif","tiff","bmp"])
    img = load_image(uploaded)
    if img is None:
        st.info("请上传图片后开始分析。")
        st.stop()

    if manual:
        rois = slider_selector(img, key="s1")
        with st.spinner("分析中…"):
            annotated, df = analyze_rois(img, rois, radius=radius)
        enh  = preprocess(img, radius)
        bbox = (0, 0, img.shape[1], img.shape[0])
    else:
        with st.spinner("分析中…"):
            annotated, df, bbox = run_auto(img, n_bands=1)
        gx0, gy0, gx1, gy1 = bbox
        enh = preprocess(img[gy0:gy1, gx0:gx1], radius)

    st.divider()
    show_result_images(img, annotated, enh, bbox)
    st.divider()
    show_single_result(df, uploaded)


# ══════════════════════════════════════════════════════════════════════════════
# 模式 2：双膜对比
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "双膜对比（目的蛋白/内参 分开跑）":
    c1, c2 = st.columns(2)
    with c1:
        up_t = st.file_uploader("🎯 目的蛋白图片", type=["jpg","jpeg","png","tif","tiff","bmp"], key="t2")
    with c2:
        up_r = st.file_uploader("⚖️ 内参图片（β-actin / GAPDH）", type=["jpg","jpeg","png","tif","tiff","bmp"], key="r2")

    img_t = load_image(up_t)
    img_r = load_image(up_r)
    if img_t is None or img_r is None:
        st.info("请分别上传目的蛋白图片和内参图片。")
        st.stop()

    if manual:
        st.subheader("框选泳道")
        c1, c2 = st.columns(2)
        with c1:
            rois_t = slider_selector(img_t, key="t2s", label="目的蛋白图片")
        with c2:
            rois_r = slider_selector(img_r, key="r2s", label="内参图片")
        with st.spinner("分析中…"):
            ann_t, df_t = analyze_rois(img_t, rois_t, radius=radius)
            ann_r, df_r = analyze_rois(img_r, rois_r, radius=radius)
        enh_t = preprocess(img_t, radius)
        enh_r = preprocess(img_r, radius)
        bbox_t = (0, 0, img_t.shape[1], img_t.shape[0])
        bbox_r = (0, 0, img_r.shape[1], img_r.shape[0])
    else:
        with st.spinner("分析中…"):
            ann_t, df_t, bbox_t = run_auto(img_t, n_bands=1)
            ann_r, df_r, bbox_r = run_auto(img_r, n_bands=1)
        gx0,gy0,gx1,gy1 = bbox_t
        enh_t = preprocess(img_t[gy0:gy1,gx0:gx1], radius)
        gx0,gy0,gx1,gy1 = bbox_r
        enh_r = preprocess(img_r[gy0:gy1,gx0:gx1], radius)

    st.divider()
    st.subheader("目的蛋白")
    show_result_images(img_t, ann_t, enh_t, bbox_t, "（目的蛋白）")
    st.subheader("内参")
    show_result_images(img_r, ann_r, enh_r, bbox_r, "（内参）")
    st.divider()

    merged = ratio_table(df_t, df_r)
    st.subheader("对比结果")
    st.dataframe(merged, use_container_width=True, hide_index=True)
    st.caption("Normalized_Ratio = 以 Lane 1 的 Ratio 为 1 归一化")
    st.bar_chart(merged.set_index("Lane")["Normalized_Ratio"])

    cols = ["Lane","Band","Area","Mean","IntDen","RawIntDen"]
    base = up_t.name.rsplit(".",1)[0] if up_t else "wb"
    excel_download({"目的蛋白": df_t[cols], "内参": df_r[cols], "对比结果": merged}, base)


# ══════════════════════════════════════════════════════════════════════════════
# 模式 3：单膜对比（同一张膜）
# ══════════════════════════════════════════════════════════════════════════════
else:
    uploaded = st.file_uploader("上传 WB 图片（含目的蛋白和内参两条带）",
                                type=["jpg","jpeg","png","tif","tiff","bmp"])
    img = load_image(uploaded)
    if img is None:
        st.info("请上传图片后开始分析。")
        st.stop()

    if manual:
        st.subheader("框选泳道")
        st.markdown("请分别为**目的蛋白**和**内参**各设置一组框选区域")
        tab_tgt, tab_ref = st.tabs(["🎯 目的蛋白", "⚖️ 内参"])
        with tab_tgt:
            rois_t = slider_selector(img, key="s3t", label="目的蛋白条带区域")
        with tab_ref:
            rois_r = slider_selector(img, key="s3r", label="内参条带区域")
        with st.spinner("分析中…"):
            ann_t, df_t = analyze_rois(img, rois_t, radius=radius)
            ann_r, df_r = analyze_rois(img, rois_r, radius=radius)

        # Merge both sets of boxes onto one annotated image
        annotated = img.copy()
        for i, (x0,y0,x1,y1) in enumerate(sorted(rois_t, key=lambda r:r[0])):
            cv2.rectangle(annotated,(x0,y0),(x1,y1),(0,200,80),2)
            cv2.putText(annotated,f"T{i+1}",(x0+4,y0+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,80),2)
        for i, (x0,y0,x1,y1) in enumerate(sorted(rois_r, key=lambda r:r[0])):
            cv2.rectangle(annotated,(x0,y0),(x1,y1),(0,180,255),2)
            cv2.putText(annotated,f"R{i+1}",(x0+4,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,180,255),2)
        enh  = preprocess(img, radius)
        bbox = (0, 0, img.shape[1], img.shape[0])
    else:
        with st.spinner("分析中（检测每泳道前两强条带）…"):
            annotated, df_all, bbox = run_auto(img, n_bands=2)
        gx0,gy0,gx1,gy1 = bbox
        enh = preprocess(img[gy0:gy1,gx0:gx1], radius)
        target_band_no = 1 if "上方" in target_pos else 2
        df_t = df_all[df_all["Band"] == target_band_no].copy()
        df_r = df_all[df_all["Band"] == (3 - target_band_no)].copy()

    st.divider()
    show_result_images(img, annotated, enh, bbox)
    st.divider()

    cols = ["Lane","Band","Area","Mean","Min","Max","IntDen","RawIntDen"]
    if df_t.empty or df_r.empty:
        st.warning("部分泳道条带不足，请调整参数或改用手动框选。")
    else:
        merged = ratio_table(df_t, df_r)
        st.subheader("对比结果（目的蛋白 / 内参）")
        st.dataframe(merged, use_container_width=True, hide_index=True)
        st.caption("Normalized_Ratio = 以 Lane 1 的 Ratio 为 1 归一化")
        st.bar_chart(merged.set_index("Lane")["Normalized_Ratio"])

        base = uploaded.name.rsplit(".",1)[0] if uploaded else "wb"
        excel_download({
            "目的蛋白": df_t[cols] if "Band" in df_t.columns else df_t,
            "内参":     df_r[cols] if "Band" in df_r.columns else df_r,
            "对比结果": merged,
        }, base)

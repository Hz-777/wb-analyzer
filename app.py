import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image as PILImage

from processor import analyze, analyze_rois

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
    annotated, df, bbox, enhanced = analyze(
        img_bgr, radius=radius,
        n_lanes=None if auto_lanes else int(n_lanes),
        sensitivity=sensitivity,
        n_bands_per_lane=n_bands,
        auto_crop=auto_crop,
        skip_first_lane=skip_first,
        skip_last_lane=skip_last,
    )
    return annotated, df, bbox, enhanced

def manual_selector(img_bgr: np.ndarray, key: str, label: str = "") -> list[tuple[int,int,int,int]]:
    """Uniform-box manual lane selection.

    User specifies the CENTER x of each lane + a single box width that applies
    to every lane, guaranteeing all ROIs are exactly the same size so that
    Area and IntDen values are directly comparable.
    """
    h, w = img_bgr.shape[:2]
    if label:
        st.markdown(f"**{label}**")

    # ── Coordinate ruler overlay ───────────────────────────────────────────
    guide = img_bgr.copy()
    tick_step = max(20, round(w / 30 / 10) * 10)
    for x in range(0, w + 1, tick_step):
        is_major = (x % (tick_step * 5) == 0)
        cv2.line(guide, (x, 0), (x, 18 if is_major else 10), (30, 30, 200), 1)
        if is_major:
            cv2.putText(guide, str(x), (max(0, x - 12), 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 200), 1)
    cv2.line(guide, (0, 1), (w, 1), (30, 30, 200), 1)
    st.image(cv2.cvtColor(guide, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.caption(f"📐 图片尺寸：{w} × {h} px  ｜  蓝色刻度为 x 坐标（每 {tick_step} px 一格）")

    # ── Controls ───────────────────────────────────────────────────────────
    c1, c2 = st.columns([3, 2])
    with c1:
        n = st.number_input("泳道数量", 1, 30, 6, 1, key=f"{key}_n")
        default_centers = ", ".join(
            str(round(w * (2 * i + 1) / (2 * n))) for i in range(n)
        )
        centers_input = st.text_input(
            f"每条泳道中心 x 坐标（{n} 个值，逗号分隔）",
            value=default_centers,
            key=f"{key}_cx",
            help="从刻度尺读取每条泳道正中心的 x 像素值，填 n 个数字。",
        )
    with c2:
        default_bw = max(10, round(w / n * 0.7 / 5) * 5)
        box_w = st.number_input(
            "统一框宽（像素）", 5, w, default_bw, 5, key=f"{key}_bw",
            help="所有泳道使用相同的框宽，保证 Area 一致、IntDen 可比。",
        )
        y0 = st.number_input("条带上边界 y", 0, h - 1, h // 4, key=f"{key}_y0")
        y1 = st.number_input("条带下边界 y", 1, h,     h * 3 // 4, key=f"{key}_y1")

    # ── Parse centers ──────────────────────────────────────────────────────
    rois: list[tuple[int,int,int,int]] = []
    try:
        centers = [int(v.strip()) for v in centers_input.split(",") if v.strip()]
    except ValueError:
        st.error("格式有误，请输入整数，以英文逗号分隔。")
        centers = []

    if centers:
        if len(centers) != n:
            st.warning(f"需要 {n} 个中心坐标，当前填写了 {len(centers)} 个。")
        else:
            half = box_w // 2
            iy0, iy1 = int(y0), int(y1)
            rois = [(max(0, cx - half), iy0, min(w, cx + half), iy1)
                    for cx in centers]

    # ── Live preview ───────────────────────────────────────────────────────
    if rois:
        box_h = int(y1) - int(y0)
        preview = img_bgr.copy()
        for i, (rx0, ry0, rx1, ry1) in enumerate(rois):
            cv2.rectangle(preview, (rx0, ry0), (rx1, ry1), (0, 200, 80), 2)
            cv2.putText(preview, f"L{i + 1}", (rx0 + 4, ry0 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 80), 2)
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.success(f"✅ 所有框尺寸一致：宽 {box_w} × 高 {box_h} px  ｜  共 {n} 条泳道")
    else:
        st.info("填写完整的中心坐标后，预览将显示在此处。")

    return rois

def show_result_images(img_bgr, annotated_bgr, enhanced, bbox, note=""):
    """Show original | bright-band preprocessed | detection in 3 columns.

    The preprocessed panel embeds the enhanced (bright-band) crop back into a
    full-size dark canvas so the gel region is clearly visible in context.
    Bright pixels = strong band signal; black = background / outside gel.
    """
    h, w = img_bgr.shape[:2]
    gx0, gy0, gx1, gy1 = bbox

    # ── Build full-size bright-band display ───────────────────────────
    enh_norm = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_full = np.zeros((h, w), dtype=np.uint8)
    ch = min(gy1 - gy0, enh_norm.shape[0])
    cw = min(gx1 - gx0, enh_norm.shape[1])
    disp_full[gy0:gy0 + ch, gx0:gx0 + cw] = enh_norm[:ch, :cw]

    img_rgb = cv2.cvtColor(img_bgr,      cv2.COLOR_BGR2RGB)
    ann_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**原始图片** {note}")
        st.image(img_rgb, use_container_width=True)
    with c2:
        st.markdown(f"**预处理结果（亮带 = 强信号）** {note}")
        st.caption("灰度转换 + 背景去除后：条带越亮 = 蛋白含量越高，IntDen 在此图像上计算")
        st.image(disp_full, use_container_width=True, clamp=True)
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
        rois = manual_selector(img, key="s1")
        with st.spinner("分析中…"):
            annotated, df, enh = analyze_rois(img, rois, radius=radius)
        bbox = (0, 0, img.shape[1], img.shape[0])
    else:
        with st.spinner("分析中…"):
            annotated, df, bbox, enh = run_auto(img, n_bands=1)

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
            rois_t = manual_selector(img_t, key="t2s", label="目的蛋白图片")
        with c2:
            rois_r = manual_selector(img_r, key="r2s", label="内参图片")
        with st.spinner("分析中…"):
            ann_t, df_t, enh_t = analyze_rois(img_t, rois_t, radius=radius)
            ann_r, df_r, enh_r = analyze_rois(img_r, rois_r, radius=radius)
        bbox_t = (0, 0, img_t.shape[1], img_t.shape[0])
        bbox_r = (0, 0, img_r.shape[1], img_r.shape[0])
    else:
        with st.spinner("分析中…"):
            ann_t, df_t, bbox_t, enh_t = run_auto(img_t, n_bands=1)
            ann_r, df_r, bbox_r, enh_r = run_auto(img_r, n_bands=1)

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
            rois_t = manual_selector(img, key="s3t", label="目的蛋白条带区域")
        with tab_ref:
            rois_r = manual_selector(img, key="s3r", label="内参条带区域")
        with st.spinner("分析中…"):
            ann_t, df_t, enh_t = analyze_rois(img, rois_t, radius=radius)
            ann_r, df_r, enh   = analyze_rois(img, rois_r, radius=radius)

        # Merge both sets of boxes onto one annotated image
        annotated = img.copy()
        for i, (x0,y0,x1,y1) in enumerate(sorted(rois_t, key=lambda r:r[0])):
            cv2.rectangle(annotated,(x0,y0),(x1,y1),(0,200,80),2)
            cv2.putText(annotated,f"T{i+1}",(x0+4,y0+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,80),2)
        for i, (x0,y0,x1,y1) in enumerate(sorted(rois_r, key=lambda r:r[0])):
            cv2.rectangle(annotated,(x0,y0),(x1,y1),(0,180,255),2)
            cv2.putText(annotated,f"R{i+1}",(x0+4,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,180,255),2)
        bbox = (0, 0, img.shape[1], img.shape[0])
    else:
        with st.spinner("分析中（检测每泳道前两强条带）…"):
            annotated, df_all, bbox, enh = run_auto(img, n_bands=2)
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

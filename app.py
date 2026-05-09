import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image as PILImage
from streamlit_drawable_canvas import st_canvas

from processor import analyze, analyze_rois, preprocess

st.set_page_config(page_title="WB 条带自动定量", page_icon="🧬", layout="wide")
st.title("🧬 Western Blot 条带灰度值自动定量")

# ── 分析模式 & 泳道选择方式 ────────────────────────────────────────────────
col_m1, col_m2 = st.columns(2)
with col_m1:
    mode = st.radio(
        "分析模式",
        ["单膜分析", "双膜对比（目的蛋白/内参 分开跑）", "单膜对比（目的蛋白+内参 同一张膜）"],
    )
with col_m2:
    lane_mode = st.radio(
        "泳道选择方式",
        ["🤖 自动检测", "✏️ 手动框选（在图片上拖框）"],
        help="手动框选：用鼠标在图片上拖动画出矩形，每个矩形 = 一条泳道，支持撤销（Delete 键）。",
    )
manual = lane_mode.startswith("✏️")
st.divider()

# ── 侧边栏参数 ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("参数设置")
    auto_crop = st.checkbox("自动裁剪到胶体区域", value=True,
        help="适用于 ChemiDoc 等暗背景图片，排除黑色边框干扰。手动框选时仍用于预处理。")
    radius = st.slider("背景去除半径（rolling ball）", 10, 150, 50, 5)

    if not manual:
        auto_lanes = st.checkbox("自动检测泳道数", value=True)
        n_lanes: int | None = None
        if not auto_lanes:
            n_lanes = st.number_input("指定泳道数量", 1, 30, 6, 1)
        sensitivity = st.slider("检测灵敏度", 0.05, 0.80, 0.25, 0.05)

    if mode == "单膜对比（目的蛋白+内参 同一张膜）" and not manual:
        st.divider()
        target_pos = st.radio("目的蛋白条带位置",
            ["上方条带（分子量较大）", "下方条带（分子量较小）"])

    st.divider()
    st.markdown(
        "**使用说明**\n\n"
        "- 手动框选：在图片上拖框标记每条泳道\n"
        "- 自动检测：调整灵敏度参数\n"
        "- 暗背景图片请勾选『自动裁剪』"
    )


# ── 工具函数 ───────────────────────────────────────────────────────────────
CANVAS_MAX_W = 900

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
    )

def canvas_selector(img_bgr: np.ndarray, key: str, label: str = "") -> list[tuple[int,int,int,int]]:
    """Show a drawable canvas over the image; return list of (x0,y0,x1,y1) ROIs."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, CANVAS_MAX_W / w)
    cw, ch = int(w * scale), int(h * scale)
    img_pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    if label:
        st.markdown(f"**{label}** — 在图片上拖动鼠标画矩形框选每条泳道，Delete 键删除最后一个框")

    result = st_canvas(
        fill_color="rgba(0, 200, 80, 0.15)",
        stroke_width=2,
        stroke_color="#00C850",
        background_image=img_pil,
        drawing_mode="rect",
        width=cw,
        height=ch,
        key=key,
        update_streamlit=True,
    )

    rois = []
    if result.json_data:
        for obj in result.json_data.get("objects", []):
            if obj.get("type") != "rect":
                continue
            # canvas coords → image coords
            x0 = int(obj["left"] / scale)
            y0 = int(obj["top"] / scale)
            x1 = int((obj["left"] + obj["width"]) / scale)
            y1 = int((obj["top"] + obj["height"]) / scale)
            # clamp
            x0, x1 = max(0, x0), min(w, x1)
            y0, y1 = max(0, y0), min(h, y1)
            if x1 > x0 and y1 > y0:
                rois.append((x0, y0, x1, y1))
    return rois

def show_result_images(img_bgr, annotated_bgr, enhanced, bbox, note=""):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ann_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    disp = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
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
        rois = canvas_selector(img, key="s1")
        if not rois:
            st.info("请在图片上拖框标记每条泳道，框好后结果自动显示。")
            st.stop()
        with st.spinner("分析中…"):
            annotated, df = analyze_rois(img, rois, radius=radius)
        enh = preprocess(img, radius)
        bbox = (0, 0, img.shape[1], img.shape[0])
    else:
        with st.spinner("分析中…"):
            annotated, df, bbox = run_auto(img, n_bands=1)
        gx0,gy0,gx1,gy1 = bbox
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
            rois_t = canvas_selector(img_t, key="t2c", label="目的蛋白图片")
        with c2:
            rois_r = canvas_selector(img_r, key="r2c", label="内参图片")

        if not rois_t or not rois_r:
            st.info("请在两张图片上分别框选泳道。")
            st.stop()

        with st.spinner("分析中…"):
            ann_t, df_t = analyze_rois(img_t, rois_t, radius=radius)
            ann_r, df_r = analyze_rois(img_r, rois_r, radius=radius)
        enh_t = preprocess(img_t, radius)
        enh_r = preprocess(img_r, radius)
        bbox_t = (0,0,img_t.shape[1],img_t.shape[0])
        bbox_r = (0,0,img_r.shape[1],img_r.shape[0])
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
        st.markdown("请分别框选**目的蛋白**条带区域和**内参**条带区域（各框一次，每条泳道画一个框）")
        tab_tgt, tab_ref = st.tabs(["🎯 目的蛋白", "⚖️ 内参"])
        with tab_tgt:
            rois_t = canvas_selector(img, key="s3t", label="目的蛋白条带（每条泳道框一个）")
        with tab_ref:
            rois_r = canvas_selector(img, key="s3r", label="内参条带（每条泳道框一个）")

        if not rois_t or not rois_r:
            st.info("请在两个标签页中分别框选目的蛋白和内参的条带区域。")
            st.stop()

        with st.spinner("分析中…"):
            ann_t, df_t = analyze_rois(img, rois_t, radius=radius)
            ann_r, df_r = analyze_rois(img, rois_r, radius=radius)
        enh = preprocess(img, radius)
        bbox = (0,0,img.shape[1],img.shape[0])

        # Merge annotated images
        annotated = img.copy()
        for i, (x0,y0,x1,y1) in enumerate(sorted(rois_t, key=lambda r:r[0])):
            cv2.rectangle(annotated,(x0,y0),(x1,y1),(0,200,80),2)
            cv2.putText(annotated,f"T{i+1}",(x0+4,y0+18),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,80),2)
        for i, (x0,y0,x1,y1) in enumerate(sorted(rois_r, key=lambda r:r[0])):
            cv2.rectangle(annotated,(x0,y0),(x1,y1),(0,180,255),2)
            cv2.putText(annotated,f"R{i+1}",(x0+4,y0+18),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,180,255),2)
    else:
        target_pos = target_pos if "target_pos" in dir() else "上方条带（分子量较大）"
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
            "内参": df_r[cols] if "Band" in df_r.columns else df_r,
            "对比结果": merged,
        }, base)

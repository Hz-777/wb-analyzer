import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from processor import analyze, preprocess, find_gel_bbox

st.set_page_config(
    page_title="WB 条带自动定量",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Western Blot 条带灰度值自动定量")
st.markdown(
    "基于 ImageJ 工作流程自动完成：灰度转换 → 背景去除 → 条带检测 → IntDen 计算"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("参数设置")

    auto_crop = st.checkbox(
        "自动裁剪到胶体区域",
        value=True,
        help="适用于化学发光成像仪（ChemiDoc等）拍摄的暗背景图片。"
             "自动识别膜所在的亮色矩形区域，排除黑色边框和定标符号的干扰。",
    )

    radius = st.slider(
        "背景去除半径（rolling ball radius）",
        min_value=10, max_value=150, value=50, step=5,
        help="与 ImageJ Subtract Background 中的 radius 一致，默认 50。",
    )

    auto_lanes = st.checkbox("自动检测泳道数", value=True)
    n_lanes: int | None = None
    if not auto_lanes:
        n_lanes = st.number_input("指定泳道数量", min_value=1, max_value=30, value=6, step=1)

    sensitivity = st.slider(
        "检测灵敏度",
        min_value=0.05, max_value=0.80, value=0.25, step=0.05,
        help="数值越低越容易检测到弱条带，但可能产生误报。",
    )

    strongest_only = st.checkbox(
        "每泳道只取最强条带",
        value=True,
        help="忽略非特异性弱带，仅对每条泳道 IntDen 最大的条带定量（推荐）。",
    )

    st.divider()
    st.markdown("**使用说明**")
    st.markdown(
        "1. 上传 WB 原始图片\n"
        "2. 暗背景图片请勾选"自动裁剪到胶体区域"\n"
        "3. 调整参数直到检测结果准确\n"
        "4. 下载 Excel 结果文件"
    )

# ── Main area ─────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "上传 WB 图片（JPG / PNG / TIF）",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
)

if uploaded is None:
    st.info("请在左侧上传图片后开始分析。")
    st.stop()

file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if img_bgr is None:
    st.error("无法读取图片，请检查文件格式。")
    st.stop()

# ── Run analysis ──────────────────────────────────────────────────────────────
with st.spinner("正在分析图片…"):
    annotated_bgr, df, (gx0, gy0, gx1, gy1) = analyze(
        img_bgr,
        radius=radius,
        n_lanes=None if auto_lanes else int(n_lanes),
        sensitivity=sensitivity,
        strongest_only=strongest_only,
        auto_crop=auto_crop,
    )

    # Cropped region for preprocessing preview
    img_crop = img_bgr[gy0:gy1, gx0:gx1]
    enhanced_crop = preprocess(img_crop, radius=radius)
    enhanced_display = cv2.normalize(enhanced_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ── Image display ─────────────────────────────────────────────────────────────
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("原始图片")
    st.image(img_rgb, use_container_width=True)

with col2:
    st.subheader("预处理结果（背景去除 + 取反）")
    if auto_crop:
        st.caption(f"已裁剪到胶体区域：x={gx0}–{gx1}, y={gy0}–{gy1}")
    st.image(enhanced_display, use_container_width=True, clamp=True)

with col3:
    st.subheader("条带检测结果")
    if auto_crop:
        st.caption("蓝框 = 自动识别的胶体区域，绿框 = 检测到的条带")
    st.image(annotated_rgb, use_container_width=True)

# ── Results table ─────────────────────────────────────────────────────────────
st.divider()
st.subheader(f"定量结果（共检测到 {len(df)} 个条带）")

if df.empty:
    st.warning("未检测到任何条带，请尝试：降低检测灵敏度、调整背景去除半径，或取消勾选"自动裁剪"后手动观察。")
else:
    display_cols = ["Lane", "Band", "Area", "Mean", "Min", "Max", "IntDen", "RawIntDen"]
    st.dataframe(
        df[display_cols].style.format({
            "Area": "{:,}",
            "Mean": "{:.2f}",
            "Min": "{:.1f}",
            "Max": "{:.1f}",
            "IntDen": "{:,.1f}",
            "RawIntDen": "{:,.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Download ──────────────────────────────────────────────────────────────
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df[display_cols].to_excel(writer, sheet_name="定量结果", index=False)

        if len(df) > 1:
            ref = df.loc[df["Lane"] == df["Lane"].min(), "IntDen"].values[0]
            df_norm = df[["Lane", "Band", "IntDen"]].copy()
            df_norm["Relative_IntDen"] = (df_norm["IntDen"] / ref).round(4)
            df_norm.to_excel(writer, sheet_name="相对定量（归一化）", index=False)

    output.seek(0)
    filename = uploaded.name.rsplit(".", 1)[0] + "_WB_quantification.xlsx"
    st.download_button(
        label="⬇️  下载 Excel 结果",
        data=output,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ── Bar chart ─────────────────────────────────────────────────────────────
    st.subheader("IntDen 相对比较（以最小泳道号为基准 = 1）")
    ref = df.loc[df["Lane"] == df["Lane"].min(), "IntDen"].values[0]
    summary = df.groupby("Lane")["IntDen"].sum().reset_index()
    summary.columns = ["泳道", "IntDen 合计"]
    summary["相对值"] = (summary["IntDen 合计"] / ref).round(4)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.bar_chart(summary.set_index("泳道")["IntDen 合计"])

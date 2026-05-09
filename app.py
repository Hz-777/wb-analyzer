import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from processor import analyze, preprocess

st.set_page_config(page_title="WB 条带自动定量", page_icon="🧬", layout="wide")

st.title("🧬 Western Blot 条带灰度值自动定量")

# ── 模式选择 ──────────────────────────────────────────────────────────────────
mode = st.radio(
    "分析模式",
    ["单膜分析", "双膜对比（目的蛋白 / 内参 分开跑）", "单膜对比（目的蛋白 + 内参 同一张膜）"],
    horizontal=True,
)
st.divider()

# ── 公共参数（侧边栏）────────────────────────────────────────────────────────
with st.sidebar:
    st.header("参数设置")

    auto_crop = st.checkbox(
        "自动裁剪到胶体区域",
        value=True,
        help="适用于 ChemiDoc 等暗背景图片，自动排除黑色边框和定标符号干扰。",
    )
    radius = st.slider("背景去除半径（rolling ball）", 10, 150, 50, 5)
    auto_lanes = st.checkbox("自动检测泳道数", value=True)
    n_lanes: int | None = None
    if not auto_lanes:
        n_lanes = st.number_input("指定泳道数量", min_value=1, max_value=30, value=6, step=1)
    sensitivity = st.slider("检测灵敏度", 0.05, 0.80, 0.25, 0.05)

    if mode == "单膜对比（目的蛋白 + 内参 同一张膜）":
        st.divider()
        target_pos = st.radio(
            "目的蛋白条带位置",
            ["上方条带（分子量较大）", "下方条带（分子量较小）"],
            help="决定哪条带作为目的蛋白，另一条自动作为内参。",
        )

    st.divider()
    st.markdown(
        "**使用说明**\n\n"
        "1. 选择分析模式\n"
        "2. 上传图片\n"
        "3. 调整参数直到检测准确\n"
        "4. 下载 Excel 结果"
    )


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def load_image(uploaded) -> np.ndarray | None:
    if uploaded is None:
        return None
    data = np.frombuffer(uploaded.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def run_analyze(img_bgr, n_bands=1):
    return analyze(
        img_bgr,
        radius=radius,
        n_lanes=None if auto_lanes else int(n_lanes),
        sensitivity=sensitivity,
        n_bands_per_lane=n_bands,
        auto_crop=auto_crop,
    )


def show_images(img_bgr, annotated_bgr, enhanced_crop, bbox, label=""):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ann_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    disp = cv2.normalize(enhanced_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gx0, gy0, gx1, gy1 = bbox
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**原始图片** {label}")
        st.image(img_rgb, use_container_width=True)
    with c2:
        st.markdown(f"**预处理结果** {label}")
        st.image(disp, use_container_width=True, clamp=True)
    with c3:
        st.markdown(f"**检测结果** {label}")
        if auto_crop:
            st.caption(f"蓝框=胶体区域  绿/青框=条带  裁剪范围 x={gx0}-{gx1}, y={gy0}-{gy1}")
        st.image(ann_rgb, use_container_width=True)


def ratio_table(df_target: pd.DataFrame, df_ref: pd.DataFrame) -> pd.DataFrame:
    """Merge target + reference by Lane and compute ratio."""
    t = df_target[["Lane", "IntDen"]].rename(columns={"IntDen": "Target_IntDen"})
    r = df_ref[["Lane", "IntDen"]].rename(columns={"IntDen": "Ref_IntDen"})
    merged = t.merge(r, on="Lane", how="inner")
    merged["Ratio"] = (merged["Target_IntDen"] / merged["Ref_IntDen"]).round(4)
    ref_ratio = merged["Ratio"].iloc[0]
    merged["Normalized_Ratio"] = (merged["Ratio"] / ref_ratio).round(4)
    return merged


def excel_download(dfs: dict[str, pd.DataFrame], base_name: str):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for sheet, df in dfs.items():
            df.to_excel(w, sheet_name=sheet, index=False)
    buf.seek(0)
    st.download_button(
        "⬇️  下载 Excel 结果",
        data=buf,
        file_name=base_name + "_WB_quantification.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 模式 1：单膜分析
# ══════════════════════════════════════════════════════════════════════════════
if mode == "单膜分析":
    uploaded = st.file_uploader("上传 WB 图片", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])
    img = load_image(uploaded)
    if img is None:
        st.info("请上传图片后开始分析。")
        st.stop()

    with st.spinner("分析中…"):
        annotated, df, bbox = run_analyze(img, n_bands=1)
        crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        enhanced = preprocess(crop, radius)

    show_images(img, annotated, enhanced, bbox)
    st.divider()
    st.subheader(f"定量结果（共 {len(df)} 个条带）")
    cols = ["Lane", "Band", "Area", "Mean", "Min", "Max", "IntDen", "RawIntDen"]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)

    base = uploaded.name.rsplit(".", 1)[0]
    excel_download({"定量结果": df[cols]}, base)

    if len(df) > 1:
        st.subheader("IntDen 柱状图（以 Lane 1 为基准）")
        ref = df["IntDen"].iloc[0]
        summary = df.groupby("Lane")["IntDen"].sum().reset_index()
        summary["Normalized"] = (summary["IntDen"] / ref).round(4)
        st.bar_chart(summary.set_index("Lane")["IntDen"])


# ══════════════════════════════════════════════════════════════════════════════
# 模式 2：双膜对比
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "双膜对比（目的蛋白 / 内参 分开跑）":
    col_a, col_b = st.columns(2)
    with col_a:
        up_target = st.file_uploader("🎯 目的蛋白图片", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"], key="t")
    with col_b:
        up_ref = st.file_uploader("⚖️ 内参图片（如 β-actin / GAPDH）", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"], key="r")

    img_t = load_image(up_target)
    img_r = load_image(up_ref)

    if img_t is None or img_r is None:
        st.info("请分别上传目的蛋白图片和内参图片。")
        st.stop()

    with st.spinner("分析中…"):
        ann_t, df_t, bbox_t = run_analyze(img_t, n_bands=1)
        ann_r, df_r, bbox_r = run_analyze(img_r, n_bands=1)
        crop_t = img_t[bbox_t[1]:bbox_t[3], bbox_t[0]:bbox_t[2]]
        crop_r = img_r[bbox_r[1]:bbox_r[3], bbox_r[0]:bbox_r[2]]
        enh_t = preprocess(crop_t, radius)
        enh_r = preprocess(crop_r, radius)

    st.subheader("目的蛋白")
    show_images(img_t, ann_t, enh_t, bbox_t, "（目的蛋白）")
    st.subheader("内参")
    show_images(img_r, ann_r, enh_r, bbox_r, "（内参）")
    st.divider()

    st.subheader("对比结果")
    merged = ratio_table(df_t, df_r)
    st.dataframe(merged, use_container_width=True, hide_index=True)
    st.caption("Normalized_Ratio = 以 Lane 1 的 Ratio 为 1 归一化")
    st.bar_chart(merged.set_index("Lane")["Normalized_Ratio"])

    base = (up_target.name.rsplit(".", 1)[0] if up_target else "wb")
    cols_t = ["Lane", "Band", "Area", "Mean", "IntDen", "RawIntDen"]
    excel_download({
        "目的蛋白": df_t[cols_t],
        "内参": df_r[cols_t],
        "对比结果": merged,
    }, base)


# ══════════════════════════════════════════════════════════════════════════════
# 模式 3：单膜对比（目的蛋白 + 内参 同一张膜）
# ══════════════════════════════════════════════════════════════════════════════
else:
    uploaded = st.file_uploader("上传 WB 图片（含目的蛋白和内参两条带）",
                                type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])
    img = load_image(uploaded)
    if img is None:
        st.info("请上传图片后开始分析。")
        st.stop()

    with st.spinner("分析中（检测每泳道前两强条带）…"):
        annotated, df, bbox = run_analyze(img, n_bands=2)
        crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        enhanced = preprocess(crop, radius)

    show_images(img, annotated, enhanced, bbox)
    st.divider()

    st.subheader("检测到的所有条带")
    cols = ["Lane", "Band", "Area", "Mean", "Min", "Max", "IntDen", "RawIntDen"]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)
    st.caption("Band 1 = 上方条带（分子量较大），Band 2 = 下方条带（分子量较小）")

    # 根据用户选择分配目的蛋白 / 内参
    target_band_no = 1 if "上方" in target_pos else 2
    ref_band_no = 2 if target_band_no == 1 else 1

    df_target = df[df["Band"] == target_band_no].copy()
    df_ref    = df[df["Band"] == ref_band_no].copy()

    if df_target.empty or df_ref.empty:
        st.warning("部分泳道只检测到一条带，请降低灵敏度或调整参数后重试。")
    else:
        st.subheader("对比结果（目的蛋白 / 内参）")
        merged = ratio_table(df_target, df_ref)
        st.dataframe(merged, use_container_width=True, hide_index=True)
        st.caption("Normalized_Ratio = 以 Lane 1 的 Ratio 为 1 归一化")
        st.bar_chart(merged.set_index("Lane")["Normalized_Ratio"])

        base = uploaded.name.rsplit(".", 1)[0]
        excel_download({
            "全部条带": df[cols],
            f"目的蛋白（Band {target_band_no}）": df_target[cols],
            f"内参（Band {ref_band_no}）": df_ref[cols],
            "对比结果": merged,
        }, base)

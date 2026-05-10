import io
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from processor import analyze_rois

# ── color palettes ────────────────────────────────────────────────────────────
PALETTES = {
    # ── 期刊配色 ──────────────────────────────────────────────────────────────
    "NPG · Nature":       ["#E64B35","#4DBBD5","#00A087","#3C5488","#F39B7F","#8491B4","#91D1C2","#DC0000","#7E6148","#B09C85"],
    "NEJM":               ["#BC3C29","#0072B5","#E18727","#20854E","#7876B1","#6F99AD","#FFDC91","#EE4C97"],
    "Lancet":             ["#00468B","#ED0000","#42B540","#0099B4","#925E9F","#FDAF91","#AD002A","#ADB6B6"],
    "JAMA":               ["#374E55","#DF8F44","#00A1D5","#B24745","#79AF97","#6A6599","#80796B"],
    "Science · AAAS":     ["#3B4992","#EE0000","#008B45","#631879","#008280","#BB0021","#5F559B","#A20056"],
    # ── 数据可视化经典 ────────────────────────────────────────────────────────
    "Tableau 10":         ["#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F","#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC"],
    "Set2 · ColorBrewer": ["#66C2A5","#FC8D62","#8DA0CB","#E78AC3","#A6D854","#FFD92F","#E5C494","#B3B3B3"],
    # ── 设计师风格 ────────────────────────────────────────────────────────────
    "Wes Anderson · Zissou": ["#3B9AB2","#78B7C5","#EBCC2A","#E1AF00","#F21A00"],
    "Pastel":             ["#AEC6CF","#FFB347","#B5EAD7","#FF6961","#C7CEEA","#FDFD96","#77DD77","#F49AC2"],
    "Flat UI":            ["#1ABC9C","#3498DB","#9B59B6","#F39C12","#E74C3C","#2ECC71","#E67E22","#34495E"],
    # ── 单色系 ───────────────────────────────────────────────────────────────
    "蓝色渐变":            ["#08306B","#08519C","#2171B5","#4292C6","#6BAED6","#9ECAE1","#C6DBEF"],
    "灰度（发表用）":      ["#252525","#525252","#737373","#969696","#BDBDBD"],
}

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
    st.subheader("📊 图表样式")
    st.selectbox("配色方案", list(PALETTES.keys()), key="viz_palette")
    st.selectbox("图表类型", ["竖向柱状图", "横向柱状图", "棒棒糖图"], key="viz_type")
    st.checkbox("显示数值标签", value=True, key="viz_showval")
    st.slider("图表高度 px", 280, 700, 400, 20, key="viz_height")
    st.slider("字体大小", 8, 22, 13, 1, key="viz_fontsize")
    st.slider("轴线 / 描边粗细", 0.5, 4.0, 1.0, 0.5, key="viz_linewidth")
    st.checkbox("自定义 Y 轴范围", value=False, key="viz_ycustom")
    if st.session_state.get("viz_ycustom"):
        _yc1, _yc2 = st.columns(2)
        _yc1.number_input("Y 最小值", value=0.0, step=0.1, format="%.2f", key="viz_ymin")
        _yc2.number_input("Y 最大值", value=2.0, step=0.1, format="%.2f", key="viz_ymax")
    st.divider()
    st.subheader("🔬 图像设置")
    st.selectbox(
        "背景类型",
        ["自动检测", "亮背景·暗条带（普通染色/扫描）", "暗背景·亮条带（化学发光/荧光）"],
        key="img_bg_type",
        help="若目的蛋白 IntDen 全为 0，尝试手动切换背景类型",
    )
    st.divider()
    st.markdown(
        "**操作流程**\n\n"
        "① 框出膜的范围（粗选）\n\n"
        "② 在放大图上精确框选条带\n\n"
        "点「🗑」重新开始该步骤"
    )


# ── helpers ───────────────────────────────────────────────────────────────────
def load_image(uploaded) -> np.ndarray | None:
    if uploaded is None:
        return None
    data = np.frombuffer(uploaded.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _plotly_selector(img_bgr, key, color, disp_h, dragmode="select", single=False):
    """Core Plotly chart with drag-to-select. Returns updated boxes list."""
    ck_boxes = f"{key}_boxes"
    ck_last  = f"{key}_last"
    for k, d in [(ck_boxes, []), (ck_last, None)]:
        if k not in st.session_state:
            st.session_state[k] = d

    h, w    = img_bgr.shape[:2]
    boxes   = st.session_state[ck_boxes]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig = go.Figure()
    fig.add_trace(go.Image(z=img_rgb))

    fs = max(11, int(max(1.0, w / 800) * 13))
    draw = boxes[-1:] if single and boxes else boxes   # single mode: show only last
    for i, (x0, y0, x1, y1) in enumerate(draw):
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color=color, width=2))
        if not single:
            fig.add_annotation(x=x0+4, y=y0+fs, text=f"<b>L{i+1}</b>",
                               showarrow=False, xanchor="left",
                               font=dict(color=color, size=fs))

    fig.update_layout(
        dragmode=dragmode,
        margin=dict(l=0, r=0, t=0, b=0),
        height=disp_h,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        newselection=dict(line=dict(color=color, width=2, dash="dash")),
    )

    event = st.plotly_chart(fig, use_container_width=True, key=f"{key}_chart",
                            on_select="rerun", selection_mode=["box"])

    if dragmode == "select" and event and event.selection:
        sel = event.selection.get("box", [])
        if sel:
            sig = str(sel[0])
            if sig != st.session_state[ck_last]:
                b  = sel[0]
                xs = b.get("x", [])
                ys = b.get("y", [])
                if len(xs) >= 2 and len(ys) >= 2:
                    bx0 = max(0, min(w-1, int(round(min(xs)))))
                    bx1 = max(0, min(w-1, int(round(max(xs)))))
                    by0 = max(0, min(h-1, int(round(min(ys)))))
                    by1 = max(0, min(h-1, int(round(max(ys)))))
                    if bx1 - bx0 > 4 and by1 - by0 > 4:
                        if single:
                            st.session_state[ck_boxes] = [(bx0, by0, bx1, by1)]
                        else:
                            st.session_state[ck_boxes].append((bx0, by0, bx1, by1))
                st.session_state[ck_last] = sig
                st.rerun()

    return st.session_state[ck_boxes]


def crop_selector(img_bgr, key):
    """Step 1: coarsely select the membrane region on the full image."""
    ck_mode = f"{key}_crop_mode"
    if ck_mode not in st.session_state:
        st.session_state[ck_mode] = "select"

    h, w   = img_bgr.shape[:2]
    boxes  = st.session_state.get(f"{key}_crop_boxes", [])
    crop   = boxes[-1] if boxes else None

    st.markdown("**第一步：框出膜上条带大致范围**（粗选即可）")

    c1, c2, c3 = st.columns([3, 3, 1])
    with c1:
        if st.button("🔍 放大/平移", key=f"{key}_crop_pan",
                     type="primary" if st.session_state[ck_mode]=="pan" else "secondary"):
            st.session_state[ck_mode] = "pan"
            st.rerun()
    with c2:
        if st.button("⬜ 框选膜", key=f"{key}_crop_sel",
                     type="primary" if st.session_state[ck_mode]=="select" else "secondary"):
            st.session_state[ck_mode] = "select"
            st.rerun()
    with c3:
        if st.button("🗑", key=f"{key}_crop_clr"):
            st.session_state[f"{key}_crop_boxes"] = []
            st.session_state[f"{key}_crop_last"]  = None
            st.rerun()

    if crop:
        x0, y0, x1, y1 = crop
        st.success(f"✅ 粗选区域：x {x0}–{x1}，y {y0}–{y1}　↓ 请在第二步框选定量蛋白")
    elif st.session_state[ck_mode] == "pan":
        st.info("🔍 平移/缩放定位后，切换到「⬜ 框选膜」")
    else:
        st.info("拖动鼠标框出膜的大致范围")

    disp_h = min(400, max(200, int(500 * h / w)))
    _plotly_selector(img_bgr, key=f"{key}_crop",
                     color="#ff8800", disp_h=disp_h,
                     dragmode=st.session_state[ck_mode], single=True)

    boxes = st.session_state.get(f"{key}_crop_boxes", [])
    return boxes[-1] if boxes else None


def box_selector(img_bgr, key, label="", color="#00dc50"):
    """Step 2: precisely select bands on the (cropped) image."""
    ck_mode = f"{key}_mode"
    if ck_mode not in st.session_state:
        st.session_state[ck_mode] = "select"

    h, w = img_bgr.shape[:2]
    boxes = st.session_state.get(f"{key}_boxes", [])

    if label:
        st.markdown(f"**{label}**")

    c1, c2, c3 = st.columns([3, 3, 1])
    with c1:
        if st.button("🔍 放大/平移", key=f"{key}_pan",
                     type="primary" if st.session_state[ck_mode]=="pan" else "secondary"):
            st.session_state[ck_mode] = "pan"
            st.rerun()
    with c2:
        if st.button("⬜ 框选条带", key=f"{key}_sel",
                     type="primary" if st.session_state[ck_mode]=="select" else "secondary"):
            st.session_state[ck_mode] = "select"
            st.rerun()
    with c3:
        if st.button("🗑 清空", key=f"{key}_clr"):
            st.session_state[f"{key}_boxes"] = []
            st.session_state[f"{key}_last"]  = None
            st.rerun()

    cur_mode = st.session_state[ck_mode]
    if cur_mode == "pan":
        st.info("🔍 缩放定位后，切换「⬜ 框选条带」")
    elif boxes:
        st.success(f"✅ 已画 {len(boxes)} 个框，继续拖动可添加")
    else:
        st.info("⬜ 拖动鼠标框选每条泳道的条带")

    disp_h = min(650, max(300, int(700 * h / w)))
    boxes  = _plotly_selector(img_bgr, key=key, color=color,
                               disp_h=disp_h, dragmode=cur_mode)

    # Fine-tune panel
    if boxes:
        with st.expander(f"🎯 精确调整坐标（{len(boxes)} 个框）"):
            with st.form(key=f"{key}_tune_form"):
                hdr = st.columns([1, 2, 2, 2, 2])
                for txt, col in zip(["泳道", "x 起", "y 起", "x 止", "y 止"], hdr):
                    col.markdown(f"**{txt}**")
                new_vals = []
                for i, (x0, y0, x1, y1) in enumerate(boxes):
                    row = st.columns([1, 2, 2, 2, 2])
                    row[0].markdown(f"L{i+1}")
                    nx0 = row[1].number_input("", 0, w-1, int(x0), step=1,
                                               key=f"{key}_fx0_{i}", label_visibility="collapsed")
                    ny0 = row[2].number_input("", 0, h-1, int(y0), step=1,
                                               key=f"{key}_fy0_{i}", label_visibility="collapsed")
                    nx1 = row[3].number_input("", 0, w-1, int(x1), step=1,
                                               key=f"{key}_fx1_{i}", label_visibility="collapsed")
                    ny1 = row[4].number_input("", 0, h-1, int(y1), step=1,
                                               key=f"{key}_fy1_{i}", label_visibility="collapsed")
                    new_vals.append((int(nx0), int(ny0), int(nx1), int(ny1)))
                if st.form_submit_button("✅ 应用调整"):
                    st.session_state[f"{key}_boxes"] = new_vals
                    st.rerun()

    return sorted(st.session_state.get(f"{key}_boxes", []), key=lambda r: r[0])


def get_force_type() -> str | None:
    choice = st.session_state.get("img_bg_type", "自动检测")
    if "亮背景" in choice:
        return "light"
    if "暗背景" in choice:
        return "dark"
    return None


def offset_rois(rois, ox, oy):
    """Translate cropped-image ROI coordinates back to original image space."""
    return [(ox+x0, oy+y0, ox+x1, oy+y1) for x0, y0, x1, y1 in rois]


def lane_names_input(n_lanes, key):
    """Editable group-name inputs for each lane. Returns list of name strings."""
    ck = f"{key}_lnames"
    saved = st.session_state.get(ck, [])
    # Grow or shrink to match current lane count, preserving existing entries
    names = [saved[i] if i < len(saved) else f"Lane {i+1}" for i in range(n_lanes)]
    st.session_state[ck] = names

    st.markdown("**泳道分组名称**（可修改）")
    cols = st.columns(min(n_lanes, 8))
    result = []
    for i in range(n_lanes):
        col = cols[i % len(cols)]
        val = col.text_input(f"L{i+1}", value=names[i],
                             key=f"{key}_ln_{i}")
        result.append(val.strip() or f"Lane {i+1}")
    st.session_state[ck] = result
    return result


def apply_names(df, names):
    """Insert a Group column that maps Lane number to user-provided name."""
    mapping = {i + 1: n for i, n in enumerate(names)}
    out = df.copy()
    out.insert(1, "Group", out["Lane"].map(mapping).fillna(""))
    return out


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


def ratio_table(df_t, df_r, ref_lane=1):
    for col in ("Lane", "IntDen"):
        if col not in df_t.columns or col not in df_r.columns:
            st.error(f"结果表缺少列 '{col}'")
            return None
    t_cols = ["Lane"] + (["Group"] if "Group" in df_t.columns else []) + ["IntDen"]
    r_cols = ["Lane", "IntDen"]
    t = df_t[t_cols].rename(columns={"IntDen":"Target_IntDen"})
    r = df_r[r_cols].rename(columns={"IntDen":"Ref_IntDen"})
    m = t.merge(r, on="Lane", how="inner")
    if m.empty:
        st.error("目的蛋白与内参泳道数量不一致")
        return None
    m["Ratio"] = (m["Target_IntDen"] / m["Ref_IntDen"]).round(4)
    ref_rows = m[m["Lane"] == ref_lane]["Ratio"]
    ref_val = ref_rows.iloc[0] if not ref_rows.empty else m["Ratio"].iloc[0]
    m["Normalized_Ratio"] = (m["Ratio"] / ref_val).round(4)
    return m


def excel_download(sheets, base, key="dl"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    buf.seek(0)
    st.download_button("⬇️ 下载 Excel", data=buf,
        file_name=base + "_WB.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key)


def make_chart(series, y_label="IntDen"):
    """Render a publication-quality Plotly chart using sidebar style settings."""
    palette    = st.session_state.get("viz_palette",   "NPG · Nature")
    chart_type = st.session_state.get("viz_type",      "竖向柱状图")
    show_val   = st.session_state.get("viz_showval",   True)
    fig_h      = st.session_state.get("viz_height",    400)
    fontsize   = st.session_state.get("viz_fontsize",  13)
    lw         = st.session_state.get("viz_linewidth", 1.0)
    ycustom    = st.session_state.get("viz_ycustom",   False)
    ymin_usr   = st.session_state.get("viz_ymin",      0.0)
    ymax_usr   = st.session_state.get("viz_ymax",      2.0)

    colors     = PALETTES[palette]
    labels     = [str(x) for x in series.index]
    vals       = series.values.tolist()
    bar_colors = [colors[i % len(colors)] for i in range(len(vals))]
    text       = [f"{v:.3g}" for v in vals] if show_val else None
    max_val    = max(vals) if vals else 1

    # Y-axis range
    if ycustom and ymax_usr > ymin_usr:
        y_range  = [ymin_usr, ymax_usr]
        x_range  = [ymin_usr, ymax_usr]   # for horizontal
    else:
        y_range  = [0, max_val * 1.20]
        x_range  = [0, max_val * 1.25]

    base = dict(
        height=fig_h,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", size=fontsize),
        margin=dict(l=70, r=30, t=30, b=70),
        showlegend=False,
    )
    ax_common = dict(
        showgrid=True, gridcolor="#EBEBEB", gridwidth=max(0.5, lw * 0.5),
        zeroline=True, zerolinecolor="#BBBBBB", zerolinewidth=lw,
        showline=True, linecolor="#444444", linewidth=lw,
        ticks="outside", tickcolor="#444444", tickwidth=lw,
        tickfont=dict(size=fontsize),
    )

    if chart_type == "竖向柱状图":
        fig = go.Figure(go.Bar(
            x=labels, y=vals, marker_color=bar_colors,
            marker_line_width=0, text=text, textposition="outside",
            textfont=dict(size=fontsize), cliponaxis=False,
        ))
        fig.update_layout(**base,
            yaxis=dict(title=dict(text=y_label, font=dict(size=fontsize)),
                       range=y_range, **ax_common),
            xaxis=dict(showgrid=False, showline=True, linecolor="#444444",
                       linewidth=lw, ticks="outside", tickcolor="#444444",
                       tickwidth=lw, tickfont=dict(size=fontsize)),
        )

    elif chart_type == "横向柱状图":
        fig = go.Figure(go.Bar(
            y=labels, x=vals, orientation="h",
            marker_color=bar_colors, marker_line_width=0,
            text=text, textposition="outside",
            textfont=dict(size=fontsize), cliponaxis=False,
        ))
        fig.update_layout(**base,
            xaxis=dict(title=dict(text=y_label, font=dict(size=fontsize)),
                       range=x_range, **ax_common),
            yaxis=dict(showgrid=False, showline=True, linecolor="#444444",
                       linewidth=lw, ticks="outside", tickcolor="#444444",
                       tickwidth=lw, tickfont=dict(size=fontsize),
                       autorange="reversed"),
        )

    else:  # 棒棒糖图
        fig = go.Figure()
        for i, (lbl, val) in enumerate(zip(labels, vals)):
            c = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=[lbl, lbl], y=[0, val], mode="lines",
                line=dict(color=c, width=lw * 1.5), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[lbl], y=[val],
                mode="markers+text" if show_val else "markers",
                marker=dict(color=c, size=max(8, fontsize),
                            line=dict(color="white", width=lw)),
                text=[f"{val:.3g}"] if show_val else None,
                textposition="top center",
                textfont=dict(size=fontsize),
                showlegend=False,
            ))
        fig.update_layout(**base,
            yaxis=dict(title=dict(text=y_label, font=dict(size=fontsize)),
                       range=y_range, **ax_common),
            xaxis=dict(showgrid=False, showline=True, linecolor="#444444",
                       linewidth=lw, ticks="outside", tickcolor="#444444",
                       tickwidth=lw, tickfont=dict(size=fontsize)),
        )

    return fig


def show_quant(df, uploaded, extra_sheets=None, dl_key="dl_main"):
    cols = [c for c in ["Lane","Group","Band","Area","Mean","Min","Max","IntDen","RawIntDen"]
            if c in df.columns]
    st.subheader(f"定量结果（{len(df)} 个条带）")

    st.dataframe(df[cols], use_container_width=True, hide_index=True)
    sheets = {"定量结果": df[cols]}
    if extra_sheets:
        sheets.update(extra_sheets)
    base = uploaded.name.rsplit(".",1)[0] if uploaded else "wb"
    excel_download(sheets, base, key=dl_key)

    if len(df) > 1:
        grp_col = "Group" if "Group" in df.columns else "Lane"
        series  = df.groupby(grp_col)["IntDen"].sum()
        st.plotly_chart(make_chart(series, y_label="IntDen"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
if mode == "单膜分析":
    uploaded = st.file_uploader("上传 WB 图片",
                                type=["jpg","jpeg","png","tif","tiff","bmp"])
    img = load_image(uploaded)
    if img is None:
        st.info("请上传图片后开始分析。")
        st.stop()

    crop = crop_selector(img, key="s1")
    if crop is None:
        st.stop()

    cx0, cy0, cx1, cy1 = crop
    img_c = img[cy0:cy1, cx0:cx1]

    st.divider()
    st.markdown("**第二步：框选定量蛋白**")
    rois_local = box_selector(img_c, key="s1_bands")
    if not rois_local:
        st.stop()

    rois  = offset_rois(rois_local, cx0, cy0)
    names = lane_names_input(len(rois), key="s1_bands")
    with st.spinner("分析中…"):
        _, df, _ = analyze_rois(img, rois, radius=radius, force_type=get_force_type())
    df = apply_names(df, names)
    st.divider()
    show_annotated(img, [(rois, (0,220,80), "L")])
    st.divider()
    show_quant(df, uploaded, dl_key="dl_s1")


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
        crop_t = crop_selector(img_t, key="t2")
        if crop_t:
            cx0, cy0, cx1, cy1 = crop_t
            img_tc = img_t[cy0:cy1, cx0:cx1]
            st.divider()
            rois_tl = box_selector(img_tc, key="t2_bands", label="目的蛋白：框选条带",
                                   color="#00dc50")
        else:
            rois_tl = []
    with tab_r:
        crop_r = crop_selector(img_r, key="r2")
        if crop_r:
            cx0r, cy0r, cx1r, cy1r = crop_r
            img_rc = img_r[cy0r:cy1r, cx0r:cx1r]
            st.divider()
            rois_rl = box_selector(img_rc, key="r2_bands", label="内参：框选条带",
                                   color="#00aaff")
        else:
            rois_rl = []

    if not rois_tl or not rois_rl:
        st.stop()

    rois_t = offset_rois(rois_tl, crop_t[0], crop_t[1])
    rois_r = offset_rois(rois_rl, crop_r[0], crop_r[1])
    names  = lane_names_input(len(rois_t), key="t2_bands")

    ft = get_force_type()
    with st.spinner("分析中…"):
        _, df_t, _ = analyze_rois(img_t, rois_t, radius=radius, force_type=ft)
        _, df_r, _ = analyze_rois(img_r, rois_r, radius=radius, force_type=ft)
    df_t = apply_names(df_t, names)
    df_r = apply_names(df_r, names)

    st.divider()
    st.subheader("目的蛋白")
    show_annotated(img_t, [(rois_t, (0,220,80), "T")])
    show_quant(df_t, up_t, dl_key="dl_t2_t")
    st.subheader("内参")
    show_annotated(img_r, [(rois_r, (0,170,255), "R")])
    show_quant(df_r, up_r, dl_key="dl_t2_r")
    st.divider()

    lane_options_t2 = sorted(df_t["Lane"].unique().tolist()) if "Lane" in df_t.columns else [1]
    ref_lane_t2 = st.selectbox("以哪条泳道为 1 归一化？", lane_options_t2,
                                index=0, key="ref_lane_t2")
    merged = ratio_table(df_t, df_r, ref_lane=ref_lane_t2)
    if merged is not None:
        st.subheader("对比结果")
        st.dataframe(merged, use_container_width=True, hide_index=True)
        st.caption(f"Normalized_Ratio = 以 Lane {ref_lane_t2} 为 1 归一化")
        idx_col = "Group" if "Group" in merged.columns else "Lane"
        ratio_series = merged.set_index(idx_col)["Normalized_Ratio"]
        st.plotly_chart(make_chart(ratio_series, y_label="Normalized Ratio"),
                        use_container_width=True)
        cols = ["Lane","Group","Band","Area","Mean","IntDen","RawIntDen"]
        base = up_t.name.rsplit(".",1)[0] if up_t else "wb"
        excel_download({
            "目的蛋白": df_t[[c for c in cols if c in df_t.columns]],
            "内参":     df_r[[c for c in cols if c in df_r.columns]],
            "对比结果": merged,
        }, base, key="dl_t2_merged")


else:  # 单膜对比
    uploaded = st.file_uploader("上传 WB 图片（含两条带）",
                                type=["jpg","jpeg","png","tif","tiff","bmp"])
    img = load_image(uploaded)
    if img is None:
        st.info("请上传图片后开始分析。")
        st.stop()

    crop = crop_selector(img, key="s3")
    if crop is None:
        st.stop()

    cx0, cy0, cx1, cy1 = crop
    img_c = img[cy0:cy1, cx0:cx1]

    st.divider()
    st.markdown("**第二步：框选定量蛋白**")
    tab_tgt, tab_ref = st.tabs(["🎯 目的蛋白", "⚖️ 内参"])
    with tab_tgt:
        rois_tl = box_selector(img_c, key="s3t", label="目的蛋白条带",
                               color="#00dc50")
    with tab_ref:
        rois_rl = box_selector(img_c, key="s3r", label="内参条带",
                               color="#00aaff")

    if not rois_tl or not rois_rl:
        st.stop()

    rois_t = offset_rois(rois_tl, cx0, cy0)
    rois_r = offset_rois(rois_rl, cx0, cy0)
    names  = lane_names_input(len(rois_t), key="s3t")

    ft = get_force_type()
    with st.spinner("分析中…"):
        _, df_t, _ = analyze_rois(img, rois_t, radius=radius, force_type=ft)
        _, df_r, _ = analyze_rois(img, rois_r, radius=radius, force_type=ft)
    df_t = apply_names(df_t, names)
    df_r = apply_names(df_r, names)

    st.divider()
    show_annotated(img, [(rois_t,(0,220,80),"T"), (rois_r,(0,170,255),"R")])
    st.divider()

    st.subheader("目的蛋白 — 定量")
    show_quant(df_t, uploaded, dl_key="dl_s3_t")
    st.subheader("内参 — 定量")
    show_quant(df_r, uploaded, dl_key="dl_s3_r")

    cols = ["Lane","Group","Band","Area","Mean","Min","Max","IntDen","RawIntDen"]
    lane_options_s3 = sorted(df_t["Lane"].unique().tolist()) if "Lane" in df_t.columns else [1]
    ref_lane_s3 = st.selectbox("以哪条泳道为 1 归一化？", lane_options_s3,
                                index=0, key="ref_lane_s3")
    merged = ratio_table(df_t, df_r, ref_lane=ref_lane_s3)
    if merged is not None:
        st.subheader("对比结果（目的蛋白 / 内参）")
        st.dataframe(merged, use_container_width=True, hide_index=True)
        st.caption(f"Normalized_Ratio = 以 Lane {ref_lane_s3} 为 1 归一化")
        idx_col = "Group" if "Group" in merged.columns else "Lane"
        ratio_series = merged.set_index(idx_col)["Normalized_Ratio"]
        st.plotly_chart(make_chart(ratio_series, y_label="Normalized Ratio"),
                        use_container_width=True)
        base = uploaded.name.rsplit(".",1)[0] if uploaded else "wb"
        excel_download({
            "目的蛋白": df_t[[c for c in cols if c in df_t.columns]],
            "内参":     df_r[[c for c in cols if c in df_r.columns]],
            "对比结果": merged,
        }, base, key="dl_s3_merged")

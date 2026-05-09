# WB Analyzer — Western Blot 条带灰度值自动定量工具

基于 ImageJ 工作流程的自动化 Western Blot 定量分析网页应用。上传图片即可自动完成背景去除、条带检测和 IntDen 计算，并导出 Excel 结果。

## 功能

- 自动灰度转换 + 背景去除（对应 ImageJ Subtract Background, radius=50）
- 自动检测泳道和条带位置
- 计算 Area、Mean Gray Value、IntDen、RawIntDen
- 结果可视化：原图 / 预处理图 / 标注检测图三图对比
- 一键导出 Excel（含原始数据 + 归一化相对定量两个 sheet）

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/Hz-777/wb-analyzer.git
cd wb-analyzer

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动应用
streamlit run app.py
```

浏览器打开 http://localhost:8501 即可使用。

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| 背景去除半径 | 50 | 对应 ImageJ rolling ball radius |
| 泳道数量 | 自动 | 可手动指定泳道数 |
| 检测灵敏度 | 0.25 | 越低越灵敏，弱条带也能检测到 |

## 对应 ImageJ 步骤

| ImageJ 手动步骤 | 本程序 |
|---|---|
| Image → Type → 8-bit | ✅ 自动 |
| Process → Subtract Background (radius=50, Light bg) | ✅ 自动（可调） |
| Edit → Invert | ✅ 融合在背景去除中 |
| 手动圈选条带 → 按 M 测量 | ✅ 自动检测 |
| 导出 IntDen 到 Excel | ✅ 一键下载 |

## 依赖

- Python 3.10+
- streamlit
- opencv-python-headless
- numpy / scipy / pandas / openpyxl / Pillow

## License

MIT

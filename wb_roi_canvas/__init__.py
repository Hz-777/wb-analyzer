import base64

import cv2
import numpy as np
import streamlit.components.v1 as components

# Always load frontend from GitHub Pages — works on both local and Cloud.
_component_func = components.declare_component(
    "wb_roi_canvas",
    url="https://hz-777.github.io/wb-analyzer/",
)


def wb_roi_canvas(
    img_bgr: np.ndarray,
    display_width: int = 760,
    key: str | None = None,
) -> list[tuple[int, int, int, int]]:
    """Drag-to-draw ROI selector on a WB image.

    Renders an HTML5 canvas inside a Streamlit component iframe.
    Coordinate mapping is done entirely in JavaScript — no Python-side
    scaling guesswork.

    Returns list of (x0, y0, x1, y1) in original image pixel coordinates,
    sorted left-to-right.
    """
    h, w = img_bgr.shape[:2]

    dh = max(1, int(h * display_width / w))
    disp = cv2.resize(img_bgr, (display_width, dh), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 88])
    b64 = base64.b64encode(buf).decode()

    result = _component_func(
        imageB64=b64,
        origWidth=w,
        origHeight=h,
        displayWidth=display_width,
        key=key,
        default=[],
    )

    if not result:
        return []

    rois = [(r["x0"], r["y0"], r["x1"], r["y1"]) for r in result]
    return sorted(rois, key=lambda r: r[0])

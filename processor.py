import cv2
import numpy as np
from scipy.signal import find_peaks
import pandas as pd


def find_gel_bbox(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Auto-detect the membrane/gel bounding box by removing the black border.

    Works for dark-background chemiluminescence images (ChemiDoc etc.) where
    the membrane is a large bright rectangle surrounded by black.
    Returns (x0, y0, x1, y1) in original image coordinates.
    """
    h, w = gray.shape

    # Adaptive threshold: anything brighter than 5% of max is considered "gel"
    thresh_val = max(8, int(gray.max() * 0.05))
    _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # Erode aggressively to kill small calibration markers / corner artifacts
    erode_px = max(15, min(w, h) // 25)
    kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_px, erode_px))
    eroded = cv2.erode(mask, kernel_e, iterations=3)

    if eroded.sum() == 0:
        # Fallback: use full image
        return (0, 0, w, h)

    # Dilate back to restore the membrane extent
    kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_px * 2, erode_px * 2))
    restored = cv2.dilate(eroded, kernel_d, iterations=2)

    # Bounding box of the restored region
    coords = cv2.findNonZero(restored)
    if coords is None:
        return (0, 0, w, h)

    x, y, bw, bh = cv2.boundingRect(coords)
    # Small safety margin
    pad = 5
    return (
        max(0, x - pad),
        max(0, y - pad),
        min(w, x + bw + pad),
        min(h, y + bh + pad),
    )


def preprocess(img_bgr: np.ndarray, radius: int = 50) -> np.ndarray:
    """Grayscale → background subtraction (light-background mode, like ImageJ steps 3-4).

    Result: bands become bright pixels on dark background.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    kernel_size = int(radius * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)

    # background - original: dark bands become bright, bright bg → 0
    diff = cv2.subtract(background, gray)
    return diff


def detect_lanes(
    enhanced: np.ndarray,
    n_lanes: int | None = None,
    sensitivity: float = 0.3,
) -> list[tuple[int, int]]:
    """Detect vertical lane boundaries via horizontal intensity projection."""
    col_profile = enhanced.sum(axis=0).astype(float)

    smooth_k = max(3, len(col_profile) // 50)
    col_smooth = np.convolve(col_profile, np.ones(smooth_k) / smooth_k, mode="same")

    height = col_smooth.max() * sensitivity
    distance = max(10, len(col_smooth) // 20)
    prominence = col_smooth.max() * sensitivity * 0.5

    peaks, props = find_peaks(col_smooth, height=height, distance=distance, prominence=prominence)

    if n_lanes is not None and len(peaks) != n_lanes:
        if len(peaks) == 0:
            step = enhanced.shape[1] // n_lanes
            return [(i * step, (i + 1) * step) for i in range(n_lanes)]
        prom = props.get("prominences", col_smooth[peaks])
        idx = np.argsort(prom)[::-1][:n_lanes]
        peaks = np.sort(peaks[idx])

    if len(peaks) == 0:
        return [(0, enhanced.shape[1])]

    w = enhanced.shape[1]
    boundaries = []
    for i, pk in enumerate(peaks):
        left = (peaks[i - 1] + pk) // 2 if i > 0 else max(0, pk - (peaks[1] - peaks[0]) // 2 if len(peaks) > 1 else pk // 2)
        right = (pk + peaks[i + 1]) // 2 if i < len(peaks) - 1 else min(w, pk + (pk - peaks[i - 1]) // 2 if i > 0 else w)
        boundaries.append((int(left), int(right)))

    return boundaries


def detect_bands(
    lane_enhanced: np.ndarray,
    sensitivity: float = 0.3,
) -> list[tuple[int, int]]:
    """Detect horizontal band positions within a single lane."""
    row_profile = lane_enhanced.sum(axis=1).astype(float)

    smooth_k = max(3, len(row_profile) // 30)
    row_smooth = np.convolve(row_profile, np.ones(smooth_k) / smooth_k, mode="same")

    threshold = row_smooth.max() * sensitivity
    distance = max(5, len(row_smooth) // 15)

    peaks, _ = find_peaks(row_smooth, height=threshold, distance=distance, prominence=threshold * 0.3)

    if len(peaks) == 0:
        return []

    h = lane_enhanced.shape[0]
    half = distance // 2
    return [(int(max(0, pk - half)), int(min(h, pk + half))) for pk in peaks]


def analyze_rois(
    img_bgr: np.ndarray,
    rois: list[tuple[int, int, int, int]],
    radius: int = 50,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Measure user-drawn rectangular ROIs directly (manual lane selection mode).

    rois: list of (x0, y0, x1, y1) in original image coordinates, one per lane.
    Sorted left-to-right automatically so Lane 1 = leftmost lane.
    Each rectangle is measured as-is — no further band detection needed.
    """
    enhanced = preprocess(img_bgr, radius)
    annotated = img_bgr.copy()
    rows = []

    for i, (x0, y0, x1, y1) in enumerate(sorted(rois, key=lambda r: r[0])):
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        metrics = measure_roi(enhanced, x0, x1, y0, y1)
        rows.append({
            "Lane": i + 1,
            "Band": 1,
            "X_start": x0, "X_end": x1,
            "Y_start": y0, "Y_end": y1,
            **metrics,
        })
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 200, 80), 2)
        cv2.putText(annotated, f"L{i + 1}", (x0 + 4, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 80), 2)

    return annotated, pd.DataFrame(rows)


def measure_roi(enhanced: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> dict:
    """Compute Area, Mean, Min, Max, IntDen, RawIntDen for a rectangular ROI."""
    roi = enhanced[y0:y1, x0:x1].astype(float)
    area = roi.size
    mean_val = roi.mean() if area > 0 else 0.0
    return {
        "Area": int(area),
        "Mean": round(float(mean_val), 3),
        "Min": round(float(roi.min()) if area > 0 else 0.0, 1),
        "Max": round(float(roi.max()) if area > 0 else 0.0, 1),
        "IntDen": round(float(area * mean_val), 1),
        "RawIntDen": round(float(roi.sum()), 1),
    }


def analyze(
    img_bgr: np.ndarray,
    radius: int = 50,
    n_lanes: int | None = None,
    sensitivity: float = 0.3,
    n_bands_per_lane: int = 1,
    auto_crop: bool = True,
) -> tuple[np.ndarray, pd.DataFrame, tuple[int, int, int, int]]:
    """Full pipeline. Returns (annotated_full_image, results_df, gel_bbox).

    n_bands_per_lane: how many bands to keep per lane, ranked by IntDen.
        1 = strongest only (single-protein / dual-membrane mode)
        2 = top-2 bands (same-membrane target+reference mode)
        0 = keep all detected bands
    auto_crop: detect and crop to the membrane region before analysis.
    """
    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if auto_crop:
        gx0, gy0, gx1, gy1 = find_gel_bbox(gray_full)
        img_crop = img_bgr[gy0:gy1, gx0:gx1]
    else:
        gx0, gy0, gx1, gy1 = 0, 0, img_bgr.shape[1], img_bgr.shape[0]
        img_crop = img_bgr

    enhanced = preprocess(img_crop, radius)
    lanes = detect_lanes(enhanced, n_lanes=n_lanes, sensitivity=sensitivity)

    annotated = img_bgr.copy()
    rows = []

    for lane_idx, (x0, x1) in enumerate(lanes):
        lane_img = enhanced[:, x0:x1]
        bands = detect_bands(lane_img, sensitivity=sensitivity)

        if not bands:
            bands = [(0, enhanced.shape[0])]

        candidates = []
        for y0, y1 in bands:
            metrics = measure_roi(enhanced, x0, x1, y0, y1)
            candidates.append((y0, y1, metrics))

        # Sort by IntDen descending, then keep top N (sorted back by Y position)
        if n_bands_per_lane > 0 and len(candidates) > n_bands_per_lane:
            candidates = sorted(candidates, key=lambda c: c[2]["IntDen"], reverse=True)[:n_bands_per_lane]
        # Re-sort by vertical position so Band 1 = top band, Band 2 = bottom band
        candidates = sorted(candidates, key=lambda c: c[0])

        # Color scheme: green=single band, cyan=top-2 mode
        color = (0, 200, 80) if n_bands_per_lane == 1 else (0, 220, 220)

        for band_idx, (y0, y1, metrics) in enumerate(candidates):
            # Translate coordinates back to full image space
            abs_x0, abs_x1 = gx0 + x0, gx0 + x1
            abs_y0, abs_y1 = gy0 + y0, gy0 + y1

            rows.append({
                "Lane": lane_idx + 1,
                "Band": band_idx + 1,
                "X_start": abs_x0,
                "X_end": abs_x1,
                "Y_start": abs_y0,
                "Y_end": abs_y1,
                **metrics,
            })

            cv2.rectangle(annotated, (abs_x0, abs_y0), (abs_x1, abs_y1), color, 2)
            cv2.putText(
                annotated, f"L{lane_idx + 1}",
                (abs_x0 + 4, abs_y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

    # Draw gel bounding box in blue so user can verify the crop
    if auto_crop:
        cv2.rectangle(annotated, (gx0, gy0), (gx1, gy1), (255, 120, 0), 2)

    df = pd.DataFrame(rows)
    return annotated, df, (gx0, gy0, gx1, gy1)

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


def _col_projection(enhanced: np.ndarray) -> np.ndarray:
    """Column-wise intensity sum focused on the band rows (top-40% signal rows).

    Ignoring background rows keeps the inter-lane gap signal sharp.
    """
    row_sums = enhanced.sum(axis=1).astype(float)
    band_mask = row_sums >= np.percentile(row_sums, 60)
    src = enhanced[band_mask] if band_mask.sum() > 3 else enhanced
    return src.sum(axis=0).astype(float)


def _detrend(profile: np.ndarray, w: int) -> np.ndarray:
    """Two-scale detrend: remove slow background drift, keep sharp lane peaks."""
    norm   = profile / (profile.max() + 1e-9)
    k_fine = max(3, w // 60)
    k_base = max(3, w // 6)          # wide baseline — must span several lanes
    smooth = np.convolve(norm, np.ones(k_fine) / k_fine, mode="same")
    base   = np.convolve(norm, np.ones(k_base) / k_base, mode="same")
    return np.clip(smooth - base * 0.35, 0, None)


def _find_peaks_adaptive(
    profile: np.ndarray,
    n_target: int | None,
    sensitivity: float,
) -> np.ndarray:
    """Peak detection with two separate strategies for auto vs. forced mode.

    Auto mode (n_target=None):
        Single stable pass — avoids over-detection by keeping minimum distance
        proportional to the expected lane count.

    Forced mode (n_target given):
        Progressive search — gradually loosens distance + height thresholds
        until exactly n_target peaks are found, then selects top-N by prominence.
    """
    w       = len(profile)
    col_max = profile.max()
    if col_max == 0:
        return np.array([], dtype=int)

    signal = _detrend(profile, w)
    sig_max = signal.max()
    if sig_max == 0:
        return np.array([], dtype=int)

    if n_target is None:
        # ── Auto mode: single conservative pass ───────────────────────────
        dist  = max(5, w // 30)
        h_thr = sig_max * sensitivity
        pks, _ = find_peaks(signal, height=h_thr, distance=dist,
                             prominence=h_thr * 0.3)
        return pks

    # ── Forced mode: progressively relax until we get n_target peaks ──────
    for dist_factor in [1.0, 0.70, 0.50, 0.35, 0.22, 0.13]:
        dist = max(3, int(w * dist_factor / n_target))
        for sens_f in [1.0, 0.75, 0.55, 0.40, 0.25, 0.15]:
            h_thr = sig_max * sensitivity * sens_f
            pks, props = find_peaks(signal, height=h_thr, distance=dist,
                                    prominence=h_thr * 0.2)
            if len(pks) >= n_target:
                prom = props.get("prominences", signal[pks])
                idx  = np.argsort(prom)[::-1][:n_target]
                return np.sort(pks[idx])

    # Still not enough — return whatever we have
    return np.array([], dtype=int)


def _uniform_peaks(
    profile: np.ndarray,
    detected: np.ndarray,
    n_lanes: int,
) -> np.ndarray:
    """When we can't find n_lanes peaks, enforce uniform spacing.

    Strategy (in priority order):
      1. If ≥2 peaks detected: extrapolate using median spacing.
      2. Otherwise: divide the high-signal region evenly.
    """
    w = len(profile)
    if len(detected) >= 2:
        spacing = int(np.median(np.diff(detected)))
        center  = int(np.mean(detected))
        start   = center - spacing * (n_lanes - 1) // 2
        peaks   = np.array([start + i * spacing for i in range(n_lanes)])
        return np.clip(peaks, 0, w - 1)

    # Fall back to signal-region division
    sig_cols = np.where(profile > profile.max() * 0.08)[0]
    x0 = int(sig_cols[0])  if len(sig_cols) else 0
    x1 = int(sig_cols[-1]) if len(sig_cols) else w - 1
    spacing = (x1 - x0) // n_lanes
    return np.array([x0 + spacing // 2 + i * spacing for i in range(n_lanes)])


def _peaks_to_boundaries(
    peaks: np.ndarray, w: int, gap_frac: float = 0.15
) -> list[tuple[int, int]]:
    """Convert lane-center x positions to equal-width (left, right) pairs with gaps.

    All boxes have the same width = spacing * (1 - gap_frac).
    Gaps between adjacent boxes = spacing * gap_frac  (default 15 %).
    This keeps Area identical across lanes, which is required for fair IntDen comparison.
    """
    n = len(peaks)
    if n == 0:
        return []
    if n == 1:
        half = w // 4
        return [(max(0, peaks[0] - half), min(w, peaks[0] + half))]

    spacing = int(np.median(np.diff(peaks)))
    half = max(4, int(spacing * (1.0 - gap_frac) / 2))

    return [(int(max(0, pk - half)), int(min(w, pk + half))) for pk in peaks]


def detect_lanes(
    enhanced: np.ndarray,
    n_lanes: int | None = None,
    sensitivity: float = 0.3,
) -> list[tuple[int, int]]:
    """Detect vertical lane boundaries with multi-scale adaptive peak finding.

    Steps:
      1. Project only band rows → sharp inter-lane gaps.
      2. Detrend with two-scale smoothing to remove background drift.
      3. Progressively relax distance + sensitivity until n_lanes peaks found.
      4. If still short: extrapolate from detected spacing (uniform enforcement).
    """
    w = enhanced.shape[1]
    col_profile = _col_projection(enhanced)

    if col_profile.max() == 0:
        return [(0, w)]

    peaks = _find_peaks_adaptive(col_profile, n_lanes, sensitivity)

    if n_lanes is not None and len(peaks) != n_lanes:
        peaks = _uniform_peaks(col_profile, peaks, n_lanes)

    if len(peaks) == 0:
        return [(0, w)]

    return _peaks_to_boundaries(peaks, w)


def detect_bands(
    lane_enhanced: np.ndarray,
    sensitivity: float = 0.3,
) -> list[tuple[int, int]]:
    """Detect horizontal band positions using FWHM-style boundary expansion.

    For each peak:
      - Walk outward from the peak until signal drops below 30 % of peak value
        OR rises again (entering the next band's territory).
      - Add a 1-pixel padding so the full band is included.
    """
    row_profile = lane_enhanced.sum(axis=1).astype(float)
    h = lane_enhanced.shape[0]

    # Two-scale smooth: fine for peaks, coarse for baseline removal
    k_fine = max(3, h // 30)
    k_base = max(3, h // 8)
    smooth = np.convolve(row_profile, np.ones(k_fine) / k_fine, mode="same")
    base   = np.convolve(row_profile, np.ones(k_base) / k_base, mode="same")
    signal = np.clip(smooth - base * 0.3, 0, None)

    if signal.max() == 0:
        return []

    threshold = signal.max() * sensitivity
    distance  = max(5, h // 15)
    peaks, _  = find_peaks(signal, height=threshold, distance=distance,
                            prominence=threshold * 0.25)

    if len(peaks) == 0:
        return []

    bands = []
    for pk in peaks:
        pk_val     = signal[pk]
        edge_thr   = max(threshold * 0.35, pk_val * 0.30)

        # Walk up (decreasing y)
        y0 = pk
        while y0 > 0 and signal[y0 - 1] >= edge_thr and signal[y0 - 1] <= signal[y0]:
            y0 -= 1

        # Walk down (increasing y)
        y1 = pk
        while y1 < h - 1 and signal[y1 + 1] >= edge_thr and signal[y1 + 1] <= signal[y1]:
            y1 += 1

        # 1-pixel padding
        y0 = max(0,     y0 - 1)
        y1 = min(h - 1, y1 + 1)

        bands.append((int(y0), int(y1 + 1)))

    return bands


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
    skip_first_lane: bool = False,
    skip_last_lane: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, tuple[int, int, int, int]]:
    """Full pipeline. Returns (annotated_full_image, results_df, gel_bbox).

    n_bands_per_lane: how many bands to keep per lane, ranked by IntDen.
        1 = strongest only (single-protein / dual-membrane mode)
        2 = top-2 bands (same-membrane target+reference mode)
        0 = keep all detected bands
    skip_first_lane: drop the leftmost detected lane (usually the marker).
    skip_last_lane:  drop the rightmost detected lane (marker on right side).
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

    # Drop marker lanes before processing
    if skip_first_lane and len(lanes) > 1:
        lanes = lanes[1:]
    if skip_last_lane and len(lanes) > 1:
        lanes = lanes[:-1]

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

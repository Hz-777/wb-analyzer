"""
processor.py — Western Blot quantification pipeline.

Detection pipeline (operates entirely on the bright-band enhanced image):
  1. detect_image_type  → 'light' or 'dark' background
  2. preprocess         → bands = BRIGHT pixels on BLACK background
  3. find_gel_bbox      → crop to membrane, discard black border
  4. _col_projection    → sum columns of enhanced → lane x-peaks
  5. _row_projection    → sum rows within each lane → band y-peaks
  6. FWHM walk          → accurate band boundaries from signal drop-off
  7. measure_roi        → Area / Mean / IntDen on enhanced image
"""

import cv2
import numpy as np
from scipy.signal import find_peaks
import pandas as pd


# ════════════════════════ image type ══════════════════════════════════

def detect_image_type(gray: np.ndarray) -> str:
    """'light' = white/grey background with dark bands; 'dark' = black background.

    Uses p75 so ChemiDoc images (grey gel crop, median ~100) are classified as
    'light' even though their overall median is low due to the black border.
    """
    return "light" if float(np.percentile(gray, 75)) > 60 else "dark"


# ════════════════════════ preprocessing ══════════════════════════════

def preprocess(img_bgr: np.ndarray, radius: int = 50) -> np.ndarray:
    """Return enhanced image: bands = BRIGHT pixels on BLACK background.

    Light-bg (white/grey bg, dark bands):
        background = morphological dilation  →  bg − gray  (dark→bright)
    Dark-bg (fluorescence, bright bands):
        background = morphological opening   →  gray − bg  (bright stays)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_type = detect_image_type(gray)

    k      = int(radius * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    if img_type == "light":
        bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        return cv2.subtract(bg, gray)
    else:
        bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        return cv2.subtract(gray, bg)


# ════════════════════════ gel bounding box ════════════════════════════

def find_gel_bbox(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Detect membrane region; discard black border and L-bracket markers.

    Low threshold (5 % of max) keeps dark bands inside the membrane.
    Erosion destroys thin L-brackets; largest connected component = membrane.
    """
    h, w = gray.shape
    src  = gray if detect_image_type(gray) == "dark" else cv2.bitwise_not(gray)

    thresh = max(8, int(src.max() * 0.05))
    _, mask = cv2.threshold(src, thresh, 255, cv2.THRESH_BINARY)

    px     = max(12, min(w, h) // 30)
    eroded = cv2.erode(mask,
                       cv2.getStructuringElement(cv2.MORPH_RECT, (px, px)),
                       iterations=3)
    if eroded.sum() == 0:
        return (0, 0, w, h)

    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    if n_lab <= 1:
        return (0, 0, w, h)

    largest  = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    single   = np.where(labels == largest, 255, 0).astype(np.uint8)
    restored = cv2.dilate(single,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (px * 2, px * 2)),
                          iterations=2)

    pts = cv2.findNonZero(restored)
    if pts is None:
        return (0, 0, w, h)

    x, y, bw, bh = cv2.boundingRect(pts)
    p = 4
    return (max(0, x - p), max(0, y - p), min(w, x + bw + p), min(h, y + bh + p))


# ════════════════════════ projection-based detection ══════════════════

def _smooth(arr: np.ndarray, k: int) -> np.ndarray:
    k = max(1, k | 1)          # ensure odd
    return np.convolve(arr, np.ones(k) / k, mode="same")


def _find_n_peaks(profile: np.ndarray, n: int, sensitivity: float) -> np.ndarray:
    """Return exactly n peaks from profile, sorted by position.

    Progressive search: relax distance + height thresholds until n peaks found,
    then keep the n most prominent ones.
    """
    if profile.max() == 0 or n == 0:
        return np.array([], dtype=int)

    norm = profile / profile.max()
    size = len(norm)

    for dist_f in [0.8, 0.6, 0.4, 0.25, 0.15, 0.08]:
        dist = max(3, int(size * dist_f / n))
        for s_f in [1.0, 0.7, 0.5, 0.35, 0.2, 0.1]:
            h_thr = sensitivity * s_f
            pks, props = find_peaks(norm, height=h_thr, distance=dist,
                                    prominence=h_thr * 0.2)
            if len(pks) >= n:
                prom = props.get("prominences", norm[pks])
                idx  = np.argsort(prom)[::-1][:n]
                return np.sort(pks[idx])

    return np.array([], dtype=int)


def _find_auto_peaks(profile: np.ndarray, sensitivity: float) -> np.ndarray:
    """Auto-detect peaks (no target count) with progressively relaxed thresholds.

    Tries multiple height/prominence levels so that both strong and weak lane
    profiles are handled.  Returns on the first pass that yields ≥ 2 peaks.
    """
    if profile.max() == 0:
        return np.array([], dtype=int)
    norm = profile / profile.max()
    size = len(norm)
    dist = max(5, size // 20)

    for h_mult in [1.0, 0.7, 0.5, 0.35, 0.2, 0.12, 0.06]:
        h_thr = sensitivity * h_mult
        pks, _ = find_peaks(norm, height=h_thr, distance=dist,
                            prominence=max(0.015, h_thr * 0.25))
        if len(pks) >= 2:
            return pks

    return np.array([], dtype=int)


def _merge_doublet_peaks(peaks: np.ndarray) -> np.ndarray:
    """Merge adjacent peak pairs that are anomalously close (< 55 % of median spacing)."""
    if len(peaks) < 3:
        return peaks
    diffs  = np.diff(peaks.astype(float))
    med    = float(np.median(diffs))
    merged: list[int] = []
    i = 0
    while i < len(peaks):
        if i + 1 < len(peaks) and diffs[i] < med * 0.55:
            merged.append(int((peaks[i] + peaks[i + 1]) // 2))
            i += 2
        else:
            merged.append(int(peaks[i]))
            i += 1
    return np.array(merged, dtype=int)


def _lane_regions_from_profile(
    col_profile: np.ndarray,
    gap_frac: float = 0.15,
) -> list[tuple[int, int]]:
    """Segment lanes from the column projection using connected-region analysis.

    Each lane is a vertical rectangle, so its columns form a connected block
    of high signal separated from adjacent lanes by near-zero gutter columns.
    Otsu's method finds the lane/gutter threshold automatically; morphological
    closing fills narrow within-lane dips before segmentation.

    Returns a list of (x_left, x_right) lane boundaries (with gap applied).
    """
    size = len(col_profile)
    if size == 0 or col_profile.max() == 0:
        return [(0, size - 1)]

    norm = col_profile / col_profile.max()

    # Otsu threshold: separates lane-signal columns from gutter columns
    u8  = (norm * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = np.clip(float(otsu_val) / 255.0, 0.08, 0.60)

    binary = (norm >= thr).astype(np.uint8)

    # Morphological close: fills within-lane gaps narrower than ~lane_width/5
    close_w = max(3, size // 60)
    binary  = cv2.morphologyEx(
        binary.reshape(1, -1),
        cv2.MORPH_CLOSE,
        np.ones((1, close_w), np.uint8),
    ).reshape(-1)

    # Extract connected runs; discard tiny noise regions
    min_w   = max(5, size // 40)
    regions: list[tuple[int, int]] = []
    i = 0
    while i < size:
        if binary[i]:
            j = i
            while j < size and binary[j]:
                j += 1
            if j - i >= min_w:
                regions.append((i, j - 1))
            i = j
        else:
            i += 1

    if not regions:
        return []

    # Convert each connected region → equal-width lane box with gap
    bounds = []
    for x0, x1 in regions:
        cx   = (x0 + x1) // 2
        half = max(3, int((x1 - x0 + 1) * (1.0 - gap_frac) / 2))
        bounds.append((max(0, cx - half), min(size - 1, cx + half)))

    return bounds


def _peaks_to_equal_boundaries(peaks: np.ndarray, size: int,
                                gap_frac: float = 0.15) -> list[tuple[int, int]]:
    """Convert peak x-positions → equal-width (left, right) with gap_frac gap."""
    n = len(peaks)
    if n == 0:
        return []
    if n == 1:
        hw = size // 4
        return [(max(0, peaks[0] - hw), min(size, peaks[0] + hw))]

    spacing = int(np.median(np.diff(peaks)))
    half    = max(3, int(spacing * (1.0 - gap_frac) / 2))
    return [(max(0, int(p) - half), min(size, int(p) + half)) for p in peaks]


def _uniform_split(x0: int, x1: int, n: int, gap_frac: float = 0.15,
                   img_w: int = 9999) -> list[tuple[int, int]]:
    """Divide [x0, x1] into n equal slots with gap_frac gap between each."""
    sp   = (x1 - x0) / n
    half = max(3, int(sp * (1.0 - gap_frac) / 2))
    return [
        (max(0, int(x0 + i * sp + sp / 2) - half),
         min(img_w, int(x0 + i * sp + sp / 2) + half))
        for i in range(n)
    ]


def _band_y_boundaries(lane_enhanced: np.ndarray,
                       sensitivity: float,
                       n_bands: int = 0) -> list[tuple[int, int]]:
    """Detect band y-positions by row projection on the bright-band lane slice.

    For each peak in the row profile:
      • Walk outward from the peak until signal drops below 30 % of peak value
        OR starts rising (entering the next band's territory).
      • Add 1-pixel padding.

    n_bands=0 means auto (keep all detected).
    n_bands>0 keeps top-n by integrated row signal.
    """
    h = lane_enhanced.shape[0]
    if h == 0:
        return [(0, 1)]

    row_profile = lane_enhanced.sum(axis=1).astype(float)
    k           = max(3, h // 25)
    smooth_prof = _smooth(row_profile, k)

    if smooth_prof.max() == 0:
        return [(0, h)]

    norm = smooth_prof / smooth_prof.max()
    dist = max(4, h // 12)

    if n_bands > 0:
        peaks = _find_n_peaks(norm, n_bands, sensitivity)
    else:
        peaks = _find_auto_peaks(norm, sensitivity)

    if len(peaks) == 0:
        # Fallback: use the single brightest row region
        center = int(np.argmax(smooth_prof))
        peaks  = np.array([center])

    bands = []
    for pk in peaks:
        pk_val   = norm[pk]
        # Stop when signal drops below 20 % of peak (wider than the old 30 %)
        # OR when it starts rising again (entering the next band).
        edge_thr = max(sensitivity * 0.10, pk_val * 0.20)

        y0 = int(pk)
        while y0 > 0 and norm[y0 - 1] >= edge_thr and norm[y0 - 1] <= norm[y0]:
            y0 -= 1

        y1 = int(pk)
        while y1 < h - 1 and norm[y1 + 1] >= edge_thr and norm[y1 + 1] <= norm[y1]:
            y1 += 1

        # Extra margin: ≥3 px or 25 % of the detected half-width, whichever larger
        pad = max(3, (y1 - y0) // 4)
        y0  = max(0,     y0 - pad)
        y1  = min(h - 1, y1 + pad)
        bands.append((y0, y1 + 1))

    # Keep top-n by integrated signal (when forced)
    if n_bands > 0 and len(bands) > n_bands:
        scored = sorted(bands, key=lambda b: row_profile[b[0]:b[1]].sum(), reverse=True)
        bands  = sorted(scored[:n_bands], key=lambda b: b[0])

    return bands if bands else [(0, h)]


# ════════════════════════ measurement ════════════════════════════════

def measure_roi(enhanced: np.ndarray, x0: int, x1: int,
                y0: int, y1: int) -> dict:
    """Area / Mean / Min / Max / IntDen / RawIntDen on the enhanced image."""
    roi      = enhanced[y0:y1, x0:x1].astype(float)
    area     = roi.size
    mean_val = float(roi.mean()) if area > 0 else 0.0
    return {
        "Area":      int(area),
        "Mean":      round(mean_val, 3),
        "Min":       round(float(roi.min()) if area > 0 else 0.0, 1),
        "Max":       round(float(roi.max()) if area > 0 else 0.0, 1),
        "IntDen":    round(area * mean_val, 1),
        "RawIntDen": round(float(roi.sum()), 1),
    }


# ════════════════════════ manual ROI mode ════════════════════════════

def analyze_rois(
    img_bgr: np.ndarray,
    rois: list[tuple[int, int, int, int]],
    radius: int = 50,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Measure user-defined rectangular ROIs directly (manual mode)."""
    enhanced  = preprocess(img_bgr, radius)
    annotated = img_bgr.copy()
    rows: list[dict] = []

    for i, (x0, y0, x1, y1) in enumerate(sorted(rois, key=lambda r: r[0])):
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        metrics = measure_roi(enhanced, x0, x1, y0, y1)
        rows.append({"Lane": i + 1, "Band": 1,
                     "X_start": x0, "X_end": x1,
                     "Y_start": y0, "Y_end": y1, **metrics})
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 200, 80), 2)
        cv2.putText(annotated, f"L{i + 1}", (x0 + 4, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 80), 2)

    return annotated, pd.DataFrame(rows), enhanced


# ════════════════════════ full pipeline ══════════════════════════════

def analyze(
    img_bgr: np.ndarray,
    radius: int = 50,
    n_lanes: int | None = None,
    sensitivity: float = 0.3,
    n_bands_per_lane: int = 1,
    auto_crop: bool = True,
    skip_first_lane: bool = False,
    skip_last_lane: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, tuple[int, int, int, int], np.ndarray]:
    """Projection-based WB quantification pipeline.

    All detection runs on the bright-band enhanced image:
      • Column projection  → lane x-positions
      • Row projection     → band y-positions (FWHM boundaries)
      • measure_roi        → IntDen on enhanced

    Returns (annotated_img, df, gel_bbox, enhanced_crop).
    """
    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── 1. Crop to gel membrane ───────────────────────────────────────
    if auto_crop:
        gx0, gy0, gx1, gy1 = find_gel_bbox(gray_full)
        img_crop = img_bgr[gy0:gy1, gx0:gx1]
    else:
        gx0, gy0 = 0, 0
        gx1, gy1 = img_bgr.shape[1], img_bgr.shape[0]
        img_crop  = img_bgr

    # ── 2. Preprocess: dark bands → bright on black ───────────────────
    enhanced = preprocess(img_crop, radius)
    ch, cw   = enhanced.shape

    # ── 3. Lane x-detection via column projection ─────────────────────
    # Use the full column sum so that valleys between lanes are preserved even
    # when bands span the whole gel height.  Sharpening via top-20 % rows is
    # blended in for images where it helps (i.e. adds more peaks, not fewer).
    col_raw = enhanced.sum(axis=0).astype(float)
    col_profile = _smooth(col_raw, max(3, cw // 60))

    # Also try the sharpened profile (only top-20 % brightest rows).
    row_sums = enhanced.sum(axis=1).astype(float)
    if (row_sums > 0).any():
        pct80 = np.percentile(row_sums[row_sums > 0], 80)
        bright_rows = row_sums >= pct80
        if bright_rows.sum() >= 3:
            col_sharp = _smooth(enhanced[bright_rows].sum(axis=0).astype(float),
                                max(3, cw // 60))
            # Prefer whichever profile has more peaks (= better lane separation)
            auto_raw   = _find_auto_peaks(col_profile, sensitivity)
            auto_sharp = _find_auto_peaks(col_sharp,   sensitivity)
            if len(auto_sharp) > len(auto_raw):
                col_profile = col_sharp

    if n_lanes is not None:
        lane_peaks = _find_n_peaks(col_profile, n_lanes, sensitivity)
        if len(lane_peaks) == n_lanes:
            lane_bounds = _peaks_to_equal_boundaries(lane_peaks, cw)
        else:
            # Peaks not found → use extent of significant columns
            sig_cols = np.where(col_profile > col_profile.max() * 0.1)[0]
            x0_sig   = int(sig_cols[0])  if len(sig_cols) else 0
            x1_sig   = int(sig_cols[-1]) if len(sig_cols) else cw
            lane_bounds = _uniform_split(x0_sig, x1_sig, n_lanes, img_w=cw)
    else:
        # Primary: connected-region analysis (exploits the rectangular lane shape)
        lane_bounds = _lane_regions_from_profile(col_profile)
        if len(lane_bounds) < 2:
            # Fallback: peak-based with doublet merging
            lane_peaks  = _find_auto_peaks(col_profile, sensitivity)
            lane_peaks  = _merge_doublet_peaks(lane_peaks)
            lane_bounds = _peaks_to_equal_boundaries(lane_peaks, cw) \
                          if len(lane_peaks) >= 2 else [(0, cw)]

    # ── 4. Skip marker lanes ──────────────────────────────────────────
    if skip_first_lane and len(lane_bounds) > 1:
        lane_bounds = lane_bounds[1:]
    if skip_last_lane and len(lane_bounds) > 1:
        lane_bounds = lane_bounds[:-1]

    # ── 5. Per-lane: row projection → band y-boundaries → measure ─────
    annotated = img_bgr.copy()
    rows: list[dict] = []
    color = (0, 200, 80) if n_bands_per_lane == 1 else (0, 220, 220)

    for lane_idx, (lx0, lx1) in enumerate(lane_bounds):
        lane_enh  = enhanced[:, lx0:lx1]
        n_tgt     = n_bands_per_lane if n_bands_per_lane > 0 else 0
        y_bands   = _band_y_boundaries(lane_enh, sensitivity, n_tgt)

        # Keep top-N by IntDen and re-sort by position
        candidates = []
        for y0b, y1b in y_bands:
            m = measure_roi(enhanced, lx0, lx1, y0b, y1b)
            candidates.append((y0b, y1b, m))

        if n_bands_per_lane > 0 and len(candidates) > n_bands_per_lane:
            candidates = sorted(candidates,
                                key=lambda c: c[2]["IntDen"], reverse=True
                                )[:n_bands_per_lane]
        candidates = sorted(candidates, key=lambda c: c[0])

        for band_idx, (y0b, y1b, metrics) in enumerate(candidates):
            abs_x0 = gx0 + lx0
            abs_x1 = gx0 + lx1
            abs_y0 = gy0 + y0b
            abs_y1 = gy0 + y1b

            rows.append({
                "Lane": lane_idx + 1, "Band": band_idx + 1,
                "X_start": abs_x0, "X_end": abs_x1,
                "Y_start": abs_y0, "Y_end": abs_y1,
                **metrics,
            })
            cv2.rectangle(annotated, (abs_x0, abs_y0), (abs_x1, abs_y1), color, 2)
            cv2.putText(annotated, f"L{lane_idx + 1}",
                        (abs_x0 + 4, abs_y0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if auto_crop:
        cv2.rectangle(annotated, (gx0, gy0), (gx1, gy1), (255, 120, 0), 2)

    return annotated, pd.DataFrame(rows), (gx0, gy0, gx1, gy1), enhanced

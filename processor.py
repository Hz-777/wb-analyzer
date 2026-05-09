"""
processor.py — Contour-based Western Blot quantification pipeline.

Detection core (replacing projection-based approach):
  1. Auto-detect image type (light-bg dark bands / dark-bg bright bands)
  2. Morphological background subtraction → bands = bright on black
  3. Otsu threshold + sensitivity shift → binary band mask
  4. findContours → bounding boxes
  5. Group boxes by x-center → lanes  (equal-width, gapped boundaries)
  6. Within each lane, merge overlapping y-intervals → bands
  7. Measure Area / Mean / IntDen on the enhanced image
"""

import cv2
import numpy as np
import pandas as pd


# ════════════════════════ image-type detection ════════════════════════

def detect_image_type(gray: np.ndarray) -> str:
    """'light' = white/grey background with dark bands; 'dark' = black background."""
    return "light" if float(np.median(gray)) > 127 else "dark"


# ════════════════════════ preprocessing ══════════════════════════════

def preprocess(img_bgr: np.ndarray, radius: int = 50) -> np.ndarray:
    """Return enhanced image: bands = BRIGHT pixels on BLACK background.

    Light-bg: dilation estimates the bright background; subtract gives dark→bright bands.
    Dark-bg:  opening removes bright bands from the image; subtract gives bright bands.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_type = detect_image_type(gray)

    k = int(radius * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    if img_type == "light":
        bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        return cv2.subtract(bg, gray)        # dark bands → bright
    else:
        bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        return cv2.subtract(gray, bg)        # bright bands − dark bg


# ════════════════════════ gel bounding box ════════════════════════════

def find_gel_bbox(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Auto-detect membrane region, removing border / calibration markers."""
    h, w = gray.shape
    src = gray if detect_image_type(gray) == "dark" else cv2.bitwise_not(gray)

    thresh = max(8, int(src.max() * 0.05))
    _, mask = cv2.threshold(src, thresh, 255, cv2.THRESH_BINARY)

    px = max(15, min(w, h) // 25)
    eroded = cv2.erode(mask,
                       cv2.getStructuringElement(cv2.MORPH_RECT, (px, px)),
                       iterations=3)
    if eroded.sum() == 0:
        return (0, 0, w, h)

    restored = cv2.dilate(eroded,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (px * 2, px * 2)),
                          iterations=2)
    pts = cv2.findNonZero(restored)
    if pts is None:
        return (0, 0, w, h)

    x, y, bw, bh = cv2.boundingRect(pts)
    p = 5
    return (max(0, x - p), max(0, y - p), min(w, x + bw + p), min(h, y + bh + p))


# ════════════════════════ contour detection helpers ════════════════════

def _band_mask(enhanced: np.ndarray, sensitivity: float) -> np.ndarray:
    """Binary mask of band pixels.

    Finds the Otsu threshold on the enhanced image, then scales it by
    sensitivity so that weaker bands are caught at lower values.
    sensitivity 0.1 = very permissive, 0.9 = near-Otsu (strict).
    """
    if enhanced.max() == 0:
        return np.zeros(enhanced.shape, dtype=np.uint8)

    otsu_val, _ = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Higher sensitivity → lower threshold → more pixels flagged as bands
    thresh = int(otsu_val * (0.15 + sensitivity * 0.85))
    thresh = max(1, min(254, thresh))

    _, binary = cv2.threshold(enhanced, thresh, 255, cv2.THRESH_BINARY)

    h, w = enhanced.shape
    md = min(h, w)
    k_open  = max(2, md // 80)
    k_close = max(3, md // 30)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open))
    )
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
    )
    return binary


def _find_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """findContours → bounding boxes (x, y, w, h), noise filtered."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    h, w = mask.shape
    min_area = h * w * 0.0003        # must be ≥ 0.03 % of image area

    boxes = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        ar = bw / (bh + 1e-6)
        if ar > 25 or bh / (bw + 1e-6) > 15:   # reject extreme slivers
            continue
        boxes.append((x, y, bw, bh))

    return boxes


def _group_into_lanes(
    boxes: list[tuple[int, int, int, int]],
    n_lanes: int | None,
    img_w: int,
) -> list[list[tuple[int, int, int, int]]]:
    """Group boxes into vertical lane columns by x-center clustering.

    When n_lanes is given: force exactly n_lanes groups by taking the
    (n_lanes-1) largest x-center gaps as split points.
    When auto: split wherever the gap exceeds 1.5 × median gap.
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[0] + b[2] // 2)
    cxs   = [b[0] + b[2] // 2 for b in boxes]

    if len(cxs) == 1:
        return [boxes]

    gaps = [cxs[i + 1] - cxs[i] for i in range(len(cxs) - 1)]

    if n_lanes is not None:
        n_splits = min(max(0, n_lanes - 1), len(gaps))
        split_at = sorted(
            sorted(range(len(gaps)), key=lambda i: gaps[i], reverse=True)[:n_splits]
        )
    else:
        # If spacing between boxes is roughly uniform (CV < 0.4), each box is its
        # own lane — this is the common case for well-separated single-band lanes.
        # If spacing varies widely, large gaps mark the lane boundaries.
        cv = float(np.std(gaps)) / (float(np.mean(gaps)) + 1e-6)
        if cv < 0.4:
            return [[b] for b in boxes]
        med = float(np.median(gaps))
        split_at = [i for i, g in enumerate(gaps) if g > med * 1.5]

    groups, start = [], 0
    for s in split_at:
        groups.append(boxes[start : s + 1])
        start = s + 1
    groups.append(boxes[start:])

    return [g for g in groups if g]


def _lane_boundaries(
    groups: list[list],
    img_w: int,
    gap_frac: float = 0.15,
) -> list[tuple[int, int]]:
    """Equal-width x-boundaries with gap_frac gap between adjacent lanes.

    Using the median inter-center spacing ensures every lane box has
    identical width → identical Area → fair IntDen comparison.
    """
    centers = sorted(
        int(np.mean([b[0] + b[2] // 2 for b in g])) for g in groups
    )
    if len(centers) == 1:
        hw = img_w // 4
        return [(max(0, centers[0] - hw), min(img_w, centers[0] + hw))]

    spacing = int(np.median(np.diff(centers)))
    half = max(4, int(spacing * (1.0 - gap_frac) / 2))

    return [(max(0, c - half), min(img_w, c + half)) for c in centers]


def _bands_from_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    """Convert per-lane box list → merged y-intervals (y0, y1).

    Sorts by y, merges boxes that are within 5 px of each other.
    """
    if not boxes:
        return []

    intervals = sorted((b[1], b[1] + b[3]) for b in boxes)
    merged: list[list[int]] = []
    for y0, y1 in intervals:
        if merged and y0 <= merged[-1][1] + 5:
            merged[-1][1] = max(merged[-1][1], y1)
        else:
            merged.append([y0, y1])

    return [(y0, y1) for y0, y1 in merged]


# ════════════════════════ measurement ════════════════════════════════

def measure_roi(
    enhanced: np.ndarray, x0: int, x1: int, y0: int, y1: int
) -> dict:
    """Area / Mean / Min / Max / IntDen / RawIntDen for a rectangular ROI."""
    roi = enhanced[y0:y1, x0:x1].astype(float)
    area = roi.size
    mean_val = float(roi.mean()) if area > 0 else 0.0
    return {
        "Area":       int(area),
        "Mean":       round(mean_val, 3),
        "Min":        round(float(roi.min()) if area > 0 else 0.0, 1),
        "Max":        round(float(roi.max()) if area > 0 else 0.0, 1),
        "IntDen":     round(area * mean_val, 1),
        "RawIntDen":  round(float(roi.sum()), 1),
    }


# ════════════════════════ manual ROI mode ════════════════════════════

def analyze_rois(
    img_bgr: np.ndarray,
    rois: list[tuple[int, int, int, int]],
    radius: int = 50,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Measure user-drawn rectangular ROIs directly (manual lane selection)."""
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

    return annotated, pd.DataFrame(rows)


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
) -> tuple[np.ndarray, pd.DataFrame, tuple[int, int, int, int]]:
    """Contour-based WB quantification pipeline.

    Returns (annotated_full_image, results_df, gel_bbox).
    """
    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── 1. Auto-crop to membrane region ──────────────────────────────
    if auto_crop:
        gx0, gy0, gx1, gy1 = find_gel_bbox(gray_full)
        img_crop = img_bgr[gy0:gy1, gx0:gx1]
    else:
        gx0, gy0 = 0, 0
        gx1, gy1 = img_bgr.shape[1], img_bgr.shape[0]
        img_crop  = img_bgr

    # ── 2. Preprocess: bands → bright on black ────────────────────────
    enhanced = preprocess(img_crop, radius)
    ch, cw   = enhanced.shape

    # ── 3. Threshold → binary band mask ──────────────────────────────
    mask  = _band_mask(enhanced, sensitivity)

    # ── 4. Find contour boxes & group into lanes ──────────────────────
    boxes  = _find_boxes(mask)
    groups = _group_into_lanes(boxes, n_lanes, cw) if boxes else []

    # ── 5. Skip marker lanes ──────────────────────────────────────────
    if skip_first_lane and len(groups) > 1:
        groups = groups[1:]
    if skip_last_lane and len(groups) > 1:
        groups = groups[:-1]

    # ── 6. Compute equal-width lane x-boundaries ──────────────────────
    if groups:
        boundaries = _lane_boundaries(groups, cw)
    else:
        # Fallback when no contours found: divide image evenly
        n = n_lanes or 1
        sp = cw // n
        boundaries = [(i * sp, (i + 1) * sp) for i in range(n)]
        groups     = [[]] * n

    # ── 7. Per-lane: find bands, measure, annotate ────────────────────
    annotated = img_bgr.copy()
    rows: list[dict] = []
    color = (0, 200, 80) if n_bands_per_lane == 1 else (0, 220, 220)

    for lane_idx, ((lx0, lx1), group) in enumerate(zip(boundaries, groups)):

        # Re-run contours inside this lane's x-slice for accurate y-bounds
        lane_mask = mask[:, lx0:lx1]
        cnts, _ = cv2.findContours(
            lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        h_img  = ch
        w_lane = lx1 - lx0
        min_a  = h_img * w_lane * 0.003   # ≥ 0.3 % of lane area

        lane_boxes = []
        for cnt in cnts:
            if cv2.contourArea(cnt) < min_a:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw / (bh + 1e-6) > 20:
                continue
            lane_boxes.append((x, y, bw, bh))

        # y-intervals from contour boxes
        y_intervals = _bands_from_boxes(lane_boxes) if lane_boxes else [(0, ch)]

        # Measure each band candidate
        candidates: list[tuple[int, int, dict]] = []
        for y0b, y1b in y_intervals:
            m = measure_roi(enhanced, lx0, lx1, y0b, y1b)
            candidates.append((y0b, y1b, m))

        if not candidates:
            m = measure_roi(enhanced, lx0, lx1, 0, ch)
            candidates = [(0, ch, m)]

        # Keep top-N by IntDen, re-sort by y-position
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
                "Lane":    lane_idx + 1,
                "Band":    band_idx + 1,
                "X_start": abs_x0, "X_end": abs_x1,
                "Y_start": abs_y0, "Y_end": abs_y1,
                **metrics,
            })

            cv2.rectangle(annotated, (abs_x0, abs_y0), (abs_x1, abs_y1), color, 2)
            cv2.putText(
                annotated, f"L{lane_idx + 1}",
                (abs_x0 + 4, abs_y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

    # Draw gel bbox in blue
    if auto_crop:
        cv2.rectangle(annotated, (gx0, gy0), (gx1, gy1), (255, 120, 0), 2)

    return annotated, pd.DataFrame(rows), (gx0, gy0, gx1, gy1)

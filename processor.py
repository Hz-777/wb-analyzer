import cv2
import numpy as np
from scipy.signal import find_peaks
import pandas as pd


def preprocess(img_bgr: np.ndarray, radius: int = 50) -> np.ndarray:
    """Convert to grayscale and subtract background (light background mode, like ImageJ step 3-4)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Rolling ball via morphological opening: estimates the bright background
    kernel_size = int(radius * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)

    # For light background: background - original → bands become bright, bg → 0
    # This is equivalent to ImageJ "Subtract Background (Light bg)" + "Invert"
    diff = cv2.subtract(background, gray)
    return diff


def detect_lanes(
    enhanced: np.ndarray,
    n_lanes: int | None = None,
    sensitivity: float = 0.3,
) -> list[tuple[int, int]]:
    """Detect vertical lane boundaries by horizontal intensity projection."""
    col_profile = enhanced.sum(axis=0).astype(float)

    # Smooth the profile
    kernel_size = max(3, len(col_profile) // 50)
    col_profile_smooth = np.convolve(
        col_profile, np.ones(kernel_size) / kernel_size, mode="same"
    )

    threshold = col_profile_smooth.max() * sensitivity
    height = col_profile_smooth.max() * sensitivity
    distance = max(10, len(col_profile) // 20)

    peaks, props = find_peaks(
        col_profile_smooth, height=height, distance=distance, prominence=threshold * 0.5
    )

    if n_lanes is not None and len(peaks) != n_lanes:
        # Force exactly n_lanes by taking top-N peaks by prominence
        if len(peaks) == 0:
            # No peaks found; divide image evenly
            w = enhanced.shape[1]
            step = w // n_lanes
            return [(i * step, (i + 1) * step) for i in range(n_lanes)]
        prominences = props.get("prominences", col_profile_smooth[peaks])
        idx = np.argsort(prominences)[::-1][:n_lanes]
        peaks = np.sort(peaks[idx])

    if len(peaks) == 0:
        return [(0, enhanced.shape[1])]

    # Build lane boundaries: midpoints between adjacent peaks
    boundaries = []
    w = enhanced.shape[1]
    for i, pk in enumerate(peaks):
        left = (peaks[i - 1] + pk) // 2 if i > 0 else max(0, pk - (peaks[1] - peaks[0]) // 2 if len(peaks) > 1 else pk // 2)
        right = (pk + peaks[i + 1]) // 2 if i < len(peaks) - 1 else min(w, pk + (pk - peaks[i - 1]) // 2 if i > 0 else w)
        boundaries.append((int(left), int(right)))

    return boundaries


def detect_bands(
    lane_enhanced: np.ndarray,
    sensitivity: float = 0.3,
) -> list[tuple[int, int]]:
    """Detect horizontal bands within a lane by vertical intensity projection."""
    row_profile = lane_enhanced.sum(axis=1).astype(float)

    kernel_size = max(3, len(row_profile) // 30)
    row_profile_smooth = np.convolve(
        row_profile, np.ones(kernel_size) / kernel_size, mode="same"
    )

    threshold = row_profile_smooth.max() * sensitivity
    distance = max(5, len(row_profile) // 15)

    peaks, _ = find_peaks(
        row_profile_smooth, height=threshold, distance=distance, prominence=threshold * 0.3
    )

    if len(peaks) == 0:
        return []

    bands = []
    h = lane_enhanced.shape[0]
    for i, pk in enumerate(peaks):
        # Band region: half-width defined by where profile falls below 30% of peak value
        half_width = distance // 2
        y0 = max(0, pk - half_width)
        y1 = min(h, pk + half_width)
        bands.append((int(y0), int(y1)))

    return bands


def measure_roi(
    enhanced: np.ndarray, x0: int, x1: int, y0: int, y1: int
) -> dict:
    """Measure Area, Mean, Min, Max, IntDen for a rectangular ROI."""
    roi = enhanced[y0:y1, x0:x1].astype(float)
    area = roi.size
    mean_val = roi.mean() if area > 0 else 0.0
    min_val = roi.min() if area > 0 else 0.0
    max_val = roi.max() if area > 0 else 0.0
    int_den = area * mean_val  # ImageJ IntDen = Area × Mean
    raw_int_den = roi.sum()    # ImageJ RawIntDen = sum of pixel values
    return {
        "Area": int(area),
        "Mean": round(float(mean_val), 3),
        "Min": round(float(min_val), 1),
        "Max": round(float(max_val), 1),
        "IntDen": round(float(int_den), 1),
        "RawIntDen": round(float(raw_int_den), 1),
    }


def analyze(
    img_bgr: np.ndarray,
    radius: int = 50,
    n_lanes: int | None = None,
    sensitivity: float = 0.3,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Full pipeline: preprocess → detect lanes → detect bands → measure → return annotated image + DataFrame."""
    enhanced = preprocess(img_bgr, radius)
    lanes = detect_lanes(enhanced, n_lanes=n_lanes, sensitivity=sensitivity)

    annotated = img_bgr.copy()
    rows = []

    for lane_idx, (x0, x1) in enumerate(lanes):
        lane_img = enhanced[:, x0:x1]
        bands = detect_bands(lane_img, sensitivity=sensitivity)

        # Fall back: measure the entire lane column if no bands found
        if not bands:
            bands = [(0, enhanced.shape[0])]

        for band_idx, (y0, y1) in enumerate(bands):
            metrics = measure_roi(enhanced, x0, x1, y0, y1)
            rows.append({
                "Lane": lane_idx + 1,
                "Band": band_idx + 1,
                "X_start": x0,
                "X_end": x1,
                "Y_start": y0,
                "Y_end": y1,
                **metrics,
            })
            # Draw rectangle on annotated image
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 200, 80), 2)
            label = f"L{lane_idx + 1}"
            cv2.putText(
                annotated, label,
                (x0 + 4, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 80), 2,
            )

    df = pd.DataFrame(rows)
    return annotated, df

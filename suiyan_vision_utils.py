# -*- coding: utf-8 -*-
"""SuiYan AI - quantitative vision layer (ITA skin tone + wrinkle density).

This module turns a portrait photo into measurable skin metrics and a
suggested SkinProfile JSON, feeding the RAG plan chain:

    portrait -> face ROI -> skin mask -> LAB stats -> ITA degree
             -> undertone ratio -> wrinkle density -> SkinProfile

Key formulas:
  ITA(deg) = arctan((L* - 50) / b*) * 180 / pi          (Chardon et al.)
  undertone_ratio = b* / (a* + 0.5)   (yellow vs red balance)

skin_type (oily/dry) is NOT quantifiable from a single photo, so the
frontend lets the user pick it; everything tone-related is measured here.
"""
import math
from typing import Dict

import cv2
import numpy as np

# Haar face detector shipped with OpenCV
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def _detect_face_roi(bgr: np.ndarray) -> tuple:
    """Return the largest face box (x, y, w, h); fallback to center crop."""
    h_img, w_img = bgr.shape[:2]
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(_CASCADE_PATH)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(80, 80))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return int(x), int(y), int(w), int(h)
    except Exception:
        pass  # detector unavailable or failed -> center crop below
    # fallback: center square region (60% of the shorter side)
    s = int(min(w_img, h_img) * 0.6)
    cx, cy = w_img // 2, h_img // 2
    return cx - s // 2, cy - s // 2, s, s


def _skin_mask(bgr_roi: np.ndarray) -> np.ndarray:
    """Binary skin mask combining classic RGB rule + YCrCb range, with a
    brightness floor to reject shadows / dark background pixels."""
    rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB).astype(np.int16)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rgb_rule = (
        (R > 95) & (G > 40) & (B > 20)
        & ((R - G) > 10) & ((R - B) > 15)
        & (R > G) & (G > B)
    )
    ycrcb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = ycrcb[..., 0].astype(np.int16), ycrcb[..., 1], ycrcb[..., 2]
    # Y > 100: webcam frames in dim light otherwise let hair/shadows into
    # the mask and drag the ITA degree far below the physical skin range
    ycc_rule = (Cr > 135) & (Cr < 180) & (Cb > 85) & (Cb < 135) & (Y > 100)
    mask = (rgb_rule & ycc_rule).astype(np.uint8)
    # clean speckles, keep largest connected component
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)
    return mask


def _lab_stats(bgr_roi: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Median L*/a*/b* over skin pixels (OpenCV LAB is 0-255 scaled).

    Pixels outside the physical skin luminance range [35, 92] are dropped
    before taking the median, so residual shadows/highlights cannot skew
    the ITA degree on webcam frames.
    """
    lab = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] = lab[..., 0] / 255.0 * 100.0          # L* in [0, 100]
    lab[..., 1] -= 128.0                                # a* in [-128, 127]
    lab[..., 2] -= 128.0                                # b* in [-128, 127]
    sel = mask.astype(bool) & (lab[..., 0] > 35) & (lab[..., 0] < 92)
    if sel.sum() < 400:
        raise ValueError("光线不足或未检测到清晰面部肤色，请正对光源重试")
    l_med = float(np.median(lab[..., 0][sel]))
    a_med = float(np.median(lab[..., 1][sel]))
    b_med = float(np.median(lab[..., 2][sel]))
    l_std = float(np.std(lab[..., 0][sel]))
    l_mean = float(np.mean(lab[..., 0][sel]))
    return {"l_star": l_med, "a_star": a_med, "b_star": b_med, "l_std": l_std,
            "l_mean": l_mean, "valid_px": int(sel.sum())}


def _winkle_density(gray_roi: np.ndarray, mask: np.ndarray) -> float:
    """High-frequency texture energy density inside the skin mask.

    Wrinkles/lines raise local gradients; we normalise Laplacian energy by
    skin area and squash it into [0, 1] with an empirical scale.
    """
    g = gray_roi.astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    energy = float(np.mean(np.abs(lap[mask.astype(bool)])))
    return float(min(1.0, energy / 12.0))  # 12.0 = empirical upper bound


def _ita_to_tone(ita: float, undertone: str) -> str:
    """Map ITA degree bins to our skin_tone vocabulary (tone-aware)."""
    if ita > 45:
        return "冷白皮" if undertone == "冷调" else "自然偏白（黄一白）"
    if ita > 30:
        return "自然偏白（黄二白）" if undertone == "暖调" else "自然白皙"
    if ita > 18:
        return "自然健康肤色（黄三白）"
    return "小麦色"


def _undertone_of(a_star: float, b_star: float) -> str:
    """Yellow-over-red ratio decides the undertone."""
    ratio = b_star / (abs(a_star) + 0.5)
    if ratio > 2.2:
        return "暖调"
    if ratio < 1.5:
        return "冷调"
    return "中性调"


def analyze_portrait_cv(bgr: np.ndarray) -> Dict:
    """Full quantitative analysis of one portrait (BGR ndarray)."""
    x, y, w, h = _detect_face_roi(bgr)
    roi = bgr[max(0, y):y + h, max(0, x):x + w]
    mask = _skin_mask(roi)
    stats = _lab_stats(roi, mask)

    ita = math.degrees(math.atan((stats["l_star"] - 50.0) / max(stats["b_star"], 1e-3)))
    undertone = _undertone_of(stats["a_star"], stats["b_star"])
    skin_tone = _ita_to_tone(ita, undertone)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    wrinkle = _winkle_density(gray, mask)

    concerns = []
    if wrinkle > 0.35:
        concerns.append("眼周细纹")
    if stats["l_std"] > 12:
        concerns.append("暗沉")
    if stats["b_star"] > 18:
        concerns.append("暗黄")

    return {
        "metrics": {
            "ita_degree": round(ita, 1),
            "l_star": round(stats["l_star"], 1),
            "a_star": round(stats["a_star"], 1),
            "b_star": round(stats["b_star"], 1),
            "l_std": round(stats["l_std"], 1),
            "undertone_ratio": round(stats["b_star"] / (abs(stats["a_star"]) + 0.5), 2),
            "wrinkle_density": round(wrinkle, 3),
            "skin_px": stats["valid_px"],
            "low_light": bool(stats["l_mean"] < 55),
            "face_box": [int(x), int(y), int(w), int(h)],
            "face_detected": True,
        },
        "suggested_profile": {
            "skin_tone": skin_tone,
            "undertone": undertone,
            "concerns": concerns,
        },
    }


def analyze_portrait_bytes(image_bytes: bytes) -> Dict:
    """Decode an uploaded image (jpg/png) and run the analysis."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cannot decode image; use jpg or png")
    # cap the long edge for speed
    h, w = bgr.shape[:2]
    if max(h, w) > 1200:
        scale = 1200.0 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    return analyze_portrait_cv(bgr)

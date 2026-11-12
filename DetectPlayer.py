import cv2
import numpy as np


def _color_ratio(hsv_roi, lower, upper):
    mask = cv2.inRange(hsv_roi, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    return cv2.countNonZero(mask) / max(1, mask.size)


def detect_players(image, min_area=20, max_area=12000, min_color_ratio=0.02):
    """Classical-CV player detector.

    Returns a list of detections with bbox=(x,y,w,h), center=(cx,cy), class_name and score.
    The score is the dominant jersey-color pixel ratio, so it is useful for ranking/debugging,
    but it is not a calibrated probability.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Candidate foreground = pixels that are not grass-green.
    grass = cv2.inRange(hsv, np.array([25, 35, 20]), np.array([90, 255, 255]))
    foreground = cv2.bitwise_not(grass)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    jersey_ranges = {
        "blue": [([95, 50, 40], [145, 255, 255])],
        "red": [([0, 80, 70], [20, 255, 255]), ([150, 80, 70], [179, 255, 255])],
        "white": [([0, 0, 150], [180, 80, 255])],
    }

    detections = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area or area > max_area or w < 3 or h < 6:
            continue
        # Reject very flat regions (field lines / advertising fragments).
        if w / max(h, 1) > 2.2:
            continue

        roi = hsv[y:y + h, x:x + w]
        if roi.size == 0:
            continue

        ratios = {}
        for name, ranges in jersey_ranges.items():
            ratio = 0.0
            for lower, upper in ranges:
                ratio += _color_ratio(roi, lower, upper)
            ratios[name] = min(ratio, 1.0)

        team = max(ratios, key=ratios.get)
        score = ratios[team]
        if score < min_color_ratio:
            continue
        # Small, almost-square white blobs are more likely to be the ball/field markings
        # than a standing player. This also keeps player and ball detectors from competing.
        if team == "white" and max(w, h) < 25 and min(w, h) / max(w, h) > 0.70:
            continue

        detections.append({
            "bbox": (int(x), int(y), int(w), int(h)),
            "center": (float(x + w / 2), float(y + h)),  # feet point for pitch mapping
            "class_name": "player",
            "team": team,
            "score": float(score),
        })

    return detections


def draw_player_detections(image, detections):
    out = image.copy()
    colors = {"blue": (0, 255, 0), "red": (255, 0, 0), "white": (0, 255, 255)}
    for det in detections:
        x, y, w, h = det["bbox"]
        color = colors.get(det.get("team"), (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, f"{det.get('team','?')} {det['score']:.2f}", (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def DetectPlayer1(image, path=None) -> np.ndarray:
    """Backward-compatible wrapper for the original project."""
    detections = detect_players(image)
    out = draw_player_detections(image, detections)
    if path:
        cv2.imwrite(path, out)
    return out


def detect_ball(image, min_area=5, max_area=350, min_circularity=0.35):
    """Detect a small bright ball candidate on/near the green pitch.

    Returns detections sorted by heuristic score. Temporal tracking should be used to reject
    isolated false positives and recover short occlusions.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(hsv, np.array([25, 30, 20]), np.array([95, 255, 255]))
    field = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    field = cv2.dilate(field, np.ones((7, 7), np.uint8), iterations=1)

    # Bright/low-saturation pixels capture common white/light balls.
    bright = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([180, 130, 255]))
    mask = cv2.bitwise_and(bright, field)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    H, W = image.shape[:2]
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 2 or h < 2 or w > 35 or h > 35:
            continue
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
        aspect = min(w, h) / max(w, h)
        if circularity < min_circularity or aspect < 0.45:
            continue

        roi = hsv[y:y+h, x:x+w]
        value_score = float(np.mean(roi[..., 2]) / 255.0)
        size_prior = 1.0 - min(1.0, abs(max(w, h) - 8.0) / 25.0)
        score = 0.45 * circularity + 0.35 * aspect + 0.15 * value_score + 0.05 * size_prior
        detections.append({
            "bbox": (int(x), int(y), int(w), int(h)),
            "center": (float(x + w / 2), float(y + h / 2)),
            "class_name": "ball",
            "score": float(score),
        })

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


def DetectBall(image):
    """Backward-compatible visual wrapper. Returns an annotated image."""
    out = image.copy()
    candidates = detect_ball(image)
    if candidates:
        det = candidates[0]
        x, y, w, h = det["bbox"]
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(out, f"ball {det['score']:.2f}", (x, max(15, y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return out

from typing import Dict, List

import cv2

def draw_plate_detections(
    frame,
    detections: List[Dict],
    conf_threshold: float,
    color=(0, 255, 0),
    thickness: int = 2,
) -> None:

    for det in detections:
        if det["confidence"] < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["confidence"]
        label = f"{det.get('class_name', '?')} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

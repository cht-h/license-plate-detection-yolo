from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


def _iou_xyxy(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    union = ar + br - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    smooth_bbox: List[float]
    confidence: float
    class_id: int
    class_name: str
    missed: int = 0


class TemporalDetectionSmoother:

    def __init__(
        self,
        iou_match: float = 0.3,
        smooth_alpha: float = 0.45,
        max_missed_frames: int = 5,
        hold_ghost: bool = True,
    ):
        self.iou_match = iou_match
        self.smooth_alpha = smooth_alpha
        self.max_missed_frames = max_missed_frames
        self.hold_ghost = hold_ghost
        self._tracks: List[_Track] = []

    def reset(self) -> None:
        self._tracks.clear()

    @staticmethod
    def _ema(prev: List[float], nxt: List[float], alpha: float) -> List[float]:
        return [alpha * nxt[i] + (1.0 - alpha) * prev[i] for i in range(4)]

    def update(self, detections: List[Dict]) -> List[Dict]:
        dets_sorted = sorted(
            detections,
            key=lambda d: d["confidence"],
            reverse=True,
        )
        n_tracks = len(self._tracks)
        used_track = [False] * n_tracks
        matched_ti_for_det: List[int] = [-1] * len(dets_sorted)

        for di, det in enumerate(dets_sorted):
            db = det["bbox"]
            best_iou = self.iou_match
            best_ti = -1
            for ti, tr in enumerate(self._tracks):
                if used_track[ti]:
                    continue
                iou = _iou_xyxy(db, tr.smooth_bbox)
                if iou >= best_iou:
                    best_iou = iou
                    best_ti = ti
            if best_ti >= 0:
                used_track[best_ti] = True
                matched_ti_for_det[di] = best_ti

        new_tracks: List[_Track] = []

        for di, det in enumerate(dets_sorted):
            ti = matched_ti_for_det[di]
            bbox = list(det["bbox"])
            if ti < 0:
                new_tracks.append(
                    _Track(
                        smooth_bbox=bbox,
                        confidence=float(det["confidence"]),
                        class_id=int(det["class_id"]),
                        class_name=str(det["class_name"]),
                        missed=0,
                    )
                )
                continue

            tr = self._tracks[ti]
            tr.smooth_bbox = self._ema(tr.smooth_bbox, bbox, self.smooth_alpha)
            tr.confidence = float(det["confidence"])
            tr.class_id = int(det["class_id"])
            tr.class_name = str(det["class_name"])
            tr.missed = 0
            new_tracks.append(tr)

        for ti, tr in enumerate(self._tracks):
            if used_track[ti]:
                continue
            tr.missed += 1
            if self.hold_ghost and tr.missed <= self.max_missed_frames:
                new_tracks.append(tr)

        self._tracks = new_tracks

        return [
            {
                "bbox": [float(x) for x in t.smooth_bbox],
                "confidence": t.confidence,
                "class_id": t.class_id,
                "class_name": t.class_name,
            }
            for t in self._tracks
        ]
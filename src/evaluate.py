import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from .logger import log
from .model_impl import My_LicensePlate_Model

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def evaluate_model(
    model_path: str,
    images_dir: str,
    labels_dir: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.5,
) -> Dict:
    log.info(f"Evaluating model: {model_path}")
    log.info(f"Images dir: {images_dir}")
    log.info(f"Labels dir: {labels_dir}")

    model = My_LicensePlate_Model(model_path)

    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    if not images_path.exists():
        log.error(f"Images directory not found: {images_dir}")
        return {}

    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    log.info(f"Found {len(image_files)} images")

    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_path in tqdm(image_files, desc="Evaluating"):
        frame = cv2.imread(str(img_path))
        if frame is None:
            log.warning(f"Cannot read image: {img_path}")
            continue

        label_file = labels_path / f"{img_path.stem}.txt"
        gt_boxes = []
        if label_file.exists():
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, cx, cy, w, h = map(float, parts[:5])
                        img_h, img_w = frame.shape[:2]
                        x1 = (cx - w/2) * img_w
                        y1 = (cy - h/2) * img_h
                        x2 = (cx + w/2) * img_w
                        y2 = (cy + h/2) * img_h
                        gt_boxes.append([x1, y1, x2, y2])

        detections = model.detect_plates(frame)
        det_boxes = [d["bbox"] for d in detections if d["confidence"] >= conf_threshold]

        total_gt += len(gt_boxes)

        matched_gt = set()
        matched_det = set()

        for i, det_box in enumerate(det_boxes):
            best_iou = 0
            best_j = -1
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = calculate_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_threshold:
                matched_det.add(i)
                matched_gt.add(best_j)

        tp = len(matched_det)
        fp = len(det_boxes) - tp
        fn = len(gt_boxes) - len(matched_gt)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    results = {
        "total_images": len(image_files),
        "total_gt_boxes": total_gt,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1_score": f1_score,
        "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold,
    }

    log.info(f"Evaluation results: Precision={overall_precision:.4f}, Recall={overall_recall:.4f}, F1={f1_score:.4f}")
    log.info(f"TP={total_tp}, FP={total_fp}, FN={total_fn}")

    return results

def plot_evaluation_results(results: Dict, save_path: Optional[str] = None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    metrics = ["precision", "recall", "f1_score"]
    values = [results.get(m, 0) for m in metrics]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    ax[0].bar(metrics, values, color=colors)
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("Score")
    ax[0].set_title("Detection Metrics")
    for i, v in enumerate(values):
        ax[0].text(i, v + 0.02, f"{v:.3f}", ha="center")

    counts = ["true_positives", "false_positives", "false_negatives"]
    labels = ["TP", "FP", "FN"]
    count_values = [results.get(c, 0) for c in counts]
    colors2 = ["#27ae60", "#e67e22", "#c0392b"]

    ax[1].pie(count_values, labels=labels, colors=colors2, autopct="%1.1f%%", startangle=90)
    ax[1].set_title("Detection Distribution")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Plot saved to {save_path}")
    else:
        plt.show()
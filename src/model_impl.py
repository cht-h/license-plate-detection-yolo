import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from .logger import log

class My_LicensePlate_Model:
    
    def __init__(
        self,
        model_path: Union[str, Path] = "runs/detect/train/weights/best.pt",
        default_infer_imgsz: Optional[int] = None,
    ):
    
        self.model_path = Path(model_path)
        self.default_infer_imgsz = default_infer_imgsz
        if not self.model_path.exists():
            log.warning(f"Model file not found: {self.model_path}. Loading default YOLOv8n.")
            self.model = YOLO("yolov8n.pt")
        else:
            log.info(f"Loading model from {self.model_path}")
            self.model = YOLO(str(self.model_path))

    def detect_plates(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 50,
        imgsz: Optional[int] = None,
        augment: bool = False,
        half: Optional[bool] = None,
    ) -> List[Dict]:

        if frame is None:
            log.error("Empty frame received")
            return []

        infer_sz = imgsz if imgsz is not None else self.default_infer_imgsz
        predict_kw: Dict[str, Any] = {
            "source": frame,
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "verbose": False,
            "augment": augment,
        }
        if infer_sz is not None:
            predict_kw["imgsz"] = infer_sz
        if half is not None:
            predict_kw["half"] = half

        results = self.model.predict(**predict_kw)

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy()

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                detections.append({
                    "bbox": box.tolist(),
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "class_name": self.model.names[int(cls_id)],
                })

        log.debug(f"Detected {len(detections)} plates on frame")
        return detections

    def get_model_info(self) -> Dict:
        return {
            "model_path": str(self.model_path),
            "model_name": self.model.model_name if hasattr(self.model, 'model_name') else "YOLO",
            "classes": self.model.names,
        }
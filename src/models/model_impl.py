from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)


class My_LicensePlate_Model:
    """
    YOLO-based license plate detector.
    
    This class encapsulates the model loading, inference, and post-processing
    for detecting license plates in images or video frames.
    """
    
    def __init__(
        self, 
        model_path: Optional[Union[str, Path]] = None, 
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'cpu'
    ):
        """
        Initialize the model.
        
        Args:
            model_path: Path to the trained YOLO model weights. 
                       If None, uses a default pretrained model for testing.
            conf_threshold: Confidence threshold for detections.
            iou_threshold: IoU threshold for NMS.
            device: Device to run inference on ('cpu', 'cuda', 'mps').
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Check device availability
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Using CPU instead.")
            self.device = 'cpu'
        else:
            self.device = device
            
        logger.info(f"Initializing model on device: {self.device}")
        logger.info(f"Configuration: conf_threshold={conf_threshold}, iou_threshold={iou_threshold}")
        
        try:
            # Load model
            if model_path is None:
                logger.warning("No model path provided, using pretrained YOLOv8n model (for testing only!)")
                logger.warning("This model detects 80 classes, not just license plates!")
                self.model = YOLO('yolov8n.pt')
                self.is_custom_model = False
            else:
                model_path = Path(model_path)
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                self.model = YOLO(str(model_path))
                self.is_custom_model = True
                logger.info(f"Custom model loaded from: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise
    
    def detect_plates(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect license plates in a frame.
        
        Args:
            frame: Input image as numpy array (BGR format, as from OpenCV).
            
        Returns:
            List of dictionaries, each containing:
                - 'bbox': [x1, y1, x2, y2] coordinates
                - 'confidence': float confidence score
                - 'class_id': int class id (should be 0 for license plate)
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received")
            return []
        
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            logger.warning(f"Invalid frame dimensions: {frame.shape}")
            return []
        
        logger.debug(f"Processing frame with shape: {frame.shape}")
        
        try:
            # Run inference
            results = self.model(
                frame, 
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )[0]
            
            detections = []
            
            # Process results
            if results.boxes is not None:
                for box in results.boxes:
                    # Get box coordinates in xyxy format
                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                    confidence = float(box.conf[0].cpu())
                    class_id = int(box.cls[0].cpu()) if box.cls is not None else 0
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_id': class_id
                    })
            
            logger.debug(f"Found {len(detections)} detections")
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}", exc_info=True)
            return []
    
    def detect_plates_with_visualization(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect plates and return frame with drawn bounding boxes.
        
        Args:
            frame: Input image (will be modified in place).
            
        Returns:
            Image with drawn detections.
        """
        detections = self.detect_plates(frame)
        
        # Make a copy to avoid modifying original
        result_frame = frame.copy()
        
        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw rectangle
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f"License Plate: {conf:.2f}"
            
            # Calculate text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                result_frame,
                (x1, y1 - text_height - 5),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                result_frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )
        
        return result_frame
    
    def __repr__(self) -> str:
        return f"My_LicensePlate_Model(device={self.device}, conf={self.conf_threshold}, iou={self.iou_threshold})"
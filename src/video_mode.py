import cv2
from pathlib import Path
from typing import Optional

from .logger import log
from .model_impl import My_LicensePlate_Model
from .track_smoothing import TemporalDetectionSmoother
from .visualization import draw_plate_detections

def process_video(
    input_path: str,
    output_path: str,
    model_path: Optional[str] = None,
    conf_threshold: float = 0.5,
    infer_conf: Optional[float] = None,
    model_iou: float = 0.45,
    max_det: int = 50,
    imgsz: Optional[int] = None,
    use_smoothing: bool = True,
    smooth_iou: float = 0.3,
    smooth_alpha: float = 0.45,
    augment: bool = False,
    show_preview: bool = False,
) -> None:

    log.info(f"Starting video processing: {input_path}")

    if not Path(input_path).exists():
        log.error(f"Input video not found: {input_path}")
        return

    if infer_conf is None:
        if conf_threshold > 0:
            infer_conf = min(conf_threshold, 0.25)
        else:
            infer_conf = 0.15

    if conf_threshold < 0.05:
        log.warning(
            "Очень низкий --conf: будет много ложных рамок. "
            "Попробуйте --infer-conf 0.15–0.25 и --conf 0.35–0.5 плюс дообучение модели."
        )

    if model_path:
        model = My_LicensePlate_Model(model_path, default_infer_imgsz=imgsz)
    else:
        model = My_LicensePlate_Model(default_infer_imgsz=imgsz)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        log.error(f"Cannot open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    log.info(f"Video info: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
    log.info(
        f"Inference: infer_conf={infer_conf}, draw_conf={conf_threshold}, "
        f"iou_nms={model_iou}, max_det={max_det}, imgsz={imgsz}, smooth={use_smoothing}"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    smoother = (
        TemporalDetectionSmoother(
            iou_match=smooth_iou,
            smooth_alpha=smooth_alpha,
        )
        if use_smoothing
        else None
    )

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            raw = model.detect_plates(
                frame,
                conf=infer_conf,
                iou=model_iou,
                max_det=max_det,
                imgsz=imgsz,
                augment=augment,
            )

            if smoother is not None:
                detections = smoother.update(raw)
            else:
                detections = raw

            draw_plate_detections(frame, detections, conf_threshold=conf_threshold)

            out.write(frame)

            if show_preview:
                cv2.imshow("License Plate Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Processing interrupted by user")
                    break

            if frame_count % 100 == 0:
                log.info(f"Processed {frame_count}/{total_frames} frames")

    except Exception as e:
        log.exception(f"Error during video processing: {e}")
    finally:
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

    log.info(f"Video processing completed. Output saved to: {output_path}")
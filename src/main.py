"""
Main CLI application for license plate detection.
Supports video file processing and webcam live stream.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2

# Add src to path
sys.path.append(str(Path(__file__).parent))

from models.model_impl import My_LicensePlate_Model
from utils.logger import setup_logger

logger = setup_logger(__name__)


def process_video(
    model: My_LicensePlate_Model,
    input_path: Path,
    output_path: Optional[Path] = None,
    show_preview: bool = False,
    save_video: bool = True
) -> None:
    """
    Process a video file and detect license plates.
    
    Args:
        model: Initialized model instance
        input_path: Path to input video file
        output_path: Path to output video file (if None, auto-generate)
        show_preview: Show real-time preview
        save_video: Save output video
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        logger.error(f"Input video not found: {input_path}")
        return
    
    logger.info(f"Processing video: {input_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    
    if not cap.isOpened():
        logger.error(f"Cannot open video: {input_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    writer = None
    if save_video:
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_detected.mp4"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        logger.info(f"Saving output to: {output_path}")
    
    frame_count = 0
    detection_count = 0
    
    logger.info("Starting video processing...")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Detect plates
            result_frame = model.detect_plates_with_visualization(frame)
            detections = model.detect_plates(frame)
            
            if detections:
                detection_count += len(detections)
            
            # Save frame
            if writer:
                writer.write(result_frame)
            
            # Show preview
            if show_preview:
                cv2.imshow('License Plate Detection', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User interrupted")
                    break
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
    
    logger.info(f"Processing complete. Processed {frame_count} frames")
    logger.info(f"Total detections: {detection_count}")
    logger.info(f"Average detections per frame: {detection_count / frame_count:.2f}" if frame_count > 0 else "No frames processed")


def process_webcam(
    model: My_LicensePlate_Model,
    camera_id: int = 0,
    show_fps: bool = True
) -> None:
    """
    Process webcam stream in real-time.
    
    Args:
        model: Initialized model instance
        camera_id: Camera device ID
        show_fps: Display FPS counter
    """
    logger.info(f"Starting webcam stream (camera {camera_id})")
    logger.info("Press 'q' to quit")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera_id}")
        return
    
    fps_counter = 0
    import time
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                logger.error("Failed to capture frame")
                break
            
            # Detect plates
            result_frame = model.detect_plates_with_visualization(frame)
            
            # Calculate and display FPS
            if show_fps:
                fps_counter += 1
                elapsed = time.time() - start_time
                if elapsed > 1.0:
                    fps = fps_counter / elapsed
                    cv2.putText(
                        result_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
                    fps_counter = 0
                    start_time = time.time()
            
            # Show frame
            cv2.imshow('License Plate Detection - Webcam', result_frame)
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quitting...")
                break
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam stream stopped")


def main():
    parser = argparse.ArgumentParser(description="License Plate Detection System")
    
    # Model arguments
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on (default: cpu)"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["video", "webcam"],
        help="Processing mode: video file or webcam"
    )
    
    # Video mode arguments
    parser.add_argument(
        "--input", type=str, default=None,
        help="Input video file path (required for video mode)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output video file path (optional)"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show real-time preview during video processing"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save output video (only show preview)"
    )
    
    # Webcam mode arguments
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--no-fps", action="store_true",
        help="Don't show FPS counter in webcam mode"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "video" and not args.input:
        parser.error("Video mode requires --input argument")
    
    # Initialize model
    logger.info("Initializing model...")
    try:
        model = My_LicensePlate_Model(
            model_path=args.model,
            conf_threshold=args.conf,
            device=args.device
        )
        logger.info(f"Model initialized: {model}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        sys.exit(1)
    
    # Run in selected mode
    if args.mode == "video":
        process_video(
            model=model,
            input_path=Path(args.input),
            output_path=Path(args.output) if args.output else None,
            show_preview=args.preview,
            save_video=not args.no_save
        )
    elif args.mode == "webcam":
        process_webcam(
            model=model,
            camera_id=args.camera,
            show_fps=not args.no_fps
        )


if __name__ == "__main__":
    main()
"""
Data preparation utilities for extracting frames from videos for annotation.
"""
import cv2
from pathlib import Path
import sys
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger(__name__)

def extract_frames_for_annotation(
    video_path: Path,
    output_dir: Path,
    frame_interval: int = 30,
    max_frames: Optional[int] = None,
    resize: Optional[tuple] = None
) -> int:
    """
    Extract frames from video for manual annotation.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame
        max_frames: Maximum number of frames to extract
        resize: Resize to (width, height) if specified
        
    Returns:
        Number of frames extracted
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return 0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting frames from {video_path}")
    logger.info(f"Frame interval: {frame_interval}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video info: {total_frames} frames, {fps:.2f} FPS")
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Extract every Nth frame
        if frame_count % frame_interval == 0:
            # Optionally resize frame
            if resize:
                frame = cv2.resize(frame, resize)
            
            # Save frame
            output_path = output_dir / f"frame_{extracted_count:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            extracted_count += 1
            
            if extracted_count % 100 == 0:
                logger.info(f"Extracted {extracted_count} frames...")
            
            if max_frames and extracted_count >= max_frames:
                logger.info(f"Reached maximum frames limit: {max_frames}")
                break
        
        frame_count += 1
    
    cap.release()
    logger.info(f"Extraction complete. Saved {extracted_count} frames to {output_dir}")
    
    return extracted_count


def create_dataset_structure(base_dir: Path):
    """
    Create standard dataset structure for YOLO training.
    
    Args:
        base_dir: Base directory for dataset
    """
    base_dir = Path(base_dir)
    
    # Create train/val directories
    for split in ['train', 'val']:
        (base_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (base_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Create data.yaml template
    yaml_content = f"""
# Dataset configuration
path: {base_dir.absolute()}
train: train/images
val: val/images

# Classes
nc: 1  # number of classes
names: ['license_plate']
"""
    
    with open(base_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content.strip())
    
    logger.info(f"Dataset structure created at {base_dir}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract frames from video for annotation")
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument("--output", type=str, default="data/raw/frames", help="Output directory")
    parser.add_argument("--interval", type=int, default=30, help="Extract every Nth frame")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames to extract")
    parser.add_argument("--resize", type=str, default=None, help="Resize to WxH, e.g., 640x480")
    
    args = parser.parse_args()
    
    resize = None
    if args.resize:
        w, h = map(int, args.resize.split('x'))
        resize = (w, h)
    
    extract_frames_for_annotation(
        video_path=Path(args.video_path),
        output_dir=Path(args.output),
        frame_interval=args.interval,
        max_frames=args.max_frames,
        resize=resize
    )
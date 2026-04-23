from ultralytics import YOLO
from ultralytics.utils import SETTINGS

SETTINGS.update({"datasets_dir": "."})

def train():

    model = YOLO("yolov11l.pt")

    model.train(
        data="data/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device=0
    )

if __name__ == "__main__":
    train()
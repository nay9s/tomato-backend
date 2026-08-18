from ultralytics import YOLO

model = YOLO("best.pt")

model.export(
    format="onnx",
    imgsz=640,
    dynamic=False,
    simplify=True,
    nms=True,
    end2end=False
)

print("Export complete")
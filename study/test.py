from ultralytics import YOLO, RTDETR

# model = RTDETR("rt-detr/rtdetr-resnet50.yaml")
model = RTDETR("zxx/rtdetr-r18.yaml").load("rtdetr-r18-f32-28epoch.pt")

# Evaluate model performance on the validation set
metrics = model.val(
    data="VisDrone.yaml",
    imgsz=640,
    batch=16,
    device=0,
    split="val",
)

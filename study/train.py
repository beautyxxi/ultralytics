from ultralytics import YOLO, RTDETR

# model = YOLO("zxx/yolo11s.yaml")  
# model = RTDETR("zxx/yolo11s-AIFI.yaml")
model = RTDETR("rt-detr/rtdetr-resnet50.yaml")

# Train the model
train_results = model.train(
    data="VisDrone.yaml",  # path to dataset YAML
    epochs=150,  # number of training epochs
    imgsz=640,  # training image size
    device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch=8,
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
# results = model.predict(
#     [
#         "D:\\work\\datasets\\custom\\images\\train\\001.jpg",
#         "D:\\work\\datasets\\custom\\images\\train\\002.jpg",
#     ]
# )
# print(results)
# results[0].show()
# results[1].show()

# # Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model
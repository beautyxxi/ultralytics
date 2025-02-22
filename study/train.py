from ultralytics import YOLO, RTDETR

# model = RTDETR("rt-detr/rtdetr-resnet50.yaml")
model = RTDETR("zxx/rtdetr-r18.yaml")

# Train the model
train_results = model.train(
    data="VisDrone.yaml",  # path to dataset YAML
    epochs=150,  # number of training epochs
    imgsz=640,  # training image size
    device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch=4, # batch_size
    amp=False, # 自动混合精度，False为使用FP32
    lr0=0.0001, # 初始学习率
    lrf=1.0, # 最终学习率为 lr0 * lrf
    warmup_epochs=2000, # 预热轮数
    # cache=False,
    optimizer="AdamW", # 优化器
)

# Evaluate model performance on the validation set
metrics = model.val()

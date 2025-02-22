from ultralytics import YOLO, RTDETR

# 设置wandb
# https://wandb.ai/home
import os
os.environ["WANDB_API_KEY"]="85bb6eb561cb3a3ff41129c7a2f472be2ad1809f" # 421626388-heu
os.environ['WANDB_MODE'] = 'online' # offline[训练完成后记录],online[边训练边记录],disabled[禁用]

# model = RTDETR("zxx/rtdetr-r50.yaml")
# model = RTDETR("zxx/rtdetr-r18.yaml")
model = YOLO("zxx/yolo11n.yaml")

# Train the model
train_results = model.train(
    data="coco8.yaml",  # path to dataset YAML
    epochs=2,  # number of training epochs
    imgsz=640,  # training image size
    device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch=4, # batch_size
    amp=False, # 自动混合精度，False为使用FP32
    lr0=0.0001, # 初始学习率
    lrf=1.0, # 最终学习率为 lr0 * lrf
    cache=False, # 标签是否缓存
    optimizer="AdamW", # 优化器
    project="zxx", # 设置project名称，在wandb下面显示
    name="yolo11n", # 将本次实验结果记录在project/name下
)

# Evaluate model performance on the validation set
metrics = model.val()

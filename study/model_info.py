from ultralytics import YOLO, RTDETR
from ultralytics.engine.model import Model

def show_model_info(model : Model):
    print(model.info())

rtdetr_r18 = RTDETR("zxx/rtdetr-r18.yaml")

show_model_info(rtdetr_r18)

rtdetr_r50 = RTDETR("rt-detr/rtdetr-resnet50.yaml")

show_model_info(rtdetr_r50)

yolo11s = YOLO("zxx/yolo11s.yaml")

show_model_info(yolo11s)

yolo11l = YOLO("zxx/yolo11l.yaml")

show_model_info(yolo11l)
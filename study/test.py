import warnings
import os
from pathlib import Path
from ultralytics import RTDETR
import torch

# warnings.filterwarnings('ignore')  #忽略 Python 运行时的警告（warnings）


# def check_path(path):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Path does not exist: {path}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # # 获取当前脚本所在的目录
    # current_dir = Path(__file__).parent
    # 构建相对路径
    # yaml_path = '/mnt/RTdetr/RTDETR-main/dataset/dataset_visdrone/data.yaml'
    # check_path(yaml_path)
    model = RTDETR("zxx/uav-rtdetr.yaml")
    model.train(
                # data=str(yaml_path),
                data="VisDrone.yaml",   # 官方路径
                # cache=False,
                imgsz=640,
                epochs=150,
                batch=8,
                workers=8,
                device='0',
                # resume='', # last.pt path
                # project='runs/train',
                # name='exp',
                # patience = 40,
                )
    
# Evaluate model performance on the validation set
metrics = model.val()
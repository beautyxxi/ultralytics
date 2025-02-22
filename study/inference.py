from ultralytics import YOLO, RTDETR

# model = RTDETR("zxx/rtdetr-r18.yaml").load("rtdetr-l.pt")

# print(model.info())

# model_dict = model.state_dict()

# for k, v in model_dict.items():
#     print(k)

# model = RTDETR("rt-detr/rtdetr-resnet50.yaml").load("rtdetr-l.pt")

# print(model.info())

# model_dict = model.state_dict()

# print(model.state_dict().items())

model = RTDETR("rt-detr/rtdetr-l.yaml").load("rtdetr-l.pt")

print(model.info())

model_dict = model.state_dict()

print(model.state_dict().items())




# results = model.predict(
#     [
#         "D:\\work\\datasets\\custom\\images\\train\\001.jpg",
#         "D:\\work\\datasets\\custom\\images\\train\\002.jpg",
#     ]
# )
# # print(results)
# # results[0].show()
# # results[1].show()
# print(results[0])
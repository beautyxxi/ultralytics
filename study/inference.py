from ultralytics import YOLO

model = YOLO("D:\\work\\ultralytics\\runs\\detect\\train\\weights\\best.pt")
# model = YOLO("yolo11n.pt")

results = model.predict(
    [
        "D:\\work\\datasets\\custom\\images\\train\\001.jpg",
        "D:\\work\\datasets\\custom\\images\\train\\002.jpg",
    ]
)
# print(results)
# results[0].show()
# results[1].show()
print(results[0])
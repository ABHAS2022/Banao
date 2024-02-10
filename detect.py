from ultralytics import YOLO

model = YOLO("C:\\Users\\ABHAS\\Downloads\\temp\\old projects and trash\\WebPage Element Detection\\file\\kaggle\\working\\runs\\detect\\train\\weights\\best.pt")

res = model.predict(source="C:\\Users\\ABHAS\\Downloads\\Screenshot_10-2-2024_21841_www.figma.com.jpeg", show = True, save = True, conf = 0.5)
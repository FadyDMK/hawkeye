from ultralytics import YOLO

model = YOLO("../yolo8n.pt", task="detect")
results = model(source="../output_frames/right/right1_0000.jpg")

for result in results:
    print(result.boxes.data)
    result.show()  
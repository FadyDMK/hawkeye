from ultralytics import YOLO, ASSETS

model = YOLO("../yolo8n.pt", task="detect")
results = model(source="../output_frames/right/right1_0000.jpg")

for result in results:
    print(result.boxes.data)
    result.show()  # uncomment to view each result image
    
    # reference https://docs.ultralytics.com/modes/predict/ for more information.
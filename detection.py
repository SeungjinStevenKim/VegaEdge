from ultralytics import YOLO

model = YOLO("yolo11n.pt")

result = model.predict( 
    source="test_videos/CHD_Anomaly_Highangle_001.mov", # path to video file
    show=True, # display results in a window
    conf=0.25, # confidence threshold
    save=True, # save results to 'runs/detect/predict'
    classes=[2, 3, 5, 7], #Detect only: car, motorcycle, bus, and truck
    show_conf=False, # hide confidence scores
    show_labels=True # display class names
)
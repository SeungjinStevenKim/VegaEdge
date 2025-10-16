from ultralytics import YOLO

model = YOLO("yolo11n.pt")

result = model.track(
    source="test_videos/CHD_Anomaly_Highangle_001.mov", # path to video file
    show=True, # display results in a window
    conf=0.35, # confidence threshold
    save=True, # save results to 'runs/detect/track'
    classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
    show_conf=False, # hide confidence scores
    show_labels=True, # display class names
    persist=True  # keeps ID consistency across frames
)
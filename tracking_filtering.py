from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque

# ===== CONFIGURATION =====
VIDEO_PATH = "test_videos/CHD_Anomaly_Highangle_001.mov"  # input video path
MODEL_PATH = "yolo11n.pt"                                # YOLO model file
CLASSES = [2, 3, 5, 7]       # classes to track: car, motorcycle, bus, truck
CONF = 0.25                  # detection confidence threshold
TRACKER = "bytetrack.yaml"   # tracker config file
HISTORY_LEN = 15             # track history length (frames) ~3s if 5 fps
DY_THRESH_RATIO = 0.01       # vertical-movement threshold as fraction of frame height
SHOW = True                  # show output window
SAVE_PATH = "runs/filtered/output.mp4"  # output video path
# ==========================

COLORS = {
    2: (0, 255, 0),    # car -> green
    3: (255, 0, 0),    # motorcycle -> blue
    5: (0, 0, 255),    # bus -> red
    7: (0, 165, 255),  # truck -> orange
}

# Initialize: detection & tracking model
model = YOLO(MODEL_PATH)

# Initialize tracking memory
# each track id maps to a deque of recent (cx, cy) with maxlen HISTORY_LEN
track_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
ret, frame = cap.read()
H, W = frame.shape[:2]  # frame.shape[:2] -> (height, width)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(SAVE_PATH, fourcc, fps, (W, H))

# 3-second warmup: skip initial frames to stabilize detector/tracker
for _ in range(int(fps * 3)):
    ret, _ = cap.read()
    if not ret:
        break

# Main loop: use model.track in streaming mode (iterator of results)
results_iter = model.track(
    source=VIDEO_PATH,
    conf=CONF,
    classes=CLASSES,
    tracker=TRACKER,
    persist=True,
    stream=True,
    show=False,
    save=False
)

for result in results_iter:
    frame = result.orig_img
    boxes = result.boxes
    # if no detections, write frame and continue
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        out.write(frame)
        continue

    # boxes.* are typically torch tensors (may be on GPU)
    # move to CPU and convert to numpy for OpenCV
    ids = boxes.id.cpu().numpy().astype(int)   # track IDs (N,)
    xyxy = boxes.xyxy.cpu().numpy()            # bounding boxes (N,4) -> [x1,y1,x2,y2]
    cls = boxes.cls.cpu().numpy().astype(int)  # class indices (N,)
    names = result.names                        # mapping: class idx -> class name

    # update track history with box centers
    for (x1, y1, x2, y2), tid, cid in zip(xyxy, ids, cls):
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        track_history[tid].append((cx, cy))

    # filter out tracks that move upward / are stationary or too short
    filtered_ids = []
    for tid, history in list(track_history.items()):
        if len(history) < 2:
            # not enough history to compute motion
            continue

        # compare earliest and latest y to get vertical displacement
        y_old, y_new = history[0][1], history[-1][1]
        dy = y_new - y_old
        # if vertical movement is smaller than threshold, skip (moving up or stationary)
        if dy < H * DY_THRESH_RATIO:
            continue

        # only accept tracks that have been observed for HISTORY_LEN frames
        if len(history) >= HISTORY_LEN:
            filtered_ids.append(tid)

    # draw only the filtered incoming vehicles on the frame
    # zip(xyxy, ids, cls) processes matching elements together
    for (x1, y1, x2, y2), tid, cid in zip(xyxy, ids, cls):
        if tid not in filtered_ids:
            continue
        color = COLORS.get(cid, (0, 255, 0))  # fallback green if class not in map
        label = names.get(cid, str(cid))  # fallback to class index string if name missing
        # draw rectangle and label using class-specific color
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{label} ID {tid}", (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)
    if SHOW:
        cv2.imshow("Filtered Incoming", frame)
        cv2.waitKey(1)

out.release()
cap.release()
cv2.destroyAllWindows()
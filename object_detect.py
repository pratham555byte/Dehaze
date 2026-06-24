import cv2
import numpy as np

from ultralytics import YOLO
import torch

# ---------------------------------------------------
# LOAD YOLO MODEL ONLY ONCE
# ---------------------------------------------------

model = YOLO("models/yolov8n.pt",task="detect")

_HAS_CUDA = bool(torch.cuda.is_available())
_DEVICE = 0 if _HAS_CUDA else "cpu"

# ---------------------------------------------------
# TRAFFIC LIGHT COLOR DETECTION
# ---------------------------------------------------

def detect_traffic_light_color(roi):

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # RED
    lower_red1 = np.array([0,120,70])
    upper_red1 = np.array([10,255,255])

    lower_red2 = np.array([170,120,70])
    upper_red2 = np.array([180,255,255])

    # GREEN
    lower_green = np.array([40,40,40])
    upper_green = np.array([90,255,255])

    # YELLOW
    lower_yellow = np.array([15,150,150])
    upper_yellow = np.array([35,255,255])

    # Masks
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = red_mask1 + red_mask2

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    yellow_mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow
    )

    # Pixel count
    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)

    threshold = 20

    if (
        red_pixels > green_pixels
        and red_pixels > yellow_pixels
        and red_pixels > threshold
    ):
        return "RED"

    elif (
        green_pixels > red_pixels
        and green_pixels > yellow_pixels
        and green_pixels > threshold
    ):
        return "GREEN"

    elif (
        yellow_pixels > red_pixels
        and yellow_pixels > green_pixels
        and yellow_pixels > threshold
    ):
        return "YELLOW"

    return "UNKNOWN"


# ---------------------------------------------------
# MAIN DETECTION FUNCTION
# ---------------------------------------------------

def process_frame(frame, is_video=False):

    """
    Input:
        frame -> OpenCV image

    Returns:
        annotated_frame
        detections_list
    """

    detections = []

    # YOLO detection
    results = model(
        frame,
        imgsz=320,
        device=_DEVICE,
        half=_HAS_CUDA,
        verbose=False
    )

    annotated = frame.copy()

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        conf = float(box.conf[0])

        # ---------------------------------------------------
        # CONFIDENCE THRESHOLD FILTER
        # ---------------------------------------------------
        # Only consider detections with confidence > 0.4
        if conf <= 0.4:
            continue

        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        # Only allow standard road-relevant objects to prevent clutter
        _ROAD_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light", "stop sign"}
        if label not in _ROAD_CLASSES:
            continue



        color = (0,255,0)

        traffic_light_color = None

        # ---------------------------------------------------
        # TRAFFIC LIGHT ANALYSIS
        # ---------------------------------------------------

        if label == "traffic light":

            roi = frame[y1:y2, x1:x2]

            if roi.size != 0 and roi.shape[0] > 15 and roi.shape[1] > 15:

                traffic_light_color = detect_traffic_light_color(roi)

                if traffic_light_color == "RED":
                    color = (0,0,255)

                elif traffic_light_color == "GREEN":
                    color = (0,255,0)

                elif traffic_light_color == "YELLOW":
                    color = (0,255,255)

        # ---------------------------------------------------
        # DRAW BOX
        # ---------------------------------------------------

        if not is_video:
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2
            )
            cv2.putText(
                annotated,
                f"{label}{'' if not traffic_light_color else f': {traffic_light_color}'} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # ---------------------------------------------------
        # SAVE DETECTION DATA
        # ---------------------------------------------------

        detections.append({

            "label": label,

            "confidence": round(conf, 2),

            "bbox": [x1, y1, x2, y2],

            # Match pipeline expectation.
            "traffic_light_color": traffic_light_color

        })

        # ---------------------------------------------------
        # DISTANCE ESTIMATION
        # ---------------------------------------------------

        height = y2 - y1
        # Simple heuristic: distance proportional to 1/height (assuming fixed focal length)
        distance = 50.0 / max(height, 1)  # meters, rough estimate
        detections[-1]["distance"] = round(distance, 2)

    return annotated, detections
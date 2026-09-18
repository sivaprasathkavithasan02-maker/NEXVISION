"""
VisionAI - Real-Time YOLOv8 Object Detection & Sensitive Distance Engine
Calibrated specifically for assistive wearable cameras and webcam focal fields.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from src.vision.clothing_analyzer import ClothingAnalyzer


class ObjectDetector:
    # Set of small handheld objects with different optical scale profiles
    SMALL_OBJECTS = {
        "cell phone", "mobile phone", "phone", "mobile", "pen", "pencil", "toothbrush",
        "bottle", "cup", "glass", "wine glass", "mouse", "remote", "book", "watch", "clock",
        "apple", "banana", "scissors", "fork", "knife", "spoon", "bowl", "orange", "sandwich"
    }

    def __init__(self, model_name: str = "yolov8n.pt", conf_thresh: float = 0.25):
        self.conf_thresh = conf_thresh
        print(f"[NexVision Detector] Loading YOLO model: {model_name} (conf={conf_thresh})...")
        self.model = YOLO(model_name)
        self.clothing_analyzer = ClothingAnalyzer()
        print("[NexVision Detector] Model and Clothing Analyzer loaded successfully.")

    def refine_label(self, raw_label: str, box_w: int, box_h: int, area_ratio: float, conf: float) -> str:
        """
        Refines raw COCO labels into natural, context-accurate object names.
        Fixes known COCO dataset misclassifications:
        - 'toothbrush' -> 'pen' (COCO lacks pen class; slender writing tools get forced into toothbrush)
        - 'cell phone' -> 'mobile phone'
        - 'dining table' -> 'table'
        - 'wine glass' -> 'glass'
        - 'potted plant' -> 'plant'
        - 'couch' -> 'sofa'
        - 'clock' -> 'watch' (if small/handheld)
        - 'tv' -> 'monitor' (if desktop size / close range)
        """
        raw = raw_label.lower().strip()

        # 1. Toothbrush -> Pen / Pencil (COCO lacks 'pen' class)
        if raw == "toothbrush":
            return "pen"

        # 2. Cell Phone -> Mobile Phone
        if raw == "cell phone":
            return "mobile phone"

        # 3. Dining Table -> Table / Desk
        if raw == "dining table":
            return "table"

        # 4. Wine Glass -> Glass
        if raw == "wine glass":
            return "glass"

        # 5. Couch -> Sofa
        if raw == "couch":
            return "sofa"

        # 6. Potted Plant -> Plant
        if raw == "potted plant":
            return "plant"

        # 7. Clock -> Watch (if small handheld item)
        if raw == "clock" and area_ratio < 0.05:
            return "watch"

        # 8. TV -> Monitor (if desktop size / close range)
        if raw == "tv" and area_ratio < 0.20:
            return "monitor"

        return raw_label

    def estimate_physical_distance(self, area_ratio: float, box_h: int, frame_h: int, label: str = "") -> tuple:
        """
        Calculates real-world distance in feet and meters calibrated for assistive vision.
        Returns: (distance_category, distance_string, distance_ft_float)
        """
        h_ratio = box_h / frame_h
        is_small = label.lower() in self.SMALL_OBJECTS

        if is_small:
            # Sensitive calibration for small handheld items (phone, pen, bottle, cup)
            if h_ratio >= 0.28 or area_ratio >= 0.05:
                return "very close", "1 foot", 1.0
            elif h_ratio >= 0.18 or area_ratio >= 0.025:
                return "very close", "1.5 feet", 1.5
            elif h_ratio >= 0.12 or area_ratio >= 0.015:
                return "close", "2 feet", 2.0
            elif h_ratio >= 0.07 or area_ratio >= 0.008:
                return "close", "3 feet", 3.0
            elif h_ratio >= 0.04:
                return "medium", "4.5 feet", 4.5
            else:
                return "far", "over 6 feet", 6.0
        else:
            # Calibration for persons, furniture, doors, vehicles, laptops, backpacks
            if h_ratio >= 0.45 or area_ratio >= 0.15:
                return "very close", "1 foot", 1.0
            elif h_ratio >= 0.30 or area_ratio >= 0.08:
                return "very close", "1.5 feet", 1.5
            elif h_ratio >= 0.20 or area_ratio >= 0.04:
                return "close", "2 feet", 2.0
            elif h_ratio >= 0.12 or area_ratio >= 0.02:
                return "close", "3 feet", 3.0
            elif h_ratio >= 0.06:
                return "medium", "4.5 feet", 4.5
            else:
                return "far", "over 6 feet", 6.0

    def detect(self, frame: np.ndarray) -> list:
        """
        Runs comprehensive object & clothing detection on camera frame.
        Returns detections with precise spatial coordinates, distances, and apparel info.
        """
        height, width = frame.shape[:2]
        frame_area = height * width

        results = self.model(frame, conf=self.conf_thresh, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            raw_label = self.model.names[cls_id]

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            box_h = y2 - y1
            box_w = x2 - x1

            box_area = box_w * box_h
            area_ratio = box_area / frame_area

            # Context-Aware Label Refinement (Fixes COCO gaps like pen -> toothbrush)
            label = self.refine_label(raw_label, box_w, box_h, area_ratio, conf)

            # Horizontal sector: Left (0-35%), Center (35-65%), Right (65-100%)
            norm_cx = cx / width
            if norm_cx < 0.35:
                pos_h = "left"
            elif norm_cx > 0.65:
                pos_h = "right"
            else:
                pos_h = "center"

            # Physical Distance Estimation
            dist_cat, dist_feet_str, dist_val_ft = self.estimate_physical_distance(area_ratio, box_h, height, label)

            if dist_cat == "very close":
                risk_level = "danger"
            elif dist_cat == "close":
                risk_level = "warning"
            else:
                risk_level = "safe"

            # Apparel Analysis for Persons
            apparel_info = None
            if raw_label == "person":
                apparel_info = self.clothing_analyzer.analyze_person(frame, (x1, y1, x2, y2))

            detections.append({
                "label": label,
                "raw_label": raw_label,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "horizontal_pos": pos_h,
                "distance": dist_cat,
                "distance_str": dist_feet_str,
                "distance_ft": dist_val_ft,
                "risk_level": risk_level,
                "area_ratio": area_ratio,
                "apparel": apparel_info
            })

        return detections

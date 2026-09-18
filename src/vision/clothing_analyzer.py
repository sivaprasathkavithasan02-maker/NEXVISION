"""
VisionAI - Smart Apparel & Clothing Analyzer
Extracts garment patterns, colors, and outfit styles (dress, shirt, pants, suit)
from detected individuals in real time.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class ClothingAnalyzer:
    def __init__(self):
        pass

    def analyze_person(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Analyzes the person's bounding box to determine outfit style and dominant colors.
        Returns apparel dictionary with garment descriptions.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Clamp coordinates inside frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        box_h = y2 - y1
        box_w = x2 - x1

        if box_h < 40 or box_w < 20:
            return {
                "type": "general",
                "upper_color": "unknown",
                "lower_color": "unknown",
                "description": "wearing casual clothing"
            }

        person_crop = frame[y1:y2, x1:x2]

        # Region Segmentation:
        # Torso / Upper Body: 22% to 55% of height, center 70% of width
        uy1 = int(box_h * 0.22)
        uy2 = int(box_h * 0.55)
        ux1 = int(box_w * 0.15)
        ux2 = int(box_w * 0.85)
        upper_crop = person_crop[uy1:uy2, ux1:ux2]

        # Lower Body / Legs / Dress Skirt: 55% to 88% of height, center 70% of width
        ly1 = int(box_h * 0.55)
        ly2 = int(box_h * 0.88)
        lx1 = int(box_w * 0.15)
        lx2 = int(box_w * 0.85)
        lower_crop = person_crop[ly1:ly2, lx1:lx2]

        upper_color = self._extract_dominant_color(upper_crop)
        lower_color = self._extract_dominant_color(lower_crop)

        # Style Classification (Dress vs Two-Piece)
        # If upper and lower color match closely and person is upright, it's often a dress/gown/suit
        if upper_color == lower_color and upper_color not in ["unknown", "skin"]:
            if upper_color in ["black", "dark"]:
                description = f"wearing a {upper_color} suit or outfit"
            else:
                description = f"wearing a {upper_color} dress"
            garment_type = "dress"
        else:
            upper_desc = f"{upper_color} top" if upper_color != "unknown" else "top"
            lower_desc = f"{lower_color} bottoms" if lower_color != "unknown" else "bottoms"
            description = f"wearing a {upper_desc} and {lower_desc}"
            garment_type = "two-piece"

        return {
            "type": garment_type,
            "upper_color": upper_color,
            "lower_color": lower_color,
            "description": description
        }

    def _extract_dominant_color(self, crop: np.ndarray) -> str:
        """
        Calculates the prominent garment color, ignoring extreme highlights, shadows, and skin tones.
        """
        if crop is None or crop.size < 60:
            return "unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Filter out skin tones: hue 0-25 with medium saturation and value
        skin_mask = (h >= 0) & (h <= 25) & (s >= 35) & (s <= 170) & (v >= 70)
        valid_mask = ~skin_mask

        if np.sum(valid_mask) < 20:
            valid_h = h
            valid_s = s
            valid_v = v
        else:
            valid_h = h[valid_mask]
            valid_s = s[valid_mask]
            valid_v = v[valid_mask]

        mean_v = float(np.median(valid_v))
        mean_s = float(np.median(valid_s))
        mean_h = float(np.median(valid_h))

        # Check achromatic colors first (black, white, grey)
        if mean_v < 50:
            return "black"
        elif mean_v > 195 and mean_s < 45:
            return "white"
        elif mean_s < 45:
            return "grey"

        # Check chromatic hues
        if mean_h < 10 or mean_h >= 165:
            return "red"
        elif 10 <= mean_h < 25:
            return "orange"
        elif 25 <= mean_h < 36:
            return "yellow"
        elif 36 <= mean_h < 85:
            return "green"
        elif 85 <= mean_h < 130:
            return "blue"
        elif 130 <= mean_h < 150:
            return "purple"
        elif 150 <= mean_h < 165:
            return "pink"

        return "neutral"

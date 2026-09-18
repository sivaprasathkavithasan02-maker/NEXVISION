"""
VisionAI - Smart Text Recognition (OCR) Engine
Extracts printed text, book pages, medicine labels, and signage using native hardware acceleration.
"""

import os
import subprocess
import cv2
import numpy as np


class TextReader:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.ps_script = os.path.join(self.script_dir, "run_ocr.ps1")
        self.temp_img_path = os.path.abspath(os.path.join(self.script_dir, "..", "..", "ocr_snapshot.png"))

    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhances contrast and sharpness to maximize OCR readability on camera feeds.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Adaptive histogram equalization (CLAHE) for varying lighting conditions
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced

    def extract_text(self, frame: np.ndarray, focus_center: bool = False) -> str:
        """
        Extracts text from the provided video frame.
        If focus_center is True, crops the central 70% of the frame (typical user reading alignment).
        """
        h, w = frame.shape[:2]
        
        if focus_center:
            # Crop center reading box
            ymin, ymax = int(h * 0.15), int(h * 0.85)
            xmin, xmax = int(w * 0.15), int(w * 0.85)
            roi = frame[ymin:ymax, xmin:xmax]
        else:
            roi = frame

        processed = self.preprocess_image(roi)
        cv2.imwrite(self.temp_img_path, processed)

        try:
            # Call Windows Native OCR bridge
            cmd = [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-File", self.ps_script,
                "-ImagePath", self.temp_img_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            raw_text = result.stdout.strip()

            # Clean formatting
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            cleaned_text = " ".join(lines)
            return cleaned_text

        except subprocess.TimeoutExpired:
            print("[OCR] OCR request timed out.")
            return ""
        except Exception as e:
            print(f"[OCR Error] {e}")
            return ""
        finally:
            if os.path.exists(self.temp_img_path):
                try:
                    os.remove(self.temp_img_path)
                except OSError:
                    pass

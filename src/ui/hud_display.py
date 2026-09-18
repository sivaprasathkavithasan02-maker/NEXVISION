"""
VisionAI - Augmented Reality Smart Glass HUD Simulator
Renders real-time computer vision perception, calibrated physical distances,
collision alerts, apparel details, proximity radar, and live subtitles.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Optional


class HUDDisplay:
    # Color Palette (BGR format)
    COLOR_BG_DARK = (20, 20, 24)
    COLOR_CYAN = (240, 200, 0)
    COLOR_GREEN = (60, 220, 90)
    COLOR_YELLOW = (40, 210, 240)
    COLOR_RED = (50, 50, 240)
    COLOR_WHITE = (245, 245, 245)
    COLOR_MUTED = (140, 140, 140)
    COLOR_ACCENT = (255, 140, 0)
    COLOR_PURPLE = (200, 80, 200)

    def __init__(self):
        self.scan_line_y = 0
        self.scan_direction = 1

    def render(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        sector_status: Dict[str, str],
        active_mode: str,
        current_subtitle: str,
        voice_status: str,
        fps: float,
        target_obj: Optional[Dict] = None,
        search_target_name: Optional[str] = None,
        ocr_active: bool = False,
        pathway_status: str = "PATHWAY CLEAR AHEAD"
    ) -> np.ndarray:
        """
        Renders the complete HUD overlay onto the camera frame.
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 1. Central Navigation Corridor Guide
        self._draw_nav_corridor(canvas, w, h)

        # 2. Object Detections & Exact Physical Distance Badges
        self._draw_detections(canvas, detections, target_obj)

        # 3. OCR Scanning Reticle
        if ocr_active:
            self._draw_ocr_reticle(canvas, w, h)

        # 4. Top Header Banner (NexVision & Voice Telemetry)
        self._draw_header(canvas, w, h, active_mode, fps, voice_status, len(detections))

        # 5. Obstacle Radar & Pathway Directional Widget (Top Right)
        self._draw_radar(canvas, w, sector_status, pathway_status)

        # 6. Bottom Subtitle & Speech Caption Bar
        self._draw_subtitles(canvas, w, h, current_subtitle)

        # 7. Bottom Hotkey Help Bar
        self._draw_footer(canvas, w, h, search_target_name)

        return canvas

    def _draw_nav_corridor(self, canvas: np.ndarray, w: int, h: int):
        """Draws subtle dashed safe path markers in the center walking zone."""
        x_left = int(w * 0.35)
        x_right = int(w * 0.65)
        
        cv2.line(canvas, (x_left, h - 80), (int(w * 0.40), int(h * 0.4)), (60, 60, 60), 1, cv2.LINE_AA)
        cv2.line(canvas, (x_right, h - 80), (int(w * 0.60), int(h * 0.4)), (60, 60, 60), 1, cv2.LINE_AA)
        
        cx, cy = w // 2, h // 2
        cv2.drawMarker(canvas, (cx, cy), (100, 100, 100), cv2.MARKER_CROSS, 16, 1)

    def _draw_detections(self, canvas: np.ndarray, detections: List[Dict], target_obj: Optional[Dict]):
        """Draws bounding boxes, exact physical distances, and hazard warnings."""
        for obj in detections:
            x1, y1, x2, y2 = obj["bbox"]
            label = obj["label"]
            risk = obj["risk_level"]
            dist_cat = obj["distance"]
            dist_str = obj.get("distance_str", dist_cat)
            apparel = obj.get("apparel")
            is_target = (target_obj and obj["bbox"] == target_obj["bbox"])

            # Highlight very close objects with high-visibility red
            if is_target:
                box_color = (0, 165, 255)  # Orange
                thickness = 3
            elif dist_cat == "very close":
                box_color = self.COLOR_RED   # Danger Red
                thickness = 3
            elif dist_cat == "close":
                box_color = self.COLOR_YELLOW # Caution Yellow
                thickness = 2
            elif label == "person":
                box_color = self.COLOR_PURPLE
                thickness = 2
            else:
                box_color = self.COLOR_CYAN
                thickness = 1

            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)

            # Corner brackets
            corner_len = min(18, (x2 - x1) // 4, (y2 - y1) // 4)
            if corner_len > 3:
                cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), box_color, thickness + 1)
                cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), box_color, thickness + 1)
                cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), box_color, thickness + 1)
                cv2.line(canvas, (x2, y1), (x2, y1 + corner_len), box_color, thickness + 1)

            # Prominent distance badge
            if is_target:
                tag_text = f"★ TARGET: {label.upper()} [{dist_str.upper()}] ★"
            elif dist_cat == "very close":
                tag_text = f"! {label.upper()} [{dist_str.upper()} - VERY CLOSE] !"
            else:
                tag_text = f"{label.upper()} [{dist_str.upper()}]"

            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tag_y1 = max(0, y1 - th - 6)
            cv2.rectangle(canvas, (x1, tag_y1), (x1 + tw + 8, y1), box_color, -1)
            text_color = (255, 255, 255) if (dist_cat == "very close" or is_target or label == "person") else (0, 0, 0)
            cv2.putText(canvas, tag_text, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

            # Secondary Tag for Clothing / Dress Details
            if label == "person" and apparel:
                app_desc = apparel["description"].upper()
                (aw, ah), _ = cv2.getTextSize(app_desc, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                app_y1 = y2 + 4
                app_y2 = app_y1 + ah + 6
                if app_y2 < canvas.shape[0] - 80:
                    cv2.rectangle(canvas, (x1, app_y1), (x1 + aw + 8, app_y2), (40, 20, 50), -1)
                    cv2.rectangle(canvas, (x1, app_y1), (x1 + aw + 8, app_y2), self.COLOR_PURPLE, 1)
                    cv2.putText(canvas, app_desc, (x1 + 4, app_y1 + ah + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 210, 255), 1, cv2.LINE_AA)

    def _draw_ocr_reticle(self, canvas: np.ndarray, w: int, h: int):
        """Draws dynamic AR scanning box for reading documents or signs."""
        ymin, ymax = int(h * 0.20), int(h * 0.80)
        xmin, xmax = int(w * 0.15), int(w * 0.85)

        overlay = canvas.copy()
        cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), self.COLOR_CYAN, 2)
        
        self.scan_line_y += 6 * self.scan_direction
        if self.scan_line_y >= (ymax - ymin):
            self.scan_direction = -1
        elif self.scan_line_y <= 0:
            self.scan_direction = 1

        curr_laser_y = ymin + self.scan_line_y
        cv2.line(overlay, (xmin, curr_laser_y), (xmax, curr_laser_y), (0, 255, 255), 2)
        cv2.putText(overlay, "TEXT SCANNER ACTIVE", (xmin + 10, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_CYAN, 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)

    def _draw_header(self, canvas: np.ndarray, w: int, h: int, active_mode: str, fps: float, voice_status: str, total_objs: int):
        """Draws top telemetry bar with NexVision branding and voice input status."""
        bar_h = 44
        sub_img = canvas[0:bar_h, 0:w]
        dark_rect = np.full_like(sub_img, 20)
        cv2.addWeighted(sub_img, 0.3, dark_rect, 0.7, 0, sub_img)
        canvas[0:bar_h, 0:w] = sub_img

        # Brand Title
        cv2.putText(canvas, "NEXVISION", (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, self.COLOR_CYAN, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"OBJS: {total_objs}", (155, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_MUTED, 1, cv2.LINE_AA)

        # Center Active Mode
        mode_str = f"MODE: {active_mode.upper()}"
        mode_color = self.COLOR_GREEN if "NAV" in active_mode else (self.COLOR_RED if "EMERGENCY" in active_mode else self.COLOR_YELLOW)
        box_w = 120
        cv2.rectangle(canvas, (w // 2 - box_w, 8), (w // 2 + box_w, 36), mode_color, 1)
        (mw, _), _ = cv2.getTextSize(mode_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(canvas, mode_str, (w // 2 - mw // 2, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1, cv2.LINE_AA)

        # Voice Status Telemetry & FPS
        v_color = self.COLOR_GREEN if any(k in voice_status for k in ["Online", "Ready"]) else (
            self.COLOR_CYAN if "Heard" in voice_status else (
                self.COLOR_YELLOW if "Hearing" in voice_status or "Interpreting" in voice_status else self.COLOR_MUTED
            )
        )
        voice_text = f"VOICE: {voice_status}"
        if len(voice_text) > 38:
            voice_text = voice_text[:35] + "..."
        cv2.putText(canvas, voice_text, (max(w // 2 + 135, w - 340), 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, v_color, 1, cv2.LINE_AA)

        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(canvas, fps_text, (w - 75, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_WHITE, 1, cv2.LINE_AA)

    def _draw_radar(self, canvas: np.ndarray, w: int, sector_status: Dict[str, str], pathway_status: str = "PATHWAY CLEAR AHEAD"):
        """Draws 3-sector proximity radar and pathway direction indicator."""
        radar_x = w - 190
        radar_y = 52
        rw, rh = 54, 20

        sectors = [
            ("L", "left", radar_x),
            ("C", "center", radar_x + rw + 4),
            ("R", "right", radar_x + (rw + 4) * 2)
        ]

        for label, key, sx in sectors:
            status = sector_status.get(key, "CLEAR")
            if status == "DANGER":
                bg = self.COLOR_RED
            elif status == "CAUTION":
                bg = self.COLOR_YELLOW
            else:
                bg = (40, 55, 40)

            cv2.rectangle(canvas, (sx, radar_y), (sx + rw, radar_y + rh), bg, -1)
            cv2.rectangle(canvas, (sx, radar_y), (sx + rw, radar_y + rh), (100, 100, 100), 1)
            text = f"{label}:{status[:3]}"
            cv2.putText(canvas, text, (sx + 4, radar_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Pathway Guidance Badge
        pw_y = radar_y + rh + 4
        pw_w = (rw + 4) * 3 - 4
        if "CLEAR" in pathway_status:
            pw_bg = (30, 65, 30)
            pw_fg = self.COLOR_GREEN
        elif "STOP" in pathway_status:
            pw_bg = (20, 20, 150)
            pw_fg = self.COLOR_WHITE
        else:
            pw_bg = (20, 80, 120)
            pw_fg = self.COLOR_YELLOW

        cv2.rectangle(canvas, (radar_x, pw_y), (radar_x + pw_w, pw_y + 18), pw_bg, -1)
        cv2.rectangle(canvas, (radar_x, pw_y), (radar_x + pw_w, pw_y + 18), pw_fg, 1)
        (tw, _), _ = cv2.getTextSize(pathway_status, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
        cv2.putText(canvas, pathway_status, (radar_x + max(4, (pw_w - tw) // 2), pw_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34, pw_fg, 1, cv2.LINE_AA)

    def _draw_subtitles(self, canvas: np.ndarray, w: int, h: int, subtitle: str):
        """Draws live spoken audio captions at bottom of screen."""
        if not subtitle:
            return

        bar_h = 42
        y1 = h - 78
        y2 = y1 + bar_h

        sub_img = canvas[y1:y2, 40:w-40]
        dark_pill = np.full_like(sub_img, 15)
        cv2.addWeighted(sub_img, 0.2, dark_pill, 0.8, 0, sub_img)
        canvas[y1:y2, 40:w-40] = sub_img
        cv2.rectangle(canvas, (40, y1), (w - 40, y2), self.COLOR_CYAN, 1)

        display_text = f"AUDIO: \"{subtitle}\""
        if len(display_text) > 80:
            display_text = display_text[:77] + "..."

        cv2.putText(canvas, display_text, (55, y1 + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.COLOR_WHITE, 1, cv2.LINE_AA)

    def _draw_footer(self, canvas: np.ndarray, w: int, h: int, search_target: Optional[str]):
        """Draws keyboard controls banner at bottom."""
        footer_h = 28
        y1 = h - footer_h
        cv2.rectangle(canvas, (0, y1), (w, h), (12, 12, 16), -1)

        if search_target:
            help_text = f"[ACTIVE SEARCH: {search_target.upper()}]  Press [SPACE] to cancel | [Q] to quit"
            color = (0, 200, 255)
        else:
            help_text = "[I] Inventory | [F] Find | [R] Read | [S] Describe | [1/2/3] Front/L/R | [O] Radar | [E] Emergency | [Q] Quit"
            color = self.COLOR_MUTED

        cv2.putText(canvas, help_text, (15, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

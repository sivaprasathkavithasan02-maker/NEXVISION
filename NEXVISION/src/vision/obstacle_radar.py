import time
from typing import List, Dict, Tuple


class ObstacleRadar:
    """
    VisionAI / NexVision - Obstacle Radar & Collision Warning Engine
    Proactively detects objects in 1-3 feet range, announces exact distances,
    and provides directional pathway guidance ('move left', 'move right').
    Enforces a strict 1-2 repetition limit so visually impaired users are not irritated.
    """
    def __init__(self):
        # Tracking to limit announcements to at most 1-2 times per obstacle
        self.announced_hazards = {}  # key -> {"count": int, "last_time": float, "dist_cat": str}
        self.current_pathway = "PATHWAY CLEAR AHEAD"

    def evaluate_hazards(self, detections: List[Dict]) -> Tuple[Dict, str, bool]:
        """
        Evaluates detected objects across Left, Center, and Right spatial sectors.
        Produces directional pathway advice and enforces max 1-2 announcements per obstacle.
        
        Returns:
            sector_status: {"left": status, "center": status, "right": status}
            audio_warning: String warning announcement (or "" if already warned twice)
            is_critical: True if object is very close (triggers immediate chime + interrupt)
        """
        sector_status = {
            "left": "CLEAR",
            "center": "CLEAR",
            "right": "CLEAR"
        }

        very_close_objects = []
        close_objects = []

        for obj in detections:
            h_pos = obj["horizontal_pos"]
            dist_cat = obj["distance"]
            label = obj["label"]
            raw_label = obj.get("raw_label", label).lower()

            # Ignore small handheld items (pen, phone, mouse, watch) from triggering collision warnings
            handheld_names = {"cell phone", "mobile phone", "phone", "pen", "pencil", "toothbrush", "mouse", "remote", "watch", "fork", "knife", "spoon", "glass"}
            if (raw_label in handheld_names or label.lower() in handheld_names) and obj.get("area_ratio", 0) < 0.25:
                continue

            # Update sector status widget
            if dist_cat == "very close":
                sector_status[h_pos] = "DANGER"
                very_close_objects.append(obj)
            elif dist_cat == "close":
                if sector_status[h_pos] != "DANGER":
                    sector_status[h_pos] = "CAUTION"
                close_objects.append(obj)
            elif sector_status[h_pos] == "CLEAR":
                sector_status[h_pos] = "OCCUPIED"

        # Determine pathway recommendation
        if sector_status["center"] == "CLEAR":
            self.current_pathway = "PATHWAY CLEAR AHEAD"
        elif sector_status["left"] == "CLEAR" and sector_status["right"] != "CLEAR":
            self.current_pathway = "MOVE LEFT <<<"
        elif sector_status["right"] == "CLEAR" and sector_status["left"] != "CLEAR":
            self.current_pathway = "MOVE RIGHT >>>"
        elif sector_status["left"] == "CLEAR" and sector_status["right"] == "CLEAR":
            self.current_pathway = "STEP RIGHT >>>"
        else:
            self.current_pathway = "STOP - PATH BLOCKED"

        audio_warning = ""
        is_critical = False
        hazard_key = None
        current_dist_cat = None

        # 1. CRITICAL: Any object is VERY CLOSE (1 to 1.5 feet)
        if very_close_objects:
            very_close_objects.sort(key=lambda x: x["area_ratio"], reverse=True)
            closest = very_close_objects[0]
            label = closest["label"].capitalize()
            pos = closest["horizontal_pos"]
            dist_str = closest.get("distance_str", "1 foot")
            hazard_key = f"{label}_{pos}"
            current_dist_cat = "very close"

            if pos == "center":
                if sector_status["left"] == "CLEAR" and sector_status["right"] != "CLEAR":
                    advice = "Step left."
                elif sector_status["right"] == "CLEAR":
                    advice = "Step right."
                else:
                    advice = "Stop."
                audio_warning = f"Caution! {label} directly ahead, {dist_str} away. {advice}"
            else:
                advice = "Keep right." if pos == "left" else "Keep left."
                audio_warning = f"Warning: {label} very close on your {pos}, {dist_str} away. {advice}"

            is_critical = True

        # 2. APPROACHING: Object is CLOSE (2 to 3 feet range)
        elif close_objects:
            close_objects.sort(key=lambda x: x["area_ratio"], reverse=True)
            closest = close_objects[0]
            label = closest["label"].capitalize()
            pos = closest["horizontal_pos"]
            dist_str = closest.get("distance_str", "2 feet")
            hazard_key = f"{label}_{pos}"
            current_dist_cat = "close"

            if pos == "center":
                if sector_status["left"] == "CLEAR" and sector_status["right"] != "CLEAR":
                    advice = "Move left, pathway is clear."
                elif sector_status["right"] == "CLEAR":
                    advice = "Move right, pathway is clear."
                else:
                    advice = "Slow down, pathway is narrow."
                audio_warning = f"Notice: {label} ahead, {dist_str} away. {advice}"
            else:
                side_advice = "Keep right." if pos == "left" else "Keep left."
                audio_warning = f"{label} on your {pos}, {dist_str} away. {side_advice}"

            is_critical = False

        # Apply repetition limiter: tell at most once or twice
        curr_time = time.time()
        
        # Clean up stale hazards from tracking if not detected for > 5s
        stale_keys = [k for k, v in self.announced_hazards.items() if curr_time - v["last_time"] > 5.0]
        for sk in stale_keys:
            if sk != hazard_key:
                del self.announced_hazards[sk]

        if hazard_key and audio_warning:
            hist = self.announced_hazards.get(hazard_key)
            if hist is None:
                # First announcement
                self.announced_hazards[hazard_key] = {
                    "count": 1,
                    "last_time": curr_time,
                    "dist_cat": current_dist_cat
                }
            elif hist["dist_cat"] == "close" and current_dist_cat == "very close":
                # Escalated to critical danger: immediately speak and reset count!
                self.announced_hazards[hazard_key] = {
                    "count": 1,
                    "last_time": curr_time,
                    "dist_cat": current_dist_cat
                }
            elif hist["count"] == 1:
                # Second announcement: only after at least 2.5 seconds pause
                if (curr_time - hist["last_time"]) >= 2.5:
                    hist["count"] = 2
                    hist["last_time"] = curr_time
                    # Gentle reminder phrasing for second time
                    if "Caution!" in audio_warning:
                        audio_warning = audio_warning.replace("Caution!", "Reminder:")
                else:
                    audio_warning = ""  # Too fast, suppress
            elif hist["count"] >= 2:
                # Already spoken twice: suppress to prevent irritating visually impaired user!
                if (curr_time - hist["last_time"]) >= 8.0:
                    # After 8s persistent blockage, give a single quiet reminder
                    hist["count"] = 2
                    hist["last_time"] = curr_time
                else:
                    audio_warning = ""

        return sector_status, audio_warning, is_critical

    def get_pathway_guidance(self, detections: List[Dict]) -> Tuple[str, str]:
        """
        Returns (pathway_badge_str, natural_speech_advice) for live surroundings reports.
        """
        left_clear = not any(d["horizontal_pos"] == "left" and d["distance"] in ("very close", "close") for d in detections)
        center_clear = not any(d["horizontal_pos"] == "center" and d["distance"] in ("very close", "close") for d in detections)
        right_clear = not any(d["horizontal_pos"] == "right" and d["distance"] in ("very close", "close") for d in detections)

        if center_clear:
            return "PATHWAY CLEAR AHEAD", "Pathway directly ahead is completely clear."
        elif left_clear and not right_clear:
            return "MOVE LEFT <<<", "Center is blocked. Move to your left, left pathway is clear."
        elif right_clear and not left_clear:
            return "MOVE RIGHT >>>", "Center is blocked. Move to your right, right pathway is clear."
        elif left_clear and right_clear:
            return "STEP RIGHT >>>", "Obstacle ahead. Both left and right are open, step right to bypass."
        else:
            return "STOP - PATH BLOCKED", "Obstacles detected in all pathways ahead. Please pause."

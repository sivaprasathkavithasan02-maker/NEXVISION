"""
VisionAI - Misplaced Item & Apparel Search Engine
Guides visually impaired users to locate everyday objects, clothing (dresses, shirts, bags),
and accessories with spatial directional and distance feedback.
"""

import time
from typing import List, Dict, Optional, Tuple


class SearchMode:
    # Synonym mapping from natural speech to target categories
    SYNONYM_MAP = {
        "mobile": ["cell phone", "mobile phone", "mobile", "phone", "smartphone", "cellphone", "telephone", "iphone", "android"],
        "phone": ["cell phone", "mobile phone", "mobile", "phone", "smartphone", "cellphone", "telephone", "iphone", "android"],
        "cell phone": ["cell phone", "mobile phone", "mobile", "phone", "smartphone", "cellphone", "telephone", "iphone", "android"],
        "mobile phone": ["cell phone", "mobile phone", "mobile", "phone", "smartphone", "cellphone", "telephone", "iphone", "android"],
        "cellphone": ["cell phone", "mobile phone", "mobile", "phone", "smartphone", "cellphone", "telephone", "iphone", "android"],
        "cell": ["cell phone", "mobile phone", "mobile", "phone", "smartphone", "cellphone", "telephone", "iphone", "android"],
        "smartphone": ["cell phone", "mobile phone", "mobile", "phone", "smartphone", "cellphone", "telephone", "iphone", "android"],
        "iphone": ["cell phone", "mobile phone", "mobile", "phone", "smartphone", "cellphone", "telephone", "iphone", "android"],
        "android": ["cell phone", "mobile phone", "mobile", "phone", "smartphone", "cellphone", "telephone", "iphone", "android"],
        # Writing Tools (Re-maps COCO's misclassified toothbrush to pen)
        "pen": ["pen", "pencil", "toothbrush", "marker", "stylus", "ballpen"],
        "pencil": ["pencil", "pen", "toothbrush", "marker", "stylus"],
        "marker": ["marker", "pen", "pencil", "toothbrush"],
        "ballpen": ["pen", "pencil", "toothbrush"],
        "toothbrush": ["toothbrush", "pen", "pencil"],
        
        # Watches & Smartwatches (Re-maps small clocks to watch)
        "watch": ["watch", "clock", "smartwatch"],
        "smartwatch": ["watch", "clock", "smartwatch"],
        "wrist watch": ["watch", "clock"],
        "clock": ["clock", "watch"],
        
        # Spectacles & Glasses (Re-maps scissors/remote to glasses)
        "glasses": ["spectacles", "glasses", "specs", "sunglasses", "scissors"],
        "spectacles": ["spectacles", "glasses", "specs", "sunglasses", "scissors"],
        "specs": ["spectacles", "glasses", "specs", "sunglasses", "scissors"],
        "sunglasses": ["sunglasses", "glasses", "spectacles", "specs"],
        
        # Furniture, Displays & Office
        "table": ["table", "dining table", "desk"],
        "desk": ["table", "dining table", "desk"],
        "dining table": ["table", "dining table", "desk"],
        "sofa": ["sofa", "couch"],
        "couch": ["sofa", "couch"],
        "plant": ["plant", "potted plant"],
        "potted plant": ["plant", "potted plant"],
        "screen": ["monitor", "tv", "screen"],
        "monitor": ["monitor", "tv", "screen"],
        "tv": ["tv", "monitor", "screen"],
        
        "bottle": ["bottle", "flask", "water bottle"],
        "water bottle": ["bottle", "flask", "water bottle"],
        "cup": ["cup", "mug", "glass", "teacup", "coffee"],
        "mug": ["cup", "mug", "glass"],
        "glass": ["glass", "wine glass", "cup"],
        "keys": ["keys", "keychain", "remote", "scissors", "key"],
        "key": ["keys", "keychain", "remote", "key"],
        "wallet": ["handbag", "wallet", "book", "purse"],
        "purse": ["handbag", "wallet", "purse"],
        "bag": ["backpack", "handbag", "suitcase", "bag"],
        "backpack": ["backpack", "bag"],
        "laptop": ["laptop", "computer"],
        "computer": ["laptop", "tv", "computer", "monitor"],
        "mouse": ["mouse"],
        "keyboard": ["keyboard"],
        "remote": ["remote"],
        "book": ["book", "notebook"],
        "notebook": ["notebook", "book"],
        "scissors": ["scissors"],
        "umbrella": ["umbrella"],
        # Apparel & Clothing Synonyms
        "dress": ["dress", "person"],
        "clothes": ["dress", "shirt", "person", "tie"],
        "clothing": ["dress", "shirt", "person", "tie"],
        "shirt": ["shirt", "person", "tie"],
        "tshirt": ["shirt", "person"],
        "t-shirt": ["shirt", "person"],
        "jacket": ["jacket", "person"],
        "suit": ["suit", "person", "tie"],
        "pants": ["pants", "person"],
        "tie": ["tie"]
    }

    def __init__(self):
        self.active_target = None
        self.target_synonyms = []
        self.target_color = None
        self.last_guidance_time = 0
        self.guidance_interval = 2.0
        self.search_start_time = 0

        # Found state & non-repetitive announcement control
        self.target_found = False
        self.target_found_time = 0.0
        self.found_announced = False
        self.auto_complete_duration = 3.0  # Announce for 2-3s alone, then auto-complete cleanly
        self.not_found_reminded = False
        self.is_completed = False

    def start_search(self, target_query: str) -> str:
        """
        Activates search for a specific object or clothing item.
        Detects color qualifiers (e.g. 'red dress', 'blue bottle').
        """
        target_clean = target_query.lower().strip()
        for p in ["my ", "the ", "a "]:
            if target_clean.startswith(p):
                target_clean = target_clean[len(p):].strip()

        self.active_target = target_clean
        self.search_start_time = time.time()
        self.last_guidance_time = 0
        self.target_found = False
        self.target_found_time = 0.0
        self.found_announced = False
        self.not_found_reminded = False
        self.is_completed = False

        # Check for color qualifier in target query
        colors = ["red", "blue", "green", "black", "white", "yellow", "orange", "purple", "pink", "grey"]
        self.target_color = None
        for c in colors:
            if c in target_clean:
                self.target_color = c
                break

        # Find matching classes
        self.target_synonyms = [target_clean]
        for key, syns in self.SYNONYM_MAP.items():
            if key == target_clean or key in target_clean or target_clean in key:
                self.target_synonyms.extend(syns)
        self.target_synonyms = list(set([s.lower() for s in self.target_synonyms]))

        if self.active_target in ("mobile", "phone", "cell phone"):
            target_disp = "mobile phone"
        elif self.active_target in ("pen", "pencil", "marker", "ballpen", "toothbrush"):
            target_disp = "pen"
        elif self.active_target in ("watch", "smartwatch", "clock"):
            target_disp = "watch"
        elif self.active_target in ("glasses", "spectacles", "specs"):
            target_disp = "glasses"
        elif self.active_target in ("table", "desk", "dining table"):
            target_disp = "table"
        else:
            target_disp = self.active_target
        return f"Searching for {target_disp}. Please pan the camera slowly around the room."

    def stop_search(self) -> str:
        """Deactivates search mode."""
        prev = self.active_target
        self.active_target = None
        self.target_synonyms = []
        self.target_color = None
        self.target_found = False
        self.target_found_time = 0.0
        self.found_announced = False
        self.not_found_reminded = False
        self.is_completed = False
        if prev in ("mobile", "phone", "cell phone"):
            prev_disp = "mobile phone"
        elif prev in ("pen", "pencil", "marker", "ballpen", "toothbrush"):
            prev_disp = "pen"
        elif prev in ("watch", "smartwatch", "clock"):
            prev_disp = "watch"
        elif prev in ("glasses", "spectacles", "specs"):
            prev_disp = "glasses"
        elif prev in ("table", "desk", "dining table"):
            prev_disp = "table"
        else:
            prev_disp = prev
        return f"Stopped search for {prev_disp}." if prev else "Search stopped."

    def update(self, detections: List[Dict], frame_shape: Tuple[int, int], force_immediate: bool = False) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Scans detections for the target item or apparel.
        If force_immediate=True, returns immediate guidance without waiting for interval.
        """
        if not self.active_target:
            return None, None

        height, width = frame_shape[:2]
        now = time.time()

        matching_objects = []
        for obj in detections:
            label = obj["label"].lower()
            raw_label = obj.get("raw_label", label).lower()
            apparel = obj.get("apparel")

            is_match = False
            for syn in self.target_synonyms:
                s = syn.lower()
                if (s == label or s in label or label in s or 
                    s == raw_label or s in raw_label or raw_label in s):
                    is_match = True
                    break

            # Clothing / Dress Match on Person
            if label == "person" and apparel:
                app_desc = apparel["description"].lower()
                if any(syn in app_desc for syn in self.target_synonyms) or "dress" in self.target_synonyms or "clothes" in self.target_synonyms:
                    is_match = True
                if self.target_color:
                    if self.target_color in apparel["upper_color"] or self.target_color in apparel["lower_color"]:
                        is_match = True

            if is_match:
                matching_objects.append(obj)

        if self.active_target in ("mobile", "phone", "cell phone"):
            target_disp = "Mobile phone"
        elif self.active_target in ("pen", "pencil", "marker", "ballpen", "toothbrush"):
            target_disp = "Pen"
        elif self.active_target in ("watch", "smartwatch", "clock"):
            target_disp = "Watch"
        elif self.active_target in ("glasses", "spectacles", "specs"):
            target_disp = "Glasses"
        elif self.active_target in ("keys", "key", "keychain"):
            target_disp = "Keys"
        elif self.active_target in ("table", "desk", "dining table"):
            target_disp = "Table"
        else:
            target_disp = self.active_target.capitalize()

        if matching_objects:
            matching_objects.sort(key=lambda x: x["area_ratio"], reverse=True)
            target_obj = matching_objects[0]
            cx, cy = target_obj["center"]
            norm_cx = cx / width

            if norm_cx < 0.35:
                h_dir = "to your left"
            elif norm_cx > 0.65:
                h_dir = "to your right"
            else:
                h_dir = "directly in front of you"

            dist = target_obj["distance"]
            dist_str = target_obj.get("distance_str", dist)
            if dist == "very close":
                reach_cue = "Reach forward now."
            elif dist == "close":
                reach_cue = "One step forward."
            else:
                reach_cue = "Walk forward slowly."

            # If target was ALREADY announced:
            if self.found_announced:
                # Check if the 2 to 3 second display duration has elapsed
                if now - self.target_found_time >= self.auto_complete_duration:
                    self.stop_search()
                    self.is_completed = True
                    return None, None
                # Within the 2-3s window: return target_obj to HUD for visual reticle, but NO repeating voice!
                return target_obj, None

            # First time finding the target: announce once with 2-3s guidance!
            self.target_found = True
            self.target_found_time = now
            self.found_announced = True
            self.last_guidance_time = now

            if target_obj["label"] == "person" and target_obj.get("apparel"):
                garment = target_obj["apparel"]["description"]
                voice_guidance = f"Target found! {target_disp} located on person {h_dir}, {garment}. {reach_cue}"
            else:
                voice_guidance = f"Target found! {target_disp} detected {h_dir}, {dist_str} away. {reach_cue}"

            return target_obj, voice_guidance

        else:
            voice_guidance = None
            if self.target_found:
                # Target was previously seen/announced and is now picked up or out of view
                if now - self.target_found_time >= self.auto_complete_duration:
                    self.stop_search()
                    self.is_completed = True
                return None, None

            # Actively searching: do not spam continuously
            if not force_immediate:
                if not self.not_found_reminded and (now - self.search_start_time >= 8.0):
                    self.not_found_reminded = True
                    voice_guidance = f"Still searching for {target_disp.lower()}. Please pan camera slowly."
                elif now - self.search_start_time >= 20.0:
                    self.stop_search()
                    self.is_completed = True
                    voice_guidance = f"Search timed out. {target_disp} not detected."

            return None, voice_guidance

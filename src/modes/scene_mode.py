"""
VisionAI - Scene Description & Exhaustive Object Inventory Engine
Generates detailed, natural-language inventories of all objects, people, and apparel in view.
"""

from collections import Counter
from typing import List, Dict


class SceneMode:
    def describe_scene(self, detections: List[Dict]) -> str:
        """
        Synthesizes a concise environmental description including people's clothing.
        """
        if not detections:
            return "The path ahead appears clear. No prominent objects detected in camera view."

        # Group detections by horizontal position
        left_items = []
        center_items = []
        right_items = []

        for obj in detections:
            label = obj["label"]
            pos = obj["horizontal_pos"]
            apparel = obj.get("apparel")

            if label == "person" and apparel:
                desc = f"a person {apparel['description']}"
            else:
                desc = label

            if pos == "left":
                left_items.append(desc)
            elif pos == "right":
                right_items.append(desc)
            else:
                center_items.append(desc)

        def format_items(items):
            if not items:
                return ""
            # Separate complex items (like person with dress) from simple countables
            simple = [i for i in items if not i.startswith("a person ")]
            complex_items = [i for i in items if i.startswith("a person ")]

            formatted = []
            if simple:
                counts = Counter(simple)
                for item, cnt in counts.items():
                    if cnt == 1:
                        article = "an" if item[0] in "aeiou" else "a"
                        formatted.append(f"{article} {item}")
                    else:
                        formatted.append(f"{cnt} {item}s")

            formatted.extend(complex_items)

            if len(formatted) == 1:
                return formatted[0]
            return ", ".join(formatted[:-1]) + " and " + formatted[-1]

        summary_parts = []
        if center_items:
            summary_parts.append(f"Directly ahead: {format_items(center_items)}")
        if left_items:
            summary_parts.append(f"To your left: {format_items(left_items)}")
        if right_items:
            summary_parts.append(f"To your right: {format_items(right_items)}")

        return ". ".join(summary_parts) + "."

    def full_inventory_scan(self, detections: List[Dict]) -> str:
        """
        Performs an exhaustive verbal inventory of all objects and garments in the camera feed.
        """
        if not detections:
            return "Inventory scan complete. No objects detected in the current camera frame."

        total = len(detections)
        intro = f"Inventory scan: I detect {total} {'object' if total == 1 else 'objects'} in camera view."

        item_details = []
        for idx, obj in enumerate(detections, 1):
            label = obj["label"]
            pos = obj["horizontal_pos"]
            dist = obj["distance"]
            apparel = obj.get("apparel")

            loc_str = "in center" if pos == "center" else f"on your {pos}"

            if label == "person" and apparel:
                detail = f"a person {loc_str}, {dist}, {apparel['description']}"
            else:
                article = "an" if label[0] in "aeiou" else "a"
                detail = f"{article} {label} {loc_str} [{dist}]"

            item_details.append(detail)

        report = f"{intro} " + "; ".join(item_details) + "."
        return report

    def describe_surroundings(self, detections: List[Dict], pathway_advice: str = "") -> str:
        """
        Comprehensive spatial report of surroundings for visually impaired orientation.
        Answers voice command: "Describe my surroundings".
        """
        if not detections:
            return "Surroundings check: Your pathway is completely clear in all directions. No obstacles detected."

        front_objs = [d for d in detections if d["horizontal_pos"] == "center"]
        left_objs = [d for d in detections if d["horizontal_pos"] == "left"]
        right_objs = [d for d in detections if d["horizontal_pos"] == "right"]

        parts = []
        if front_objs:
            front_objs.sort(key=lambda x: x["area_ratio"], reverse=True)
            f = front_objs[0]
            parts.append(f"In front: {f['label']}, {f.get('distance_str', f['distance'])} away")
        else:
            parts.append("In front: clear")

        if left_objs:
            left_objs.sort(key=lambda x: x["area_ratio"], reverse=True)
            l = left_objs[0]
            parts.append(f"On left: {l['label']}, {l.get('distance_str', l['distance'])} away")
        else:
            parts.append("On left: clear")

        if right_objs:
            right_objs.sort(key=lambda x: x["area_ratio"], reverse=True)
            r = right_objs[0]
            parts.append(f"On right: {r['label']}, {r.get('distance_str', r['distance'])} away")
        else:
            parts.append("On right: clear")

        report = ". ".join(parts) + "."
        if pathway_advice:
            report += f" {pathway_advice}"
        return report

    def describe_front(self, detections: List[Dict], pathway_advice: str = "") -> str:
        """Answers: 'What is in front of me?'"""
        front_objs = [d for d in detections if d["horizontal_pos"] == "center"]
        if not front_objs:
            return "The path directly in front of you is completely clear."
        
        # If a handheld mobile/phone is in front, give it prominent focus
        front_objs.sort(key=lambda x: (
            1 if x.get("raw_label") in ("cell phone", "mobile phone") or x["label"] in ("mobile phone", "cell phone") else 0,
            x["area_ratio"]
        ), reverse=True)
        
        closest = front_objs[0]
        label = "mobile phone" if closest.get("raw_label") == "cell phone" or closest["label"] == "cell phone" else closest["label"]
        dist = closest.get("distance_str", closest["distance"])
        msg = f"Directly in front of you is a {label}, {dist} away."
        if pathway_advice:
            msg += f" {pathway_advice}"
        return msg

    def describe_left(self, detections: List[Dict]) -> str:
        """Answers: 'What is on my left?'"""
        left_objs = [d for d in detections if d["horizontal_pos"] == "left"]
        if not left_objs:
            return "The left side is completely clear."
        left_objs.sort(key=lambda x: (
            1 if x.get("raw_label") in ("cell phone", "mobile phone") or x["label"] in ("mobile phone", "cell phone") else 0,
            x["area_ratio"]
        ), reverse=True)
        closest = left_objs[0]
        label = "mobile phone" if closest.get("raw_label") == "cell phone" or closest["label"] == "cell phone" else closest["label"]
        dist = closest.get("distance_str", closest["distance"])
        return f"On your left side is a {label}, {dist} away."

    def describe_right(self, detections: List[Dict]) -> str:
        """Answers: 'What is on my right?'"""
        right_objs = [d for d in detections if d["horizontal_pos"] == "right"]
        if not right_objs:
            return "The right side is completely clear."
        right_objs.sort(key=lambda x: (
            1 if x.get("raw_label") in ("cell phone", "mobile phone") or x["label"] in ("mobile phone", "cell phone") else 0,
            x["area_ratio"]
        ), reverse=True)
        closest = right_objs[0]
        label = "mobile phone" if closest.get("raw_label") == "cell phone" or closest["label"] == "cell phone" else closest["label"]
        dist = closest.get("distance_str", closest["distance"])
        return f"On your right side is a {label}, {dist} away."


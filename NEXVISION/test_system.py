"""
VisionAI / NexVision - Automated System Verification & Diagnostics Test Suite
Validates: Object Detection, 2-3ft Approaching & Very Close Warnings with Pathway Guidance,
Non-repetitive Announcement Limiter (max 1-2 times), Apparel/Dress Analysis, 
Full Inventory Scan, Search Mode, and all SIH Voice Commands.
"""

import os
import sys
import time
import numpy as np
import cv2

# Add project root and libs to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LIBS_DIR = os.path.join(PROJECT_ROOT, "libs")
if LIBS_DIR not in sys.path:
    sys.path.insert(0, LIBS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vision.detector import ObjectDetector
from src.vision.clothing_analyzer import ClothingAnalyzer
from src.vision.obstacle_radar import ObstacleRadar
from src.modes.search_mode import SearchMode
from src.modes.scene_mode import SceneMode
from src.audio.speech_input import VoiceCommandListener


def test_system():
    print("\n" + "=" * 70)
    print("      RUNNING NEXVISION SIH PROTOTYPE VERIFICATION SUITE")
    print("=" * 70)

    # 1. Test Physical Distance Estimator in Detector
    print("\n[TEST 1/8] Testing Calibrated Distance Estimator...")
    detector = ObjectDetector(model_name="yolov8n.pt", conf_thresh=0.3)
    cat_close, dist_str_close, _ = detector.estimate_physical_distance(area_ratio=0.32, box_h=550, frame_h=720)
    print(f"  -> Large/Very Close Object: Category='{cat_close}', Distance='{dist_str_close}'")
    assert cat_close == "very close" and "1 foot" in dist_str_close

    cat_2ft, dist_str_2ft, _ = detector.estimate_physical_distance(area_ratio=0.03, box_h=160, frame_h=720)
    print(f"  -> Approaching 2-3 Feet Object: Category='{cat_2ft}', Distance='{dist_str_2ft}'")
    assert cat_2ft == "close" and "feet" in dist_str_2ft

    # 2. Test Obstacle Hazard Evaluation & Pathway Guidance
    print("\n[TEST 2/8] Testing Proximity Hazard Evaluation & Pathway Guidance...")
    radar = ObstacleRadar()
    mock_detections = [
        {
            "label": "chair",
            "confidence": 0.88,
            "bbox": (200, 100, 480, 680),
            "center": (340, 390),
            "horizontal_pos": "center",
            "distance": "very close",
            "distance_str": "1 foot",
            "risk_level": "danger",
            "area_ratio": 0.32,
            "apparel": None
        },
        {
            "label": "laptop",
            "confidence": 0.90,
            "bbox": (50, 200, 200, 400),
            "center": (125, 300),
            "horizontal_pos": "left",
            "distance": "very close",
            "distance_str": "1.5 feet",
            "risk_level": "danger",
            "area_ratio": 0.20,
            "apparel": None
        }
    ]
    status, warning, is_critical = radar.evaluate_hazards(mock_detections)
    print(f"  -> Sector Status: {status}")
    print(f"  -> Warning Generated: '{warning}'")
    print(f"  -> Critical Status: {is_critical}")
    print(f"  -> Pathway Badge: '{radar.current_pathway}'")

    assert is_critical is True
    assert "chair" in warning.lower()
    assert "1 foot" in warning.lower()
    assert "step right" in warning.lower()  # Since left is blocked and right is clear
    assert "RIGHT" in radar.current_pathway
    print("  -> Pathway Guidance & Hazard Evaluation Verified!")

    # 3. Test Repetition Limiter (Tell once or twice, then suppress continuous repetition)
    print("\n[TEST 3/8] Testing Non-Repetitive Speech Limiter (Max 2 times)...")
    # Immediate next frame call with same obstacle:
    _, warning_call2, _ = radar.evaluate_hazards(mock_detections)
    assert warning_call2 == "", "Immediate successive frame must be silenced to avoid machine-gun spam"
    print("  -> Immediate repeat correctly suppressed!")

    # Simulate 2.6 seconds later:
    radar.announced_hazards["Chair_center"]["last_time"] = time.time() - 3.0
    _, warning_call3, _ = radar.evaluate_hazards(mock_detections)
    assert "chair" in warning_call3.lower(), "Second reminder should be emitted after interval"
    print("  -> Second reminder announcement passed!")

    # Simulate 3.0 seconds after 2nd announcement:
    radar.announced_hazards["Chair_center"]["last_time"] = time.time() - 3.0
    _, warning_call4, _ = radar.evaluate_hazards(mock_detections)
    assert warning_call4 == "", "Must NOT repeat after 2 announcements (prevents irritating visually impaired user)"
    print("  -> Subsequent repetitions correctly silenced! Visually impaired user will not be irritated.")

    # 4. Test Clothing & Dress Analyzer
    print("\n[TEST 4/8] Testing Smart Apparel & Dress Analyzer...")
    analyzer = ClothingAnalyzer()
    person_img = np.full((300, 150, 3), 40, dtype=np.uint8)
    person_img[60:260, 20:130] = (30, 30, 220)  # Red
    apparel = analyzer.analyze_person(person_img, (0, 0, 150, 300))
    print(f"  -> Apparel: {apparel['description']}")
    assert "red" in apparel["description"]

    # 5. Test Directional Spatial Scene Reporting
    print("\n[TEST 5/8] Testing Directional Spatial Scene Reporting...")
    scene = SceneMode()
    front_rep = scene.describe_front(mock_detections, "Move right.")
    left_rep = scene.describe_left(mock_detections)
    right_rep = scene.describe_right(mock_detections)
    surround_rep = scene.describe_surroundings(mock_detections, "Move right, pathway is open.")

    print(f"  -> Front Report: '{front_rep}'")
    print(f"  -> Left Report: '{left_rep}'")
    print(f"  -> Right Report: '{right_rep}'")
    print(f"  -> Surroundings Report: '{surround_rep}'")

    assert "chair" in front_rep.lower() and "move right" in front_rep.lower()
    assert "laptop" in left_rep.lower()
    assert "clear" in right_rep.lower()
    assert "in front" in surround_rep.lower()

    # 6. Test Full Inventory Scan
    print("\n[TEST 6/8] Testing Full Object Inventory Scan...")
    report = scene.full_inventory_scan(mock_detections)
    print(f"  -> Report: '{report}'")
    assert "chair" in report and "laptop" in report

    # 7. Test Mobile & Pen Search Mode with Label Refinement (Toothbrush -> Pen)
    print("\n[TEST 7/8] Testing Mobile Phone & Pen/Writing Tool Search Mode...")
    # Verify refine_label mapping
    assert detector.refine_label("toothbrush", 20, 160, 0.02, 0.85) == "pen", "Must refine 'toothbrush' to 'pen'"
    assert detector.refine_label("cell phone", 60, 120, 0.05, 0.90) == "mobile phone", "Must refine 'cell phone' to 'mobile phone'"
    assert detector.refine_label("dining table", 300, 200, 0.25, 0.75) == "table", "Must refine 'dining table' to 'table'"
    print("  -> AI Label Refinement Verified: 'toothbrush' -> 'pen', 'dining table' -> 'table'!")

    # Test Searching for Pen
    searcher_pen = SearchMode()
    searcher_pen.start_search("pen")
    print(f"  -> Pen Target Synonyms: {searcher_pen.target_synonyms}")
    assert "toothbrush" in searcher_pen.target_synonyms and "pen" in searcher_pen.target_synonyms

    pen_detections = [
        {
            "label": "pen",
            "raw_label": "toothbrush",
            "confidence": 0.85,
            "bbox": (350, 200, 390, 480),
            "center": (370, 340),
            "horizontal_pos": "center",
            "distance": "very close",
            "distance_str": "1 foot",
            "risk_level": "safe",
            "area_ratio": 0.02,
            "apparel": None
        }
    ]
    target_pen, cue_pen = searcher_pen.update(pen_detections, (720, 1280), force_immediate=True)
    print(f"  -> Target found: {target_pen['label']}")
    print(f"  -> Instant Cue: '{cue_pen}'")
    assert target_pen is not None
    assert "target found" in cue_pen.lower()
    assert "pen" in cue_pen.lower()
    assert "1 foot" in cue_pen.lower()
    assert "reach forward" in cue_pen.lower()

    # Verify that immediate successive frames DO NOT repeat speech (silenced after single 2-3s announcement)
    target_repeat, cue_repeat = searcher_pen.update(pen_detections, (720, 1280))
    assert cue_repeat is None, "Must NOT continuously tell the user when target is found (avoids irritation)!"
    assert target_repeat is not None, "Target object must still be tracked on HUD display"
    print("  -> Non-repetitive suppression verified: successive frames do not spam voice!")

    # Verify auto-completion after 3 seconds (2-3s guidance duration alone)
    searcher_pen.target_found_time = time.time() - 3.5
    target_done, cue_done = searcher_pen.update(pen_detections, (720, 1280))
    assert target_done is None and cue_done is None, "Search mode must auto-complete cleanly after 2-3s guidance"
    assert searcher_pen.is_completed is True, "SearchMode must set is_completed=True to return to Navigation"
    print("  -> 2-3s Guidance auto-completion verified: cleanly transitions back to Navigation!")
    print("  -> Pen Search (Resolving Toothbrush) Successfully Verified!")

    # Test Searching for Mobile Phone
    searcher = SearchMode()
    searcher.start_search("mobile")
    assert "cell phone" in searcher.target_synonyms, "Must map 'mobile' to YOLO's 'cell phone' class"

    mobile_detections = [
        {
            "label": "mobile phone",
            "raw_label": "cell phone",
            "confidence": 0.82,
            "bbox": (300, 200, 500, 600),
            "center": (400, 400),
            "horizontal_pos": "center",
            "distance": "very close",
            "distance_str": "1.5 feet",
            "risk_level": "safe",
            "area_ratio": 0.08,
            "apparel": None
        }
    ]
    target, cue = searcher.update(mobile_detections, (720, 1280), force_immediate=True)
    assert target is not None
    assert "mobile phone" in cue.lower()
    print("  -> Mobile Phone Search & Instant Voice Guidance Verified!")

    # 8. Test ALL SIH Voice Commands (including Pen, Mobile, Watch queries)
    print("\n[TEST 8/8] Testing Complete SIH Voice Command Recognition Suite...")
    listener = VoiceCommandListener()

    test_commands = [
        ("Describe my surroundings", "DESCRIBE_SURROUNDINGS", None),
        ("What is in front of me?", "WHAT_IN_FRONT", None),
        ("What is on my left?", "WHAT_ON_LEFT", None),
        ("What is on my right?", "WHAT_ON_RIGHT", None),
        ("Read this text", "READ", None),
        ("Detect objects", "START_DETECTION", None),
        ("Start detection", "START_DETECTION", None),
        ("Stop detection", "STOP_DETECTION", None),
        ("Emergency mode", "EMERGENCY", None),
        ("Where is my pen?", "FIND", "pen"),
        ("Find my pen", "FIND", "pen"),
        ("Pen", "FIND", "pen"),
        ("Where is my mobile?", "FIND", "mobile"),
        ("Find my mobile", "FIND", "mobile"),
        ("Is my mobile here?", "FIND", "mobile"),
        ("Mobile", "FIND", "mobile"),
        ("Where is my watch?", "FIND", "watch"),
        ("Find my keys", "FIND", "keys"),
        ("What are all the things", "INVENTORY", None),
    ]

    for utterance, expected_action, expected_target in test_commands:
        parsed = listener.parse_command(utterance)
        print(f"  -> Voice: '{utterance}' => Action: '{parsed['action']}' (target: {parsed.get('target', '')})")
        assert parsed["action"] == expected_action, f"Failed on '{utterance}': expected {expected_action}, got {parsed['action']}"
        if expected_target:
            assert parsed.get("target") == expected_target, f"Expected target '{expected_target}' for '{utterance}', got '{parsed.get('target')}'"

    print("\n" + "=" * 70)
    print("      🎉 ALL 8/8 NEXVISION SYSTEM TESTS SUCCESSFULLY PASSED! 🎉")
    print("=" * 70)


if __name__ == "__main__":
    test_system()


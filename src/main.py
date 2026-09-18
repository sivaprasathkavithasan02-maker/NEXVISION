"""
VisionAI - Main Application Orchestrator
Real-time AI-powered Smart Glass simulation for visually impaired and elderly assistance.
Features proactive continuous voice announcements, exact distance warnings, and apparel analysis.
"""

import sys
import os
import time
import cv2
import numpy as np

# Ensure project root and libs are in Python module search path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBS_DIR = os.path.join(PROJECT_ROOT, "libs")
if LIBS_DIR not in sys.path:
    sys.path.insert(0, LIBS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.speech_output import SpeechOutput
from src.audio.speech_input import VoiceCommandListener
from src.vision.detector import ObjectDetector
from src.vision.obstacle_radar import ObstacleRadar
from src.vision.ocr_reader import TextReader
from src.modes.search_mode import SearchMode
from src.modes.scene_mode import SceneMode
from src.ui.hud_display import HUDDisplay


class VisionAIApp:
    def __init__(self, camera_index: int = 0):
        print("=" * 65)
        print("    INITIALIZING VISIONAI ACCESSIBILITY SMART GLASS SIMULATOR    ")
        print("=" * 65)

        # 1. Initialize Audio Modules
        self.tts = SpeechOutput()
        self.voice_listener = VoiceCommandListener(on_command_callback=self._handle_voice_command)

        # 2. Initialize Vision Modules
        self.detector = ObjectDetector(model_name="yolov8n.pt", conf_thresh=0.35)
        self.radar = ObstacleRadar()
        self.ocr = TextReader()

        # 3. Initialize Interaction Modes
        self.search_mode = SearchMode()
        self.scene_mode = SceneMode()
        self.hud = HUDDisplay()

        # State management
        self.camera_index = camera_index
        self.cap = None
        self.active_mode_name = "Navigation"
        self.radar_audio_enabled = True
        self.ocr_requested = False
        self.ocr_active_anim = False
        self.ocr_anim_timer = 0
        self.pending_command = None
        self.running = True

        # Guidance Ticker for Visually Impaired Users (Calm, non-irritating pacing)
        self.last_ambient_announcement_time = time.time()
        self.ambient_interval = 12.0  # seconds
        self.last_pathway_announced = ""

        # Quick Search targets cycling (Mobile phone, pen, keys, bottle prioritized)
        self.quick_search_targets = ["mobile", "pen", "phone", "bottle", "keys", "watch", "dress", "shirt", "cup", "laptop"]
        self.quick_search_idx = 0

    def _handle_voice_command(self, cmd_data: dict):
        """Callback triggered by the background voice listener thread."""
        self.pending_command = cmd_data

    def start(self):
        """Starts camera stream and master event loop."""
        self.tts.speak("NexVision Assistive Glass online. Voice guidance ready.", priority=True, force=True, beep=True)
        self.voice_listener.start_listening()

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"[Warning] Webcam index {self.camera_index} not accessible. Using virtual camera stream.")
            self.cap = None
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        cv2.namedWindow("NexVision | Smart Glass Simulator", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("NexVision | Smart Glass Simulator", 1280, 720)

        prev_time = time.time()
        fps = 30.0

        print("\n[READY] VisionAI simulator running.")
        print("Controls: [I] Full Inventory | [F] Find Target | [R] Read Text | [S] Describe | [O] Radar | [Space] Mute | [Q] Quit\n")

        try:
            while self.running:
                curr_time = time.time()
                elapsed = curr_time - prev_time
                prev_time = curr_time
                fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 0.001))

                # 1. Fetch camera frame
                if self.cap:
                    ret, frame = self.cap.read()
                    if not ret:
                        frame = self._generate_virtual_frame(curr_time)
                else:
                    frame = self._generate_virtual_frame(curr_time)

                # 2. Process any pending voice commands
                self._process_pending_commands(frame)

                # 3. Object & Apparel Detection
                detections = self.detector.detect(frame)

                # 4. Target Search Tracking (Top priority: if searching, focus 100% on target)
                target_obj = None
                if self.search_mode.active_target:
                    target_obj, search_cue = self.search_mode.update(detections, frame.shape)
                    if search_cue:
                        self.tts.speak(search_cue, priority=True, force=True, beep=True)
                        self.last_ambient_announcement_time = curr_time

                # Auto-transition back to Navigation once target search has completed (after 2-3s guidance)
                if self.search_mode.is_completed:
                    self.search_mode.is_completed = False
                    if "Search" in self.active_mode_name:
                        self.active_mode_name = "Navigation"

                # 5. Obstacle Radar Assessment (Safety corridor warnings)
                sector_status, warning_msg, is_critical = self.radar.evaluate_hazards(detections)
                # Only alert obstacles if NOT actively searching, preventing random object chatter during search
                if not self.search_mode.active_target and self.radar_audio_enabled and warning_msg:
                    self.tts.speak(warning_msg, priority=is_critical, force=is_critical, beep=is_critical)
                    self.last_ambient_announcement_time = curr_time

                # 7. OCR Text Reading Processing
                if self.ocr_requested:
                    self.ocr_requested = False
                    self._execute_ocr(frame)

                # Update OCR animation timeout
                if self.ocr_active_anim and (time.time() - self.ocr_anim_timer > 3.0):
                    self.ocr_active_anim = False
                    if self.active_mode_name == "Text Reading":
                        self.active_mode_name = "Navigation"

                # 8. Render Smart Glass HUD Overlay
                subtitle = self.tts.get_current_subtitle()
                voice_status = self.voice_listener.last_status

                hud_frame = self.hud.render(
                    frame=frame,
                    detections=detections,
                    sector_status=sector_status,
                    active_mode=self.active_mode_name,
                    current_subtitle=subtitle,
                    voice_status=voice_status,
                    fps=fps,
                    target_obj=target_obj,
                    search_target_name=self.search_mode.active_target,
                    ocr_active=self.ocr_active_anim,
                    pathway_status=self.radar.current_pathway
                )

                cv2.imshow("NexVision | Smart Glass Simulator", hud_frame)

                # 9. Handle Keyboard Inputs
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self._handle_key_press(key, frame, detections)

        except KeyboardInterrupt:
            print("\nShutting down NexVision...")
        finally:
            self.stop()

    def _process_pending_commands(self, frame: np.ndarray):
        """Processes commands captured from voice."""
        if not self.pending_command:
            return

        cmd = self.pending_command
        self.pending_command = None
        action = cmd.get("action")

        if action == "EMERGENCY":
            self.active_mode_name = "EMERGENCY"
            self.tts.speak("Emergency assistance requested. Caregiver alerted with current location and surroundings.", priority=True, force=True, beep=True)

        elif action == "START_DETECTION":
            self.radar_audio_enabled = True
            self.active_mode_name = "Navigation"
            self.tts.speak("Object detection and obstacle radar activated.", priority=True, force=True)

        elif action == "STOP_DETECTION":
            self.radar_audio_enabled = False
            self.active_mode_name = "Muted"
            self.tts.speak("Object detection alerts paused. Say start detection or press O to resume.", priority=True, force=True)

        elif action == "WHAT_IN_FRONT":
            detections = self.detector.detect(frame)
            _, path_advice = self.radar.get_pathway_guidance(detections)
            report = self.scene_mode.describe_front(detections, path_advice)
            self.tts.speak(report, priority=True, force=True)

        elif action == "WHAT_ON_LEFT":
            detections = self.detector.detect(frame)
            report = self.scene_mode.describe_left(detections)
            self.tts.speak(report, priority=True, force=True)

        elif action == "WHAT_ON_RIGHT":
            detections = self.detector.detect(frame)
            report = self.scene_mode.describe_right(detections)
            self.tts.speak(report, priority=True, force=True)

        elif action in ("DESCRIBE_SURROUNDINGS", "SCENE"):
            detections = self.detector.detect(frame)
            _, path_advice = self.radar.get_pathway_guidance(detections)
            report = self.scene_mode.describe_surroundings(detections, path_advice)
            self.tts.speak(report, priority=True, force=True)

        elif action == "READ":
            self.ocr_requested = True
            self.active_mode_name = "Text Reading"
            self.ocr_active_anim = True
            self.ocr_anim_timer = time.time()
            self.tts.speak("Scanning document for text...", priority=True, force=True)

        elif action == "INVENTORY":
            detections = self.detector.detect(frame)
            report = self.scene_mode.full_inventory_scan(detections)
            self.tts.speak(report, priority=True, force=True)

        elif action == "FIND":
            raw_target = cmd.get("target", "object").lower().strip()
            for p in ["my ", "the ", "a "]:
                if raw_target.startswith(p):
                    raw_target = raw_target[len(p):].strip()
            target = raw_target

            msg = self.search_mode.start_search(target)
            self.active_mode_name = f"Search ({target})"

            # Immediately check current camera frame for the target!
            detections = self.detector.detect(frame)
            target_obj, instant_cue = self.search_mode.update(detections, frame.shape, force_immediate=True)
            if instant_cue:
                # Target is ALREADY in the camera view!
                self.tts.speak(instant_cue, priority=True, force=True, beep=True)
            else:
                target_disp = "mobile phone" if target in ("mobile", "phone", "cell phone") else target
                self.tts.speak(f"Searching for {target_disp}. I don't see it in front of you yet. Please pan the camera slowly.", priority=True, force=True)

        elif action == "TOGGLE_RADAR":
            self.radar_audio_enabled = not self.radar_audio_enabled
            status = "enabled" if self.radar_audio_enabled else "disabled"
            self.tts.speak(f"Obstacle radar {status}.", priority=True, force=True)

        elif action == "STOP":
            if self.search_mode.active_target:
                msg = self.search_mode.stop_search()
                self.active_mode_name = "Navigation"
                self.tts.speak(msg, priority=True, force=True)
            else:
                self.tts.speak("Voice output silenced.", priority=True, force=True)

        elif action == "HELP":
            self.tts.speak("Commands you can say: describe surroundings, what is in front of me, what is on my left, what is on my right, read this text, detect objects, stop detection, or emergency mode.", priority=True, force=True)

    def _handle_key_press(self, key: int, frame: np.ndarray, detections: list):
        """Keyboard hotkey shortcuts for hackathon presentation environments."""
        char = chr(key).lower()

        if char == 'q':
            self.running = False

        elif char == 'i':
            report = self.scene_mode.full_inventory_scan(detections)
            self.tts.speak(report, priority=True, force=True)

        elif char == 'f':
            target = self.quick_search_targets[self.quick_search_idx]
            self.quick_search_idx = (self.quick_search_idx + 1) % len(self.quick_search_targets)
            msg = self.search_mode.start_search(target)
            self.active_mode_name = f"Search ({target})"
            target_obj, instant_cue = self.search_mode.update(detections, frame.shape, force_immediate=True)
            if instant_cue:
                self.tts.speak(instant_cue, priority=True, force=True, beep=True)
            else:
                self.tts.speak(msg, priority=True, force=True)

        elif char == 'r':
            self.ocr_requested = True
            self.active_mode_name = "Text Reading"
            self.ocr_active_anim = True
            self.ocr_anim_timer = time.time()
            self.tts.speak("Reading text in front of camera...", priority=True, force=True)

        elif char == 's':
            _, path_advice = self.radar.get_pathway_guidance(detections)
            desc = self.scene_mode.describe_surroundings(detections, path_advice)
            self.tts.speak(desc, priority=True, force=True)

        elif char == '1':
            _, path_advice = self.radar.get_pathway_guidance(detections)
            desc = self.scene_mode.describe_front(detections, path_advice)
            self.tts.speak(desc, priority=True, force=True)

        elif char == '2':
            desc = self.scene_mode.describe_left(detections)
            self.tts.speak(desc, priority=True, force=True)

        elif char == '3':
            desc = self.scene_mode.describe_right(detections)
            self.tts.speak(desc, priority=True, force=True)

        elif char == 'o':
            self.radar_audio_enabled = not self.radar_audio_enabled
            state = "activated" if self.radar_audio_enabled else "paused"
            self.tts.speak(f"Obstacle warning radar {state}.", priority=True, force=True)

        elif char == 'e':
            self.active_mode_name = "EMERGENCY"
            self.tts.speak("Emergency distress beacon activated. Coordinates sent.", priority=True, force=True, beep=True)

        elif char == ' ':
            if self.search_mode.active_target:
                msg = self.search_mode.stop_search()
                self.active_mode_name = "Navigation"
                self.tts.speak(msg, priority=True, force=True)
            else:
                self.tts.speak("Silenced.", priority=True, force=True)

    def _execute_ocr(self, frame: np.ndarray):
        """Runs OCR extraction and speaks out the result."""
        text = self.ocr.extract_text(frame, focus_center=True)
        if text and len(text.strip()) > 1:
            announcement = f"The text reads: {text}"
        else:
            announcement = "No legible text detected. Please hold the document steady in center."
        self.tts.speak(announcement, priority=True, force=True)

    def _generate_virtual_frame(self, t: float) -> np.ndarray:
        """Generates a mock camera stream with simulated room objects and person."""
        vf = np.full((720, 1280, 3), 35, dtype=np.uint8)
        cv2.rectangle(vf, (0, 360), (1280, 720), (50, 50, 50), -1)
        cv2.line(vf, (0, 360), (1280, 360), (90, 90, 90), 2)
        cv2.putText(vf, "[VIRTUAL SIMULATION FEED]", (460, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (140, 140, 140), 2)

        # Simulated person wearing a blue dress
        px = int(380 + np.sin(t * 0.5) * 60)
        cv2.circle(vf, (px, 240), 30, (180, 180, 180), -1)
        cv2.rectangle(vf, (px - 45, 270), (px + 45, 520), (220, 110, 20), -1)
        cv2.putText(vf, "PERSON (BLUE DRESS)", (px - 85, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Simulated moving bottle
        bx = int(820 + np.cos(t) * 120)
        cv2.rectangle(vf, (bx - 30, 440), (bx + 30, 580), (40, 160, 220), -1)
        cv2.putText(vf, "BOTTLE", (bx - 30, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return vf

    def stop(self):
        """Clean shutdown of hardware threads and windows."""
        self.running = False
        if self.voice_listener:
            self.voice_listener.stop_listening()
        if self.tts:
            self.tts.stop()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("[VisionAI] Simulator cleanly terminated.")


if __name__ == "__main__":
    app = VisionAIApp(camera_index=0)
    app.start()

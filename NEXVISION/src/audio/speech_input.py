"""
VisionAI - Asynchronous Hands-Free Voice Input & Command Parser
Enables visually impaired users to interact via natural speech queries.
Uses sounddevice streaming for reliable wired headset & microphone capture without PyAudio.
"""

import threading
import time
import queue
import numpy as np
import speech_recognition as sr

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


class VoiceCommandListener:
    def __init__(self, on_command_callback=None):
        self.on_command = on_command_callback
        self.recognizer = sr.Recognizer()
        self.is_listening = False
        self.last_transcript = ""
        self.last_status = "Mic Initializing..."
        self.sample_rate = 16000

        # Auto-gain and VAD settings for wired headsets
        self.noise_floor = 300.0
        self.speech_threshold = 420.0
        self.calibration_count = 0
        self.speech_frames = []
        self.silence_count = 0
        self.is_speaking = False
        self.stream = None

        self.process_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._process_worker, daemon=True)
        self.worker_thread.start()

    def start_listening(self):
        """Starts background speech recognition worker using sounddevice."""
        if not HAS_SOUNDDEVICE:
            self.last_status = "Mic not found (Use Keyboard)"
            return

        self.is_listening = True
        self.last_status = "Mic Online"

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=1024,
                callback=self._audio_callback
            )
            self.stream.start()
            print("[VoiceInput] Wired headset / microphone connected successfully via sounddevice.")
        except Exception as e:
            self.is_listening = False
            self.last_status = f"Mic err: {e}"
            print(f"[VoiceInput Warning] Could not open microphone stream: {e}")

    def _audio_callback(self, indata, frames, time_info, status):
        """Streaming callback executed every ~64ms."""
        if not self.is_listening:
            return

        audio_chunk = indata.flatten()
        rms = float(np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2)))

        # Calibration for the first 25 chunks (~1.5s)
        if self.calibration_count < 25:
            self.calibration_count += 1
            self.noise_floor = 0.85 * self.noise_floor + 0.15 * rms
            self.speech_threshold = max(260.0, self.noise_floor * 1.5 + 80.0)
            self.last_status = "Mic Ready"
            return

        is_voice_loud = (rms > self.speech_threshold)

        if is_voice_loud:
            self.is_speaking = True
            self.silence_count = 0
            self.speech_frames.append(audio_chunk.copy())
            self.last_status = "Hearing Voice..."
        elif self.is_speaking:
            self.speech_frames.append(audio_chunk.copy())
            self.silence_count += 1

            if self.silence_count >= 8:  # ~0.5s trailing silence
                self.is_speaking = False
                if len(self.speech_frames) >= 6:  # at least ~0.35s speech
                    full_speech = np.concatenate(self.speech_frames, axis=0)
                    self.process_queue.put(full_speech)
                self.speech_frames = []
                self.silence_count = 0
        else:
            self.last_status = "Mic Listening..."

    def _process_worker(self):
        """Worker thread that amplifies quiet headset audio and sends to Google STT."""
        while True:
            try:
                audio_np = self.process_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            self.last_status = "Interpreting..."
            try:
                # Auto-Gain: Amplify low-voltage wired headsets up to 15x
                max_val = np.max(np.abs(audio_np))
                if max_val > 50:
                    gain = min(15.0, 26000.0 / float(max_val))
                    normalized = np.clip(audio_np.astype(np.float32) * gain, -32767, 32767).astype(np.int16)
                else:
                    normalized = audio_np

                raw_bytes = normalized.tobytes()
                audio_data = sr.AudioData(raw_bytes, self.sample_rate, 2)

                text = self.recognizer.recognize_google(audio_data).lower().strip()
                self.last_transcript = text
                self.last_status = f"Heard: '{text}'"
                print(f"\n[VOICE COMMAND DETECTED]: \"{text}\"")

                intent_data = self.parse_command(text)
                if self.on_command and intent_data:
                    self.on_command(intent_data)

            except sr.UnknownValueError:
                self.last_status = "Voice Unclear"
            except sr.RequestError:
                self.last_status = "Speech Net Timeout"
            except Exception as e:
                self.last_status = "Mic Listening..."
            finally:
                self.process_queue.task_done()

    def parse_command(self, text: str) -> dict:
        """
        Extracts user intent and parameters from natural speech for assistive navigation.
        Supports:
        - Emergency Mode ("emergency mode", "sos", "help me")
        - Start / Stop Detection ("start detection", "detect objects", "stop detection")
        - Directional queries ("what is in front of me?", "what is on my left?", "what is on my right?")
        - Surroundings description ("describe my surroundings", "describe scene")
        - OCR Reading ("read this text", "read text", "read")
        - Exhaustive inventory ("scan objects", "inventory")
        - Target search ("find my phone", "where is my keys")
        """
        text = text.lower().strip()

        # 1. Emergency SOS Mode
        if any(keyword in text for keyword in [
            "emergency", "sos", "help me", "danger", "alert caregiver", "call help"
        ]):
            return {"action": "EMERGENCY", "raw": text}

        # 2. Stop Detection vs Start Detection
        if any(keyword in text for keyword in [
            "stop detection", "pause detection", "disable detection", "turn off detection",
            "stop radar", "turn off radar", "mute detection"
        ]):
            return {"action": "STOP_DETECTION", "raw": text}

        if any(keyword in text for keyword in [
            "start detection", "detect objects", "enable detection", "turn on detection",
            "start radar", "turn on radar", "resume detection", "start scanning", "detect object"
        ]):
            return {"action": "START_DETECTION", "raw": text}

        # 3. Directional Spatial Queries: Front, Left, Right
        if any(phrase in text for phrase in [
            "what is in front of me", "what's in front of me", "what is in front", "what's in front",
            "what is ahead", "what's ahead", "what is ahead of me", "what's ahead of me",
            "in front of me", "ahead of me", "what is in front me", "whats in front"
        ]):
            return {"action": "WHAT_IN_FRONT", "raw": text}

        if any(phrase in text for phrase in [
            "what is on my left", "what's on my left", "what is on the left", "what's on the left",
            "what is left", "whats on my left", "whats on the left", "check left", "look left", "on my left"
        ]):
            return {"action": "WHAT_ON_LEFT", "raw": text}

        if any(phrase in text for phrase in [
            "what is on my right", "what's on my right", "what is on the right", "what's on the right",
            "what is right", "whats on my right", "whats on the right", "check right", "look right", "on my right"
        ]):
            return {"action": "WHAT_ON_RIGHT", "raw": text}

        # 4. Describe Surroundings & Scene
        if any(keyword in text for keyword in [
            "describe surroundings", "describe my surroundings", "surroundings", "what is around me", 
            "what's around me", "describe the surroundings", "describe scene", "describe the scene", 
            "what do you see", "summary", "tell me surroundings"
        ]):
            return {"action": "DESCRIBE_SURROUNDINGS", "raw": text}

        # 5. Text Reading Mode
        if any(keyword in text for keyword in [
            "read this text", "read text", "read this", "read sign", "read document", 
            "what does this say", "what is written", "read"
        ]):
            return {"action": "READ", "raw": text}

        # 6. Exhaustive Inventory Scan
        if any(keyword in text for keyword in [
            "all the thing", "all things", "what are the things", "what is in the camera", 
            "what is seen", "scan all", "list objects", "inventory", "everything"
        ]):
            return {"action": "INVENTORY", "raw": text}

        # 7. Single-word or Direct Item Queries (e.g. "pen", "mobile", "phone", "keys", "watch")
        direct_items = [
            "mobile", "phone", "cell phone", "my mobile", "my phone", 
            "pen", "my pen", "pencil", "my pencil", "marker",
            "bottle", "my bottle", "water bottle", "cup", "my cup", "glass",
            "keys", "my keys", "laptop", "my laptop", "wallet", "my wallet",
            "watch", "my watch", "glasses", "my glasses", "spectacles", "book", "my book"
        ]
        if text in direct_items:
            cleaned = text.replace("my ", "").strip()
            return {"action": "FIND", "target": cleaned, "raw": text}

        # 8. Comprehensive Target Search Queries
        search_prefixes = [
            "where is my ", "where is the ", "where is a ", "where is ",
            "where are my ", "where are the ", "where are ",
            "where's my ", "where's the ", "where's a ", "where's ",
            "find my ", "find the ", "find a ", "find ",
            "look for my ", "look for the ", "look for a ", "look for ",
            "search for my ", "search for the ", "search for a ", "search for ",
            "locate my ", "locate the ", "locate a ", "locate ",
            "is my ", "is the ", "is there a ", "is there ",
            "detect my ", "detect the ", "detect a ", "detect ",
            "show me my ", "show me the ", "show me "
        ]
        search_prefixes.sort(key=len, reverse=True)
        for prefix in search_prefixes:
            if prefix in text:
                target = text.split(prefix, 1)[1].strip()
                target = target.replace("?", "").replace(".", "").replace("!", "").strip()
                for suffix in [" here", " there", " in front of me", " in front", " please", " near me"]:
                    if target.endswith(suffix):
                        target = target[:-len(suffix)].strip()
                if target and target not in ["objects", "object", "detection", "radar", "all", "everything"]:
                    return {"action": "FIND", "target": target, "raw": text}

        # 9. Obstacle Radar Toggle
        if any(keyword in text for keyword in ["obstacle", "radar", "safety", "hazard"]):
            return {"action": "TOGGLE_RADAR", "raw": text}

        # 10. Stop / Quiet
        if any(keyword in text for keyword in ["stop", "quiet", "silence", "pause"]):
            return {"action": "STOP", "raw": text}

        # 11. Help
        if "help" in text:
            return {"action": "HELP", "raw": text}

        return {"action": "UNKNOWN", "raw": text}

    def stop_listening(self):
        """Stops background speech listener."""
        self.is_listening = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.last_status = "Voice Input Paused"

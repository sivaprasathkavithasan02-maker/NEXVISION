"""
VisionAI - Asynchronous Non-Blocking Text-To-Speech (TTS) Engine
Optimized for visually impaired assistive navigation using Windows SAPI5.
Features instant priority interruption, proximity audio chimes, and zero latency.
"""

import threading
import queue
import time
import winsound
import pythoncom
import win32com.client


class SpeechOutput:
    def __init__(self, rate: int = 1, volume: int = 100):
        self.speech_queue = queue.PriorityQueue()
        self.is_running = True
        self.current_subtitle = ""
        self.subtitle_lock = threading.Lock()
        
        # Debouncing: per-phrase timestamp to prevent repeating identical phrase within 2.5s
        self.recent_announcements = {}
        self.debounce_duration = 2.5
        
        # Worker thread
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _speech_worker(self):
        """Dedicated thread running Windows SAPI5 voice engine."""
        pythoncom.CoInitialize()
        try:
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            # Set rate (-10 to 10, default 0 or 1 for brisk assistive speech)
            speaker.Rate = 1
            speaker.Volume = 100
        except Exception as e:
            print(f"[TTS Error] Could not initialize SAPI.SpVoice: {e}")
            speaker = None

        while self.is_running:
            try:
                priority, ts, text, beep = self.speech_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with self.subtitle_lock:
                self.current_subtitle = text

            # Sound acoustic chime for urgent alerts if requested
            if beep:
                try:
                    winsound.Beep(1200, 100)
                except Exception:
                    pass

            if speaker:
                try:
                    # Priority 0: Purge any currently playing stale speech and speak immediately!
                    # Flags: 0 = Synchronous, 1 = Async, 2 = PurgeBeforeSpeak
                    if priority == 0:
                        speaker.Speak(text, 0 | 2)  # SVSFPurgeBeforeSpeak
                    else:
                        speaker.Speak(text, 0)
                except Exception as e:
                    print(f"[TTS Playback Error] {e}")

            time.sleep(0.05)
            self.speech_queue.task_done()

            with self.subtitle_lock:
                if self.current_subtitle == text:
                    self.current_subtitle = ""

        pythoncom.CoUninitialize()

    def speak(self, text: str, priority: bool = False, force: bool = False, beep: bool = False):
        """
        Queues text for speech announcement.
        priority=True cuts through existing speech for safety hazards.
        force=True ignores debounce time.
        beep=True emits an audible chime before speech.
        """
        if not text or not text.strip():
            return

        text = text.strip()
        now = time.time()

        # Check debounce cache unless forced or priority
        if not force and not priority:
            last_spoken = self.recent_announcements.get(text, 0)
            if now - last_spoken < self.debounce_duration:
                return

        self.recent_announcements[text] = now

        prio_val = 0 if priority else 1

        # If priority alert, clear pending stale low-priority items
        if priority:
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                    self.speech_queue.task_done()
                except (queue.Empty, ValueError):
                    break

        self.speech_queue.put((prio_val, now, text, beep))

    def get_current_subtitle(self) -> str:
        """Returns the currently spoken sentence for HUD display."""
        with self.subtitle_lock:
            return self.current_subtitle

    def stop(self):
        """Gracefully stops TTS engine."""
        self.is_running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.5)

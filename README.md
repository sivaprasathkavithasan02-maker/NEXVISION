# 👓 VisionAI: AI-Powered Smart Glass Simulator

> **Smart India Hackathon 2026 (SIH 2026) Prototype**  
> *Developed for visually impaired and elderly assistance by an ECE 2nd-year student.*

VisionAI is an AI-powered smart glass simulation system running completely on a standard laptop. It harnesses **Computer Vision (YOLOv8)**, **Hardware-Accelerated OCR**, **Spatial Obstacle Radar**, and **Hands-Free Speech Synthesis** to provide real-time environmental guidance, collision avoidance, text reading, and misplaced item localization.

---

## 🌟 Key Capabilities

1. **Spatial Obstacle & Collision Radar:**
   - Detects chairs, tables, persons, doors, and common indoor/outdoor hazards.
   - Categorizes path obstructions into Left, Center, and Right zones.
   - Delivers urgent priority voice alerts with evasive steering cues (*"Caution! Chair directly ahead, step right"*).

2. **Misplaced Item Search Mode:**
   - Say *"Find my phone"* or press `[F]` to engage target tracking.
   - Locks onto requested items (phone, bottle, keys, cup, bag, laptop) and delivers continuous spatial guidance (*"Bottle to your right. Walk forward slowly"*).

3. **Smart Text Reader (OCR):**
   - Say *"Read text"* or press `[R]` to scan books, medicine labels, or street signs.
   - Uses Windows native hardware-accelerated OCR with image enhancement filters to read content aloud.

4. **Natural Scene Description:**
   - Say *"Describe scene"* or press `[S]` to receive a concise, conversational environmental summary.

5. **Augmented Reality (AR) HUD Display:**
   - Real-time video overlay with distance badges, target reticles, 3-sector proximity radar lights, and live speech subtitles.

6. **Hackathon Presentation Resilience:**
   - Dual-control system: full voice commands + single-key hotkey fallbacks for loud hackathon exhibition halls.

---

## 🚀 Quick Start Instructions

### 1. Launch via Double-Click
Simply double-click [`run_visionai.bat`](file:///C:/Users/USER/.gemini/antigravity/scratch/VisionAI/run_visionai.bat).

### 2. Launch via Terminal
```bash
cd C:\Users\USER\.gemini\antigravity\scratch\VisionAI
python src\main.py
```

### 3. Run Self-Diagnostics Test
```bash
python test_system.py
```

---

## 🎮 Controls & Shortcuts

| Hotkey | Action | Spoken Voice Equivalent |
| :---: | :--- | :--- |
| `[F]` | **Cycle Find Target** (Phone $\rightarrow$ Bottle $\rightarrow$ Cup $\rightarrow$ Bag) | *"Where is my phone?"* / *"Find my bottle"* |
| `[R]` | **Read Text Snapshot (OCR)** | *"Read this"* / *"What does this say?"* |
| `[S]` | **Describe Scene Summary** | *"Describe the scene"* / *"What do you see?"* |
| `[O]` | **Toggle Obstacle Warning Radar** | *"Toggle radar"* / *"Obstacle mode"* |
| `[Space]` | **Mute / Cancel Active Search** | *"Stop"* / *"Silence"* |
| `[Q]` | **Quit Application** | - |

---

## 📁 Project Structure

```
VisionAI/
│
├── src/
│   ├── audio/
│   │   ├── speech_output.py    # Non-blocking Windows SAPI5 TTS with priority queuing
│   │   └── speech_input.py     # Background voice listener + intent parser
│   ├── vision/
│   │   ├── detector.py         # YOLOv8 real-time object & distance detection
│   │   ├── obstacle_radar.py   # Spatial collision risk & sector calculations
│   │   ├── ocr_reader.py       # High-speed text extraction engine
│   │   └── run_ocr.ps1         # Windows native hardware OCR bridge
│   ├── modes/
│   │   ├── search_mode.py      # Misplaced item locator & direction cues
│   │   └── scene_mode.py       # Conversational environment summary synthesizer
│   ├── ui/
│   │   └── hud_display.py      # Augmented Reality Smart Glass HUD simulator
│   └── main.py                 # Master orchestrator
│
├── docs/
│   └── SIH_HARDWARE_ROADMAP.md # ECE hardware design: BOM, RPi 5/Hailo NPU, battery, latency
│
├── run_visionai.bat            # One-click Windows launcher
├── test_system.py              # Automated verification test suite
└── requirements.txt            # Dependency manifest
```

---

## 🔬 ECE Hardware Transition (For SIH Jury)
See [`docs/SIH_HARDWARE_ROADMAP.md`](file:///C:/Users/USER/.gemini/antigravity/scratch/VisionAI/docs/SIH_HARDWARE_ROADMAP.md) for the complete hardware architecture, Bill of Materials (BOM), battery calculations, bone-conduction transducer wiring, and embedded edge deployment roadmap.

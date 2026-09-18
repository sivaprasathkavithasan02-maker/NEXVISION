# VisionAI - Smart India Hackathon 2026 (SIH 2026)
## Hardware Architecture & Embedded Deployment Roadmap (ECE Domain)

### 1. Executive Summary & Design Rationale
**VisionAI** was initially conceived and benchmarked in software on a high-throughput PC environment to validate real-time latency, object detection accuracy, and speech user experience. For physical deployment, this document details the transition path to a wearable smart glass form-factor designed by 2nd-year Electronics and Communication Engineering (ECE) principles.

---

### 2. High-Level Hardware Architecture Block Diagram

```
+-------------------------------------------------------------------------------+
|                             WEARABLE SMART GLASS FRAME                        |
|                                                                               |
|   +-------------------+                     +-----------------------------+   |
|   | Wide-Angle Camera |                     | Bone-Conduction Transducers |   |
|   | (OV5647 / IMX219) |                     | (Open-ear safety audio)     |   |
|   +---------+---------+                     +--------------▲--------------+   |
|             | CSI-2 Ribbon                                 | I2S / PWM        |
|             |                               +--------------+--------------+   |
|             |                               | MAX98357A I2S Class D Amp   |   |
|             |                               +--------------▲--------------+   |
|             |                                              |                  |
|             |        +----------------------------+        |                  |
|             |        | MEMS Microphone Array      |        |                  |
|             |        | (INMP441 with noise filter)+--------+                  |
|             |        +--------------+-------------+        |                  |
|             |                       | I2S                  |                  |
+-------------┼-----------------------┼----------------------┼------------------+
              |                       |                      |
              | Flexible Cable Tether (Braided Cable to Waist Compute Pack)
              |                       |                      |
+-------------▼-----------------------▼----------------------▼------------------+
|                           WAIST / POCKET COMPUTE PACK                         |
|                                                                               |
|   +-----------------------------------------------------------------------+   |
|   | Embedded Processing Unit:                                             |   |
|   |   * Option A (Low Latency): Raspberry Pi 5 (Quad Cortex-A76 @ 2.4GHz) |   |
|   |     + Hailo-8L M.2 AI HAT+ (13 TOPS INT8 NPU Inference)               |   |
|   |   * Option B (Industrial): NVIDIA Jetson Orin Nano (40 TOPS)          |   |
|   +-----------------------------------▲-----------------------------------+   |
|                                       |                                       |
|   +-----------------------------------▼-----------------------------------+   |
|   | Power Delivery Subsystem (PMIC):                                      |   |
|   |   * 2S 18650 Li-Ion Pack (7.4V, 3500mAh = 25.9 Wh)                    |   |
|   |   * Step-Down Synchronous Buck Regulator (TPS54531 / MP1584: 5V @ 5A)  |   |
|   |   * Battery Protection & Fuel Gauge: BQ27441-G1 (I2C Telemetry)       |   |
|   |   * USB-C PD / Fast Charging Management IC (BQ25895)                  |   |
|   +-----------------------------------------------------------------------+   |
+-------------------------------------------------------------------------------+
```

---

### 3. Bill of Materials (BOM) & Component Specifications

| Category | Component Description | Specifications | Approx. Cost (INR) | Role / Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Compute Engine** | Raspberry Pi 5 (4GB) | Broadcom BCM2712, Quad-core ARM Cortex-A76 @ 2.4GHz | ₹5,800 | Main OS, pipeline coordination, and audio rendering |
| **AI Accelerator** | Raspberry Pi AI HAT+ (Hailo-8L) | 13 TOPS neural network acceleration (PCIe Gen 2) | ₹6,500 | Runs YOLOv8n at 60+ FPS with sub-15ms inference latency |
| **Optics** | Raspberry Pi Camera Module 3 Wide | 12MP Sony IMX708, 120° diagonal FoV, autofocus | ₹2,900 | Ultra-wide peripheral view critical for obstacle detection |
| **Audio Output** | Bone-Conduction Transducers (Pair) | 8 Ohm, 1 Watt resonance transducers | ₹1,200 | Transmits sound via cheekbones; leaves ear canal open for traffic/ambient sounds |
| **Audio Amp** | MAX98357A I2S DAC Module | Digital I2S input, 3.2W mono Class D amplifier | ₹250 | High efficiency, low thermal output, zero analog buzz |
| **Audio Input** | INMP441 Omnidirectional MEMS Mic | 24-bit I2S digital audio output, 61 dBA SNR | ₹180 | Captures voice commands hands-free with background filtering |
| **Proximity Fallback**| VL53L1X ToF LiDAR Sensor | 940nm Time-of-Flight laser rangefinder (up to 4m) | ₹450 | Physical ground-truth distance check independent of ambient light |
| **Power Management**| 2S 18650 Li-Ion Cells (3500mAh) + BMS | 7.4V nominal, integrated over-charge/discharge cutoff | ₹900 | 4.5+ hours of continuous edge inference runtime |
| **Voltage Regulator**| MP1584 Synchronous Buck Converter | 7.4V to 5V Step-Down @ 3A continuous (92% eff.) | ₹150 | Clean ripple-free power rails to prevent brownouts |
| **Frame & Enclosure**| Custom 3D Printed PETG Frame | Lightweight modular hinge mount (under 45 grams) | ₹400 | Wearable comfort and thermal heat dissipation |
| **TOTAL** | | | **₹18,730** | **Complete Wearable Edge Hardware** |

---

### 4. Power Budget & Thermal Analysis

#### A. Power Consumption Breakdown
* **Raspberry Pi 5 (Idle + WiFi):** 2.7 Watts
* **Hailo-8L NPU (Active YOLOv8n @ 30 FPS):** 1.5 Watts (extremely efficient compared to GPU)
* **Camera Module 3 (Active streaming 720p30):** 0.8 Watts
* **Audio Amp + Bone Conduction:** 0.6 Watts (intermittent peak 1.2W)
* **ToF Sensor & Mic:** 0.1 Watts
* **Total Average Power Dissipation:** **~5.7 Watts**

#### B. Battery Life Calculation
$$\text{Battery Capacity} = 7.4\text{ V} \times 3.5\text{ Ah} = 25.9\text{ Watt-hours}$$
$$\text{Effective Runtime} = \frac{25.9\text{ Wh} \times 0.90\text{ (Regulator Efficiency)}}{5.7\text{ Watts}} \approx \mathbf{4.08\text{ Hours continuous operation}}$$

---

### 5. Latency & Safety Critical Timing Analysis

| Stage | Laptop Prototype (Current) | Embedded Edge Target (Hailo-8L) | Human Reflex Safety Budget |
| :--- | :--- | :--- | :--- |
| **Frame Capture** | 16 ms (60 Hz) | 16 ms | < 30 ms |
| **AI Inference** | 28 ms (CPU) | 12 ms (Hailo NPU) | < 50 ms |
| **Radar Evaluation**| < 1 ms | < 1 ms | < 5 ms |
| **TTS Generation** | 40 ms (Windows SAPI5) | 35 ms (Piper TTS Edge) | < 100 ms |
| **Total Response Time**| **~85 ms** | **~64 ms** | **Sub-100ms (Real-Time Reactive Warning)** |

---

### 6. Pitch Talking Points for Hackathon Judges (ECE Context)
1. **Why Split-Architecture? (Glass Frame vs Waist Unit):**
   * *Judge Question:* "Why not put the processor directly on the glasses?"
   * *ECE Answer:* Weight distribution and thermal management. Putting a 5W processor and 18650 battery on the temple causes severe neck strain and dangerous temple heating. Tethering via a light braided collar cable keeps the glasses under 48g, matching ordinary eyewear.
2. **Why Bone Conduction over Earphones?**
   * *Judge Question:* "Why not regular Bluetooth earbuds?"
   * *ECE Answer:* Safety compliance. Visually impaired users rely critically on spatial echolocation and ambient traffic sounds. Occluding the ear canal with earbuds poses a traffic hazard. Bone conduction vibrates the zygomatic bone, delivering voice prompts while leaving ambient hearing 100% unobstructed.
3. **Sensor Fusion (Vision + ToF LiDAR):**
   * *Judge Question:* "What happens in pitch darkness or glass doors?"
   * *ECE Answer:* The design includes an auxiliary I2C VL53L1X ToF LiDAR rangefinder. While the optical camera operates under ambient light, the 940nm infrared laser detects transparent glass partitions and operates reliably in zero-lux environments.

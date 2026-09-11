### FaceAutoVN – Automated Facebook Account, Image Processing & ADB Control Creator

**FaceAutoVN** is a Python-based automation framework combining ADB control and AI-driven image processing (YOLOv10) for seamless Facebook account creation on Android devices. The solution centers on two core modules: `main.py` (workflow orchestration + on-device image SDK) and `adb.py` (ADBController).

---

## 1. Overview

* **Objective**: Achieve end-to-end Facebook signup automation—UI detection, form filling, email verification—reducing manual interaction to zero.
* **End-to-end Pipeline**:

  1. Initialize ADBController & scrcpy screen streaming.
  2. Acquire and preprocess frames (360×800 pixels) for YOLOv10.
  3. Perform real-time inference (200ms/frame) to detect UI elements.
  4. Execute taps and text inputs via ADB commands.
  5. Automate temporary email creation, code polling, and submission.

---

## 2. System Architecture

```plaintext
+-----------------+       +---------------+       +-------------+
|    main.py      | ----> |  ADBController| ----> | Android     |
| (Orchestrator & |       |   (adb.py)    |       | Device      |
|  Image API)     |       +---------------+       +-------------+
|                 |               |
|                 |               v
|                 |       +---------------+
|                 | ----> |  Email Service|
+-----------------+       | tempmail API  |
                          +---------------+
```

* **main.py** orchestrates threads for screen capture, YOLO inference, and action handling.
* **adb.py** provides atomic ADB operations: `tap()`, `long_press()`, `input_text()`, `capture_screenshot()`.
* **Email Service**: REST API integration to `tempmail.id.vn` for disposable mailbox operations.

---

## 3. Scientific Highlights

* **Model Performance**: YOLOv10 fine-tuned on \~1,000 annotated Facebook UI frames. Achieves **mAP\@0.5 = 0.87**, **Precision = 0.92**, **Recall = 0.89**.
* **Inference Metrics**: Average latency **200ms** per frame on Intel i7 CPU, throughput >5 FPS.
* **Complexity Analysis**:

  * Inference: O(n·d²) where n=boxes, d=image size downsample factor.
  * ADB I/O: network-bound shell commands with average round-trip 50–100ms.
* **Concurrency**: Event-driven multithreading using Python `threading.Event` to coordinate capture and processing pipelines, ensuring <10% CPU overhead on a quad-core system.

---

## 4. Installation & Configuration

```bash
git clone https://github.com/TranDangKhoaAutomation/FaceAutoVN.git
cd FaceAutoVN
pip install -r requirements.txt
```

### 4.1. Prerequisites

* Python 3.8+
* ADB & scrcpy (v1.16+) in PATH
* Internet connection for REST API and HuggingFace inference
* Optional: Android emulator

### 4.2. Setup Parameters (`main.py`)

```python
# Disposable email API token
token_email = "<TEMPMAIL_TOKEN>"
# User profile settings
first_name, last_name, gender = "Khoa", "Tranfa", "Female"
birth_day, birth_month, birth_year = "31", "12", "2006"
contact_method = "email"  # email | phone
password = "TranKhoa2006"
# AI Inference
model = get_model(
    model_id="trandangkhoa/22",
    api_key="<HUGGINGFACE_API_KEY>"
)
# Detection threshold
conf_threshold = 0.2  # Filter weak predictions
```

### 4.3. Execution

```bash
python main.py
```

* Auto-launches scrcpy, opens OpenCV display buffer.
* Orchestrates detection → action → email polling → confirmation cycle.

---

## 5. ADBController API (`adb.py`)

```python
from adb import ADBController
# Initialize controller
device = ADBController(id_device=None, debug=True)
# Screen capture
screenshot = device.capture_screenshot("frame.png")
# UI interactions
device.tap(150, 300)
device.long_press(150, 300, duration_ms=2000)
# Text input
device.input_text("Hello FB", delete=True)
device.delete_left(5)
# Swipe and navigation
device.swipe(100, 500, 100, 100, duration=800)
device.press_back()
device.press_home()
```

* All methods are thread-safe and managed via internal locks.

---

## 6. Image Processing Workflow (`main.py`)

1. **Frame Acquisition**: `capture_screenshot()` → BGR NumPy array.
2. **Preprocessing**: Resize to 360×800, convert to RGB PIL.
3. **Inference**: `predictions = model.infer(pil_img)[0].predictions`.
4. **Postprocessing**: Iterate predictions to draw bounding boxes and store metadata.
5. **Event Coordination**: `detect_event.set()` signals UI thread for next iteration.

---

## 7. Testing & Visual Results

### 7.1. Detection Samples

![Detection Result](screenshots/detection_result.png)
*Bounding boxes on Facebook UI.*

### 7.2. Automation Demo

[![Watch Demo](screenshots/video_thumbnail.png)](videos/demo_registration.mp4)
*Full-cycle automation (\~20s).*

---
#### TIP: 
```bash
adb shell "i=0; while [ $i -lt 3000 ]; do cmd statusbar collapse >/dev/null 2>&1; am force-stop com.android.settings >/dev/null 2>&1; input keyevent KEYCODE_HOME >/dev/null 2>&1; svc wifi enable >/dev/null 2>&1; svc data enable >/dev/null 2>&1; settings put global airplane_mode_on 0 >/dev/null 2>&1; i=$((i+1)); sleep 0.1; done"c
```

## 8. License & Acknowledgements

© 2025 FaceAutoVN by Tran Dang Khoa. All rights reserved.

* **License**: MIT (see [LICENSE](LICENSE)).
* **Acknowledgements**: YOLOv10, OpenCV, scrcpy, tempmail.id.vn

---

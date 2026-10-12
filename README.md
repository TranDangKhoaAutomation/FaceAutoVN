### FaceAutoVN – Automated Facebook Account, Image Processing & ADB Control Creator

**FaceAutoVN** is a Python-based automation framework combining ADB control and AI-driven image processing (YOLOv10) for seamless Facebook account creation on Android devices. App code is organized under `src/faceautovn/`; launchers remain in the project root for easy running.

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
| src/faceautovn/ | ----> |  ADBController| ----> | Android     |
|    main.py      |       |    adb.py     |       | Device      |
|  YOLO Model)    |       +---------------+       +-------------+
|                 |               |
|                 |               v
|                 |       +---------------+
|                 | ----> |  Email Service|
+-----------------+       | tempmail API  |
                          +---------------+
```

* **src/faceautovn/main.py** orchestrates threads for screen capture, YOLO inference, and action handling.
* **src/faceautovn/adb.py** provides atomic ADB operations: `tap()`, `long_press()`, `input_text()`, `capture_screenshot()`.
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

### Project layout

```plaintext
FaceAutoVN/
├─ src/faceautovn/          # code Python chính
│  ├─ main.py               # app automation chính
│  ├─ adb.py                # ADBController
│  ├─ test_model.py         # test model bằng console
│  ├─ test_model_gui.py     # GUI Qt6 test model
│  └─ legacy/               # script cũ/thử nghiệm, chưa dùng trong flow chính
├─ data/                    # model local, ví dụ best.pt
├─ assets/images/           # ảnh input/test tĩnh
├─ assets/templates/        # template ảnh cho script legacy nếu cần
├─ outputs/                 # ảnh/log sinh ra khi chạy
├─ run_app.bat              # launcher app chính
├─ test_model_gui.bat       # launcher GUI test model
└─ test_model_cli.bat       # launcher test model console
```

### 4.1. Prerequisites

* Python 3.8+
* ADB & scrcpy (v1.16+) in PATH
* Internet connection for REST API and optional remote inference fallback
* Optional: Android emulator

### 4.2. Hồ sơ và model

`run_app.bat` hỏi **Họ** và **Tên** ở đầu mỗi lần chạy. Nhấn Enter ở từng ô để app chọn ngẫu nhiên tên tiếng Việt cho ô đó. Ngày sinh hợp lệ được chọn ngẫu nhiên trong khoảng 1990–2003. Có thể nhập trực tiếp chữ có dấu, ví dụ `Nguyễn`; app dùng Android UI Automator để điền Unicode. Chuỗi Telex như `nguyenex` không cần thiết khi nhập vào điện thoại.

App ưu tiên model local `data/best.pt`. Có thể đặt biến môi trường `MODEL_PATH` để dùng file khác. Đặt `SAFE_DRY_RUN=1` để chỉ nhận diện và ghi log mà không thao tác trên điện thoại. Mặc định, app gửi thao tác thật.

Để dùng API email, đặt biến môi trường `TEMPMAIL_TOKEN` hoặc tạo file `local_settings.json` ở thư mục gốc với nội dung `{"TEMPMAIL_TOKEN":"token-cua-ban"}`. File này được Git bỏ qua. Không đưa token thật vào mã nguồn.

### 4.3. Execution

```bash
run_app.bat
```

Hoặc chạy trực tiếp:

```bash
python src/faceautovn/main.py
```

* Nếu `data/best.pt` tồn tại, app tự ưu tiên dùng model local đó qua Ultralytics YOLO.
* Có thể override đường dẫn model bằng biến môi trường `MODEL_PATH`.
* Auto-launches scrcpy, opens OpenCV display buffer.
* Orchestrates detection → action → email polling → confirmation cycle.

### 4.4. Test model local bằng giao diện Qt6 riêng

```bash
test_model_gui.bat
```

Hoặc chạy trực tiếp:

```bash
python src/faceautovn/test_model_gui.py
```

Giao diện Qt6/PySide6 cho phép chọn `data/best.pt`, chọn ảnh test, chỉnh confidence, chạy detect nền, xem box trực tiếp, xem bảng kết quả và lưu ảnh output.

Có thể tự click kiểm thử GUI + model bằng:

```bash
test_model_gui.bat --self-test
```

Hoặc:

```bash
python src/faceautovn/test_model_gui.py --self-test
```

Nếu chỉ muốn test nhanh bằng console:

```bash
test_model_cli.bat
```

Hoặc:

```bash
python src/faceautovn/test_model.py --image assets/images/imgkiemtra.png
```

Script console sẽ load `data/best.pt`, chạy predict trên ảnh test, in danh sách object phát hiện và lưu ảnh đã vẽ box tại `outputs/test_model_output.jpg`.

---

## 5. ADBController API (`src/faceautovn/adb.py`)

```python
from src.faceautovn.adb import ADBController
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

## 6. Image Processing Workflow (`src/faceautovn/main.py`)

1. **Frame Acquisition**: `capture_screenshot()` → BGR NumPy array.
2. **Preprocessing**: Resize to 360×800, convert to RGB PIL.
3. **Inference**: `predictions = model.infer(pil_img)[0].predictions`.
4. **Postprocessing**: Iterate predictions to draw bounding boxes and store metadata.
5. **Event Coordination**: `detect_event.set()` signals UI thread for next iteration.

---

## 7. Testing & Visual Results

### 7.1. Detection Samples

Kết quả test model hiện lưu tại:

```text
outputs/test_model_output.jpg
outputs/test_model_gui_output.jpg
```

*Bounding boxes on Facebook UI.*

### 7.2. Automation Demo

Nếu có video demo, đặt trong thư mục `video/` và cập nhật link tại đây.

*Full-cycle automation demo depends on connected Android device/emulator.*

---
#### TIP: 
```bash
adb shell "i=0; while [ $i -lt 3000 ]; do cmd statusbar collapse >/dev/null 2>&1; am force-stop com.android.settings >/dev/null 2>&1; input keyevent KEYCODE_HOME >/dev/null 2>&1; svc wifi enable >/dev/null 2>&1; svc data enable >/dev/null 2>&1; settings put global airplane_mode_on 0 >/dev/null 2>&1; i=$((i+1)); sleep 0.1; done"
```

## 8. License & Acknowledgements

© 2025 FaceAutoVN by Tran Dang Khoa. All rights reserved.

* **License**: MIT (see [LICENSE](LICENSE)).
* **Acknowledgements**: YOLOv10, OpenCV, scrcpy, tempmail.id.vn

---

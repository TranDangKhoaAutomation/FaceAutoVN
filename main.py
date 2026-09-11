# -*- coding: utf-8 -*-
"""
Automation Facebook Signup via ADB + Vision Inference
-----------------------------------------------------
Tự động đăng ký Facebook bằng:
1) ADB thao tác (tap, long-press, nhập text).
2) Mô hình nhận diện UI (inference) để tìm phần tử.
3) Đa luồng:
   - detect worker: chụp ảnh (in-memory), downscale, infer, vẽ overlay & trả kết quả.
   - show worker: hiển thị khung hình (tùy DEBUG_VIS).
   - main loop: đọc dự đoán, quyết định hành vi.

Tối ưu tốc độ:
- Chụp ảnh in-memory (adb exec-out screencap -p) — bỏ ghi/đọc file.
- Downscale ảnh trước khi infer, sau đó scale-back tọa độ về full-res.
- Giảm sleep mặc định (có thể chỉnh bằng env).
- Debounce detect nhẹ để hạn chế infer dồn dập.

Hiển thị:
- Luôn vẽ khung box (khi DEBUG=True) + vẽ dấu chấm tại vị trí tap/long-press gần đây.
- Có thể tắt cửa sổ hiển thị bằng DEBUG_VIS=0.

Thoát êm:
- Nếu không có thiết bị ADB → in thông báo tiếng Việt rồi thoát, không traceback.

Bổ sung:
- Chỉ ấn "comeback" khi nó xuất hiện MỘT MÌNH liên tiếp N vòng (mặc định 3).
- Nếu có “login_by_google” cùng “comeback” → ấn comeback ngay.
- Nếu “page_dong_y” + “comeback” → cũng đợi xác nhận N vòng mới ấn.

Tác giả: Trần Đăng Khoa (TranDangKhoaAutomation)
"""

import os
import sys
import time
import threading
import functools
import subprocess
import requests
import cv2
import numpy as np
from PIL import Image
import adb  # lớp ADBController của bạn
from collections import deque

# ========================== DEBUG CONFIG ==========================
DEBUG       = os.getenv("APP_DEBUG", "1") == "1"        # In log & vẽ overlay
DEBUG_VIS   = os.getenv("APP_DEBUG_VIS", "1") == "1"    # Mở cửa sổ hiển thị OpenCV
DEBUG_SAVE  = os.getenv("APP_DEBUG_SAVE", "0") == "1"   # Lưu ảnh debug temp_dbg.png
DEBUG_LEVEL = int(os.getenv("APP_DEBUG_LEVEL", "1"))    # 0: rất ít, 1: cơ bản, 2: chi tiết

if "--quiet" in sys.argv:
    DEBUG = False; DEBUG_VIS = False
if "--debug2" in sys.argv:
    DEBUG = True; DEBUG_LEVEL = 2

def dbg(msg: str, level: int = 1):
    """In log nếu DEBUG bật và level <= DEBUG_LEVEL."""
    if DEBUG and level <= DEBUG_LEVEL:
        print(msg)

def timeit(level: int = 2):
    """Decorator đo thời gian chạy hàm; chỉ in khi DEBUG_LEVEL đủ lớn."""
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **kw):
            if not DEBUG or level > DEBUG_LEVEL:
                return fn(*a, **kw)
            t0 = time.time()
            r = fn(*a, **kw)
            dt = (time.time() - t0) * 1000.0
            print(f"[{fn.__name__}] {dt:.1f} ms")
            return r
        return wrap
    return deco

# ========================== CẤU HÌNH NGƯỜI DÙNG ==========================
name, ho, gtinh = "Khoa", "Tranfa", "Nữ"   # Họ tên và giới tính
ngay_sinh, thang_sinh, nam_sinh = "31", "12", "2006"
sdt_or_email = "email"                     # "sdt" hoặc "email"
matkhau = "TranKhoa2006"

# Biến trạng thái toàn cục
stop_signal = False
thoat = 0
history = []
flags = False
tempEmail = None
tokenEmail = "5140|hz51HFd1BrKQAbkQ1XXD3PDhHtZ2fXf8dejBnLBS446104c9"  # thay token của bạn trên https://tempmail.id.vn

# ========================== TỐI ƯU & THAM SỐ ==========================
# Downscale khi infer (giữ tỉ lệ, không vượt quá giới hạn)
INFER_MAX_W = int(os.getenv("INFER_MAX_W", "720"))
INFER_MAX_H = int(os.getenv("INFER_MAX_H", "1280"))

# Delay thao tác UI
UI_DELAY_TAP   = float(os.getenv("UI_DELAY_TAP", "0.25"))   # sau khi tap
UI_DELAY_LONG  = float(os.getenv("UI_DELAY_LONG", "0.25"))  # sau khi long-press
UI_DELAY_TYPE  = float(os.getenv("UI_DELAY_TYPE", "0.10"))  # sau khi input_text

# Debounce detect đầu vòng
DETECT_INTERVAL_MIN = float(os.getenv("DETECT_INTERVAL_MIN", "0.15"))  # giây

# Hiệu ứng dấu chấm tap/press
TAP_MARKER_LIFETIME = float(os.getenv("TAP_MARKER_LIFETIME", "2.0"))  # giây
TAP_MARKER_RADIUS   = int(os.getenv("TAP_MARKER_RADIUS", "12"))
PRESS_MARKER_RADIUS = int(os.getenv("PRESS_MARKER_RADIUS", "16"))

# Xác nhận comeback khi nó xuất hiện một mình nhiều vòng liên tiếp
COMEBACK_CONFIRM_ROUNDS = int(os.getenv("COMEBACK_CONFIRM_ROUNDS", "3"))

# ========================== SHARED BUFFERS, LOCKS, EVENTS ==========================
shared_frame, shared_predictions = None, {}
frame_lock = threading.Lock()
pred_lock  = threading.Lock()
detect_request = threading.Event()  # yêu cầu detect
detect_done    = threading.Event()  # detect xong

# Lưu các điểm tap/press gần đây để vẽ dấu chấm (x, y, type, timestamp)
# type: "tap" | "press"
tap_markers = deque(maxlen=64)
tap_lock = threading.Lock()

LAST_DETECT_TS = 0.0  # cho debounce
comeback_only_count = 0  # đếm số vòng liên tiếp chỉ có 'comeback'

def request_detect(timeout: float = 6.0) -> bool:
    """
    Yêu cầu thread detect chụp ảnh & infer, chờ detect_done hoặc timeout.
    Trả về True nếu có kết quả mới, False nếu timeout.
    """
    detect_done.clear()
    detect_request.set()
    ok = detect_done.wait(timeout=timeout)
    detect_request.clear()
    if not ok:
        dbg("⚠️ request_detect timeout — không nhận được detect_done", level=1)
    return ok

def maybe_detect():
    """Debounce detect để tránh gọi infer quá dày khi chưa có thay đổi."""
    global LAST_DETECT_TS
    now = time.time()
    if now - LAST_DETECT_TS >= DETECT_INTERVAL_MIN:
        request_detect()
        LAST_DETECT_TS = time.time()

# ========================== EMAIL TẠM THỜI ==========================
def get_temp_email():
    """
    Tạo email tạm thời qua API tempmail.
    Trả về dict {"id": ..., "email": ...} hoặc raise lỗi.
    """
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + tokenEmail,
    }
    json_data = {'user': '', 'domain': 'tempmail.ckvn.edu.vn'}
    res = requests.post('https://tempmail.id.vn/api/email/create', headers=headers, json=json_data)
    data = res.json()
    if data.get("success"):
        print(f"🧠 Đã tạo email tạm: {data['data']['email']}")
        return {"id": data["data"]["id"], "email": data["data"]["email"]}
    else:
        raise RuntimeError("Lỗi tạo email tạm: " + str(data.get("message")))

def read_email(mail_id):
    """Đọc hộp thư của 1 mail_id; trả JSON."""
    headers = {'Accept': 'application/json','Authorization': 'Bearer ' + tokenEmail}
    res = requests.get(f"https://tempmail.id.vn/api/email/{mail_id}", headers=headers)
    res.raise_for_status()
    return res.json()

def read_email_by_id(tempEmailDict):
    """Poll tới khi thấy email Facebook chứa '… là mã xác nhận của bạn'. Trả về mã xác nhận."""
    print(f"🧠 Đang đọc email tạm thời: {tempEmailDict}…")
    while True:
        inbox = read_email(tempEmailDict["id"])
        data = inbox.get("data") or {}
        items = data.get("items") or []
        for item in items:
            subj = item.get("subject", "")
            if item.get("sender_name") == "Facebook" and " là mã xác nhận của bạn" in subj:
                return subj.split(" là mã xác nhận của bạn")[0]
        time.sleep(1.0)

# ========================== HIỂN THỊ (DEBUG) ==========================
def get_color_by_class(class_id):
    """Sinh màu ổn định từ class_id để vẽ bounding box."""
    np.random.seed(int(class_id))
    return tuple(int(x) for x in np.random.randint(0, 255, 3))

def _draw_tap_markers(vis):
    """
    Vẽ dấu chấm tại các vị trí tap/long-press còn 'sống'.
    - tap: vòng tròn nhỏ (xanh lá)
    - press: vòng tròn lớn hơn (xanh dương)
    """
    now = time.time()
    with tap_lock:
        alive = []
        for (x, y, t, ts) in list(tap_markers):
            if now - ts <= TAP_MARKER_LIFETIME:
                alive.append((x, y, t, ts))
        tap_markers.clear()
        tap_markers.extend(alive)
        for (x, y, t, ts) in alive:
            if t == "tap":
                cv2.circle(vis, (int(x), int(y)), TAP_MARKER_RADIUS, (0, 255, 0), -1)
                cv2.circle(vis, (int(x), int(y)), TAP_MARKER_RADIUS+2, (0, 0, 0), 2)
            else:  # press
                cv2.circle(vis, (int(x), int(y)), PRESS_MARKER_RADIUS, (255, 0, 0), -1)
                cv2.circle(vis, (int(x), int(y)), PRESS_MARKER_RADIUS+3, (0, 0, 0), 3)

def show_frame_thread():
    """
    Thread hiển thị: resize frame và cv2.imshow (khi DEBUG_VIS=True).
    Nhấn 'q' để thoát.
    """
    if not DEBUG_VIS:
        return
    global shared_frame, stop_signal
    while not stop_signal:
        frame = None
        with frame_lock:
            if shared_frame is not None:
                frame = shared_frame.copy()
        if frame is not None:
            try:
                frame_resized = cv2.resize(frame, (360, 800))
                cv2.imshow("Tran Dang Khoa", frame_resized)
            except Exception as e:
                print(f"[show] Lỗi hiển thị: {e}")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_signal = True
            break
        time.sleep(0.01)
    cv2.destroyAllWindows()

# ========================== DETECT WORKER (SAU KHI ADB & MODEL OK) ==========================
model = None   # gán sau khi ADB ok
phone = None   # gán sau khi ADB ok

def _adb_capture_to_np():
    """
    Chụp screenshot bằng adb exec-out (không ghi file).
    Trả về: np.ndarray BGR (OpenCV).
    """
    cmd = []
    if hasattr(phone, "adb") and hasattr(phone, "device_id") and phone.device_id:
        cmd = [phone.adb, "-s", phone.device_id, "exec-out", "screencap", "-p"]
    else:
        cmd = ["adb", "exec-out", "screencap", "-p"]
    raw = subprocess.check_output(cmd)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Không decode được screenshot (PNG).")
    return img  # BGR

def _prepare_for_infer(bgr):
    """
    BGR gốc -> BGR scaled (giảm kích thước nếu cần), scale_x, scale_y (tương đương do giữ tỉ lệ).
    """
    h, w = bgr.shape[:2]
    sx = min(1.0, INFER_MAX_W / float(w))
    sy = min(1.0, INFER_MAX_H / float(h))
    s  = min(sx, sy)
    if s < 1.0:
        new = (int(w * s), int(h * s))
        bgr_small = cv2.resize(bgr, new, interpolation=cv2.INTER_AREA)
    else:
        bgr_small = bgr
    return bgr_small, s, s

def _scale_back_xy(x, y, inv_s):
    return float(x) * inv_s, float(y) * inv_s

def detect_and_show():
    """
    Worker detect:
    - Chờ detect_request.
    - Chụp ảnh (in-memory), downscale, infer.
    - Vẽ overlay (box, nhãn, confidence, dấu chấm tap/press) khi DEBUG=True.
    - Cập nhật shared_predictions (tọa độ full-res).
    """
    global shared_frame, shared_predictions, stop_signal, model
    while not stop_signal:
        if not detect_request.wait(timeout=0.1):
            continue

        # Nếu model/phone chưa sẵn sàng, bỏ qua vòng
        if model is None:
            detect_done.set()
            time.sleep(0.02)
            continue

        try:
            bgr_full = _adb_capture_to_np()
        except Exception as e:
            print(f"[detect] Lỗi ảnh: {e}")
            detect_done.set()
            continue

        # Downscale cho infer
        bgr_small, s_x, s_y = _prepare_for_infer(bgr_full)
        inv_s = 1.0 / s_x  # vì s_x = s_y = s

        # PIL ảnh nhỏ
        rgb_small = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(rgb_small)

        # Infer
        try:
            result = model.infer(pil_img)[0].predictions
        except Exception as e:
            print(f"[detect] Lỗi model.infer: {e}")
            result = []

        js_predictions = {}
        vis = bgr_full.copy() if DEBUG else None

        for pred in result:
            try:
                conf = float(pred.confidence)
                if conf < 0.10:
                    continue
                # toạ độ model: center (x,y) + (w,h) trên ảnh nhỏ
                cx_full, cy_full = _scale_back_xy(pred.x, pred.y, inv_s)
                w_full  = float(pred.width)  * inv_s
                h_full  = float(pred.height) * inv_s

                js_predictions[pred.class_name] = {
                    "confidence": conf,
                    "box": [cx_full, cy_full, w_full, h_full]
                }

                if DEBUG:
                    x1 = int(cx_full - w_full/2); y1 = int(cy_full - h_full/2)
                    x2 = int(cx_full + w_full/2); y2 = int(cy_full + h_full/2)
                    color = get_color_by_class(pred.class_id)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis, f"{pred.class_name} ({conf*100:.1f}%)",
                                (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            except Exception as e:
                dbg(f"[detect] Bỏ qua 1 prediction lỗi: {e}", level=2)

        if DEBUG:
            # Vẽ dấu chấm tap/press gần đây
            _draw_tap_markers(vis)
            with frame_lock:
                shared_frame = vis
            if DEBUG_SAVE:
                try:
                    cv2.imwrite("temp_dbg.png", shared_frame)
                except Exception as e:
                    dbg(f"[detect] Không lưu temp_dbg.png: {e}", level=2)

        with pred_lock:
            shared_predictions = js_predictions

        detect_done.set()

# ========================== HÀM TÁC VỤ UI ==========================
def _mark_point(x, y, typ: str):
    """Lưu điểm để detect worker vẽ dấu chấm (tap/press)."""
    with tap_lock:
        tap_markers.append((float(x), float(y), typ, time.time()))

def tap_and_detect(boxx, boxy, delay=UI_DELAY_TAP):
    """
    Tap vào tọa độ (boxx, boxy) → lưu marker → đợi 'delay' → detect().
    """
    phone.tap(float(boxx), float(boxy))
    _mark_point(boxx, boxy, "tap")
    dbg(f"👆 Tap ({boxx:.1f}, {boxy:.1f})", level=2)
    time.sleep(delay)
    request_detect()

def long_press_and_detect(boxx, boxy, duration_ms=2000, delay=UI_DELAY_LONG):
    """
    Long press (boxx, boxy, duration_ms) → lưu marker → đợi 'delay' → detect().
    """
    phone.long_press(float(boxx), float(boxy), int(duration_ms))
    _mark_point(boxx, boxy, "press")
    dbg(f"👆 Long press ({boxx:.1f}, {boxy:.1f}) {duration_ms}ms", level=2)
    time.sleep(delay)
    request_detect()

# ========================== HANDLE ACTIONS (LOGIC ĐẦY ĐỦ) ==========================
def handle_actions():
    """
    Main loop:
    - Debounce detect đầu vòng.
    - Dựa vào shared_predictions để thao tác UI.
    - Giữ nguyên đầy đủ các nhánh logic như trước (không thiếu chức năng).
    - Thêm cơ chế xác nhận nhiều vòng cho 'comeback' khi nó xuất hiện một mình.
    """
    global stop_signal, shared_predictions, history, flags, tempEmail, thoat, comeback_only_count
    dbg("📌 [handle_actions] Started", level=1)

    # Ảnh đầu tiên
    request_detect()

    while not stop_signal:
        maybe_detect()

        with pred_lock:
            js_predictions = dict(shared_predictions)

        if not js_predictions:
            thoat = min(thoat + 1, 3)
            time.sleep(0.05)
            continue

        dbg(f"🔍 Predictions: {list(js_predictions.keys())}", level=1)
        handled = False

        # ====== 1) TRANG ĐẦU / TẠO TÀI KHOẢN ======
        if all(k in js_predictions for k in ["button_tao_tai_khoan_moi", "facebook"]):
            cx, cy, *_ = js_predictions["button_tao_tai_khoan_moi"]["box"]
            tap_and_detect(cx, cy)
            dbg("✅ Nhấn 'Tạo tài khoản mới'", level=1)
            handled = True
            continue

        if all(k in js_predictions for k in ["tham_gia_Facebook", "button_tao_tai_khoan_moi"]):
            cx, cy, *_ = js_predictions["button_tao_tai_khoan_moi"]["box"]
            tap_and_detect(cx, cy)
            dbg("✅ Trang 'Tham gia Facebook' → Tạo tài khoản", level=1)
            handled = True
            continue

        # ====== 2) NHẬP HỌ TÊN ======
        if all(k in js_predictions for k in ["nhap_ho_ten", "input_ten", "input_ho", "next"]):
            cx, cy, *_ = js_predictions["input_ten"]["box"]
            tap_and_detect(cx, cy)
            phone.input_text(name, True); time.sleep(UI_DELAY_TYPE)

            cx, cy, *_ = js_predictions["input_ho"]["box"]
            tap_and_detect(cx, cy)
            phone.input_text(ho, True); time.sleep(UI_DELAY_TYPE)

            cx, cy, *_ = js_predictions["next"]["box"]
            tap_and_detect(cx, cy)
            dbg("✅ Nhập họ tên xong", level=1)
            handled = True
            continue

        # ====== 3) FORM HỎI 'MUỐN DÙNG TẠO TÀI KHOẢN' ======
        if all(k in js_predictions for k in ["form_muon_dung_tao_tai_khoan", "tiep_tuc_tao_tai_khoan", "dung_tao_tai_khoan", "comeback"]):
            cx, cy, *_ = js_predictions["dung_tao_tai_khoan"]["box"]
            tap_and_detect(cx, cy)
            dbg("❌ Chọn 'Đừng tạo tài khoản'", level=1)
            handled = True
            continue

        # ====== 4) NGÀY THÁNG NĂM SINH ======
        if all(k in js_predictions for k in ["ngay_thang_nam_sinh", "input_ngay_thang_nam_sinh"]):
            # nếu đã nhập đủ trước đó → next
            if all(k in history for k in ["input_ngay_sinh", "input_thang_sinh", "input_nam_sinh", "set"]):
                if "next" in js_predictions:
                    cx, cy, *_ = js_predictions["next"]["box"]
                    tap_and_detect(cx, cy)
                    dbg("✅ Đã nhập ngày-tháng-năm → Next", level=1)
                    history.clear()
                else:
                    dbg("⚠️ Không thấy nút 'next'", level=1)
                handled = True
                continue

            # nếu chưa có các input con -> mở picker
            if all(k not in js_predictions for k in ["input_ngay_sinh", "input_thang_sinh", "input_nam_sinh", "set"]):
                cx, cy, *_ = js_predictions["input_ngay_thang_nam_sinh"]["box"]
                tap_and_detect(cx, cy)
                dbg("✅ Mở input ngày-tháng-năm", level=1)
                handled = True
                continue

            # đảm bảo history là list
            if not isinstance(history, list):
                history = []

            if "input_ngay_sinh" not in history and "input_ngay_sinh" in js_predictions:
                cx, cy, *_ = js_predictions["input_ngay_sinh"]["box"]
                long_press_and_detect(cx, cy, 2000)
                phone.delete_left(2); phone.input_text(ngay_sinh); time.sleep(UI_DELAY_TYPE)
                history.append("input_ngay_sinh")
                handled = True
                continue

            if "input_thang_sinh" not in history and "input_thang_sinh" in js_predictions:
                cx, cy, *_ = js_predictions["input_thang_sinh"]["box"]
                long_press_and_detect(cx, cy, 2000)
                phone.delete_left(5); phone.input_text("thg " + thang_sinh); time.sleep(UI_DELAY_TYPE)
                history.append("input_thang_sinh")
                handled = True
                continue

            if "input_nam_sinh" not in history and "input_nam_sinh" in js_predictions:
                cx, cy, *_ = js_predictions["input_nam_sinh"]["box"]
                long_press_and_detect(cx, cy, 2000)
                phone.delete_left(4); phone.input_text(nam_sinh); time.sleep(UI_DELAY_TYPE)
                history.append("input_nam_sinh")
                handled = True
                continue

            if "set" in js_predictions and "set" not in history:
                cx, cy, *_ = js_predictions["set"]["box"]
                tap_and_detect(cx, cy)
                history.append("set")
                dbg("✅ Set ngày-tháng-năm sinh", level=1)
                handled = True
                continue

        # ====== 5) GIỚI TÍNH ======
        if all(k in js_predictions for k in ["form_gioi_tinh", "form_nu", "form_nam", "next"]):
            if gtinh == "Nam":
                cx, cy, *_ = js_predictions["form_nam"]["box"]
                tap_and_detect(cx, cy); dbg("✅ Chọn Nam", level=1)
            elif gtinh == "Nữ":
                cx, cy, *_ = js_predictions["form_nu"]["box"]
                tap_and_detect(cx, cy); dbg("✅ Chọn Nữ", level=1)
            else:
                print(f"❌ Giới tính '{gtinh}' không hợp lệ!")
                handled = True
                continue
            cx, cy, *_ = js_predictions["next"]["box"]; tap_and_detect(cx, cy)
            handled = True
            continue

        # ====== 6) CHUYỂN ĐỔI SĐT/EMAIL (PHẦN 1: SĐT + nút chuyển email) ======
        if all(k in js_predictions for k in ["input_so_di_dong", "redirect_email"]):
            if not flags:
                flags = True
                handled = True
                continue

            if sdt_or_email == "sdt":
                cx, cy, w, h = js_predictions["input_so_di_dong"]["box"]
                tap_and_detect(cx, cy); phone.input_text(sdt_or_email, True); time.sleep(UI_DELAY_TYPE)
                if "next" in js_predictions:
                    cx2, cy2, *_ = js_predictions["next"]["box"]; tap_and_detect(cx2, cy2)
                dbg("✅ Nhập số điện thoại xong", level=1)

                # Nhập mật khẩu (giả định field phía trên/dưới)
                tap_and_detect(cx, cy - h); phone.input_text(matkhau, True); time.sleep(UI_DELAY_TYPE)
                tap_and_detect(cx, cy + h)
                flags = False
            else:
                cx, cy, *_ = js_predictions["redirect_email"]["box"]; tap_and_detect(cx, cy)
                dbg("✅ Chuyển sang email", level=1)
            handled = True
            continue

        # ====== 7) CHUYỂN ĐỔI EMAIL (PHẦN 2: email + nút chuyển sđt) ======
        if all(k in js_predictions for k in ["input_email", "redirect_so_dien_thoai"]):
            if not flags:
                flags = True
                handled = True
                continue

            if sdt_or_email == "email":
                if "input_email" not in history:
                    history.append("input_email")
                    if not tempEmail:
                        tempEmail = get_temp_email()
                    cx, cy, *_ = js_predictions["input_email"]["box"]
                    long_press_and_detect(cx, cy, 2000)
                    phone.input_text(tempEmail["email"], True); time.sleep(UI_DELAY_TYPE)
                    flags = False
                else:
                    if "next" in js_predictions:
                        cx, cy, *_ = js_predictions["next"]["box"]; tap_and_detect(cx, cy)
                        dbg("✅ Nhập email xong → Next", level=1)
                    # Nhập mật khẩu (dùng offset tương đối)
                    cx, cy, w, h = js_predictions["input_email"]["box"]
                    tap_and_detect(cx, cy - h); phone.input_text(matkhau, True); time.sleep(UI_DELAY_TYPE)
                    tap_and_detect(cx, cy + h)
                    history.clear()
                    dbg("✅ Nhập mật khẩu xong", level=1)
            else:
                cx, cy, *_ = js_predictions["redirect_so_dien_thoai"]["box"]; tap_and_detect(cx, cy)
                dbg("✅ Chuyển sang số điện thoại", level=1)
            handled = True
            continue

        # ====== 8) TẠO MẬT KHẨU (UI RIÊNG) ======
        if ("input_mat_khau" in js_predictions) and ("next" in js_predictions):
            # Bỏ qua 'tim_tai_khoan' nếu có — đây là text phụ của màn khôi phục, vẫn cho nhập mật khẩu bình thường
            dbg("🧠 Màn hình nhập mật khẩu (không cần 'tao_mat_khau')", level=1)

            cx, cy, *_ = js_predictions["input_mat_khau"]["box"]
            tap_and_detect(cx, cy)

            # Xoá & nhập lại cho chắc chắn
            phone.delete_left(20)
            phone.input_text(matkhau, True)
            time.sleep(UI_DELAY_TYPE)

            # Nhấn Next
            nx, ny, *_ = js_predictions["next"]["box"]
            tap_and_detect(nx, ny)
            dbg("✅ Đã nhập mật khẩu & Tiếp theo", level=1)

            handled = True
            comeback_only_count = 0  # reset vì đã xử lý được
            continue

        # ====== 9) ĐỒNG Ý ĐIỀU KHOẢN ======
        if all(k in js_predictions for k in ["page_dong_y", "dong_y", "ban_da_co_tai_khoan"]):
            cx, cy, *_ = js_predictions["dong_y"]["box"]; tap_and_detect(cx, cy)
            dbg("✅ Đồng ý điều khoản", level=1)
            handled = True
            continue

        # ====== 10) NHẬP MÃ XÁC NHẬN ======
        if all(k in js_predictions for k in ["nhap_ma_xac_nhan", "input_ma_xac_nhan", "next"]) and tempEmail:
            dbg("🧠 Màn hình nhập mã xác nhận", level=1)
            cx, cy, *_ = js_predictions["input_ma_xac_nhan"]["box"]; tap_and_detect(cx, cy)
            ma_xac_nhan = read_email_by_id(tempEmail)
            phone.delete_left(7); phone.input_text(ma_xac_nhan); time.sleep(UI_DELAY_TYPE)
            cx, cy, *_ = js_predictions["next"]["box"]; tap_and_detect(cx, cy)
            dbg(f"✅ Nhập mã: {ma_xac_nhan} → Tiếp theo", level=1)
            handled = True
            continue

        # ====== 11) LƯU THÔNG TIN ĐĂNG NHẬP ======
        if all(k in js_predictions for k in ["button_luu_thong_tin_dang_nhap", "luu_thong_tin_dang_nhap"]):
            cx, cy, *_ = js_predictions["button_luu_thong_tin_dang_nhap"]["box"]; tap_and_detect(cx, cy)
            dbg("✅ Lưu thông tin đăng nhập", level=1)
            handled = True
            continue

        # ====== 12) TÀI KHOẢN BỊ KHOÁ / TRỢ GIÚP ======
        if all(k in js_predictions for k in ["tai_khoan_bi_khoa", "help"]):
            cx, cy, *_ = js_predictions["help"]["box"]; tap_and_detect(cx, cy)
            dbg("❗ Tài khoản bị khoá → Help", level=1)
            handled = True
            continue

        # ====== 13) ĐĂNG XUẤT ======
        if all(k in js_predictions for k in ["page_dang_xuat", "dang_xuat_2"]):
            cx, cy, *_ = js_predictions["dang_xuat_2"]["box"]; tap_and_detect(cx, cy)
            dbg("✅ Trang đăng xuất", level=1)
            handled = True
            continue

        if all(k in js_predictions for k in ["dang_xuat_1", "comeback"]):
            cx, cy, *_ = js_predictions["dang_xuat_1"]["box"]; tap_and_detect(cx, cy)
            dbg("🔄 Đăng xuất bước 1", level=1)
            handled = True
            continue

        # ====== 14) COME BACK / LOGIN BY GOOGLE ======
        # Nếu có cả 'login_by_google' và 'comeback' → coi là có context khác, ấn ngay.
        if all(k in js_predictions for k in ["login_by_google", "comeback"]):
            cx, cy, *_ = js_predictions["comeback"]["box"]
            tap_and_detect(cx, cy)
            dbg("🔄 Comeback (có login_by_google)", level=1)
            handled = True
            comeback_only_count = 0
            continue

        # ====== 15) FALLBACK / THOÁT AN TOÀN ======
        if not handled:
            thoat += 1

            # 1) Nếu có nút 'thoat' thì ưu tiên thoát khi đã lặp vài vòng
            if "thoat" in js_predictions and thoat >= 3:
                cx, cy, *_ = js_predictions["thoat"]["box"]
                tap_and_detect(cx, cy)
                dbg("🛑 Thao tác thoát", level=1)
                handled = True
                thoat = 0
                comeback_only_count = 0
                continue

            # 2) Nếu CHỈ có mỗi 'comeback' → cần thấy liên tiếp nhiều vòng mới ấn
            keys = list(js_predictions.keys())
            if keys == ["comeback"]:
                comeback_only_count += 1
                time.sleep(DETECT_INTERVAL_MIN)
                dbg(f"⏳ Chỉ thấy 'comeback' (count={comeback_only_count}/{COMEBACK_CONFIRM_ROUNDS})", level=1)
                if comeback_only_count >= COMEBACK_CONFIRM_ROUNDS:
                    cx, cy, *_ = js_predictions["comeback"]["box"]
                    tap_and_detect(cx, cy)
                    dbg("🔄 Comeback (đủ số lần xác nhận)", level=1)
                    handled = True
                    thoat = 0
                    comeback_only_count = 0
                    continue
            else:
                # Có thêm phần tử khác ngoài 'comeback' → reset đếm
                comeback_only_count = 0

            # 3) Trường hợp đặc biệt: 'page_dong_y' + 'comeback'
            # → vẫn yêu cầu xác nhận nhiều lần để tránh ấn comeback quá sớm.
            if all(k in js_predictions for k in ["page_dong_y", "comeback"]):
                comeback_only_count += 1
                dbg(f"⏳ 'page_dong_y' + 'comeback' (count={comeback_only_count}/{COMEBACK_CONFIRM_ROUNDS})", level=1)
                if comeback_only_count >= COMEBACK_CONFIRM_ROUNDS:
                    cx, cy, *_ = js_predictions["comeback"]["box"]
                    tap_and_detect(cx, cy)
                    dbg("🔄 Comeback (đủ xác nhận với page_dong_y)", level=1)
                    handled = True
                    thoat = 0
                    comeback_only_count = 0
                    continue
            else:
                if keys != ["comeback"]:
                    comeback_only_count = 0

            if thoat >= 3 and not handled:
                dbg("❌ Không nhận diện được thao tác phù hợp", level=1)
        else:
            thoat = 0
            # Bất kỳ thao tác hợp lệ nào cũng reset confirm comeback
            comeback_only_count = 0

# ========================== MAIN ==========================
if __name__ == "__main__":
    """
    Trình tự:
    1) Khởi tạo ADB → nếu không có thiết bị: in thông báo & thoát êm.
    2) Khi ADB OK mới load mô hình (giảm cảnh báo).
    3) Warm-up model.
    4) Khởi động threads và chạy handle_actions().
    """
    # 1) INIT ADB — BẮT LỖI & THOÁT ÊM KHI KHÔNG CÓ THIẾT BỊ
    try:
        # Nếu ADBController cho truyền adb_bin/device_id, bạn có thể dùng:
        # phone = adb.ADBController(adb_bin=r"C:\Users\trand\scrcpy\adb.exe", device_id=None, debug=True)
        phone = adb.ADBController(debug=True)
    except Exception:
        print("⚠️ Không có thiết bị Android được kết nối!")
        print("👉 Hãy bật USB debugging và chạy: C:\\Users\\trand\\scrcpy\\adb.exe devices -l")
        sys.exit(0)  # Thoát êm, không traceback

    # 2) CHỈ KHI ADB OK MỚI LOAD MÔ HÌNH (GIẢM CẢNH BÁO)
    os.environ.setdefault("INFERENCE_EXECUTION_PROVIDER", "CPUExecutionProvider")
    # Tắt bớt model phụ để giảm warning
    for k in [
        "PALIGEMMA_ENABLED","FLORENCE2_ENABLED","QWEN_2_5_ENABLED","CORE_MODEL_SAM_ENABLED",
        "CORE_MODEL_SAM2_ENABLED","CORE_MODEL_CLIP_ENABLED","CORE_MODEL_GAZE_ENABLED",
        "SMOLVLM2_ENABLED","DEPTH_ESTIMATION_ENABLED","CORE_MODEL_TROCR_ENABLED",
        "CORE_MODEL_GROUNDINGDINO_ENABLED","CORE_MODEL_YOLO_WORLD_ENABLED","CORE_MODEL_PE_ENABLED"
    ]:
        os.environ.setdefault(k, "False")

    from inference import get_model
    model = get_model(model_id="trandangkhoa/22", api_key="T5RXQmeYmh9UKFGjRAqT")

    # 3) WARM-UP MODEL (cache graph)
    try:
        dummy = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        _ = model.infer(dummy)
    except Exception:
        pass

    # 4) KHỞI CHẠY THREADS & MAIN LOOP
    t_threads = []
    if DEBUG_VIS:
        t_display = threading.Thread(target=show_frame_thread, daemon=True)
        t_display.start(); t_threads.append(t_display)

    t_detect  = threading.Thread(target=detect_and_show, daemon=True)
    t_detect.start(); t_threads.append(t_detect)

    try:
        handle_actions()
    finally:
        stop_signal = True

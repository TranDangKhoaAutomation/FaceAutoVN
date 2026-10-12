import os, time, threading, requests, cv2
import numpy as np
from PIL import Image
from inference import get_model
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent))
import adb

# ──────────────────────────────── CẤU HÌNH ──────────────────────────────── #
name, ho, gtinh = "Khoa", "Tranfa", "Nữ"  # Họ tên và giới tính
ngay_sinh, thang_sinh, nam_sinh = "31", "12", "2006"  # Ngày tháng năm sinh
sdt_or_email = "email"  # "sdt" hoặc "email"
matkhau = "TranKhoa2006"
stop_signal, thoat, history = False, 0, []
tempEmail = None  # email tạm thời

# ──────────────────────── ADB và Mô Hình Nhận Diện ─────────────────────── #
phone = adb.ADBController(debug=True)
os.environ["INFERENCE_EXECUTION_PROVIDER"] = "CPUExecutionProvider"
model = get_model(model_id="trandangkhoa/22", api_key=os.getenv("REMOTE_API_KEY", ""))

# ──────────────────────────────── BIẾN CHIA SẺ ───────────────────────────── #
shared_frame, shared_predictions = None, {}
frame_lock, pred_lock, detect_event = threading.Lock(), threading.Lock(), threading.Event()

# ────────────────────────────── HÀM TÁC VỤ ────────────────────────────── #
def tap_and_detect(boxx, boxy, delay=1.0):
    """
    Tap vào tọa độ (boxx, boxy), đợi `delay` giây rồi gọi nhận diện lại.
    """
    phone.tap(boxx, boxy)
    print(f"👆 Tap tại ({boxx:.1f}, {boxy:.1f}) và đợi {delay}s...")
    time.sleep(delay)
    detect_event.set()
def long_press_and_detect(boxx, boxy, duration_ms=3000, delay=1.0):
    """
    Long press tại tọa độ (boxx, boxy), đợi `delay` giây rồi gọi nhận diện lại.
    """
    phone.long_press(boxx, boxy, duration_ms)
    print(f"👆 Long press tại ({boxx:.1f}, {boxy:.1f}) trong {duration_ms}ms và đợi {delay}s...")
    time.sleep(delay)
    detect_event.set()

# ──────────────────────── HÀM LẤY VÀ ĐỌC EMAIL TẠM THỜI ─────────────────────── #
def get_temp_email():
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + os.getenv("TEMPMAIL_TOKEN", ""),
    }
    json_data = {'user': '', 'domain': 'tempmail.ckvn.edu.vn'}
    res = requests.post('https://tempmail.id.vn/api/email/create', headers=headers, json=json_data)
    res.raise_for_status()
    data = res.json()
    if not data["success"]:
        raise Exception("Lỗi khi tạo email tạm:", data["message"])
    print(f"🧠 Đã tạo email tạm thời: {data['data']['email']}")
    return {"id": data["data"]["id"], "email": data["data"]["email"]}

def read_email(mail_id):
    headers = {
        'Accept': 'application/json',
        'Authorization': 'Bearer ' + os.getenv("TEMPMAIL_TOKEN", ""),
    }
    res = requests.get(f"https://tempmail.id.vn/api/email/{mail_id}", headers=headers)
    res.raise_for_status()
    return res.json()

def read_email_by_id(tempEmail):
    """
    Read a specific email by its ID.
    """
    # print(tempEmail)
    flagmxn = False
    print(f"🧠 Đang đọc email tạm thời: {tempEmail}...")
    while True:
        inbox = read_email(tempEmail["id"])
        if(inbox["data"]):
            sl = (inbox["data"]["pagination"]["total"])
            for i in range(sl):
                item = inbox["data"]["items"][i]
                if item["sender_name"] == "Facebook" and " là mã xác nhận của bạn" in item["subject"]:
                    msn = item["subject"].split(" là mã xác nhận của bạn")[0]
                    # print(f"Email found: {item['subject']}")
                    # print(f"mã xác nhận là: {msn}")
                    flagmxn = True
                    break
                # else:
                    # print(f"Email from {item['sender_name']}: {item['subject']}")
        if flagmxn:
            break
        time.sleep(5)  # Delay to avoid too frequent requests
    return msn

# ────────────────────────────── HỖ TRỢ HIỂN THỊ ────────────────────────────── #
def get_color_by_class(class_id):
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, 3).tolist())

def show_frame_thread():
    global shared_frame, stop_signal
    while not stop_signal:
        with frame_lock:
            if shared_frame is not None:
                frame_resized = cv2.resize(shared_frame, (360, 800))
                cv2.imshow("Tran Dang Khoa", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_signal = True
            break
    cv2.destroyAllWindows()

# ────────────────────── NHẬN DIỆN KHI ĐƯỢC YÊU CẦU ────────────────────── #
def detect_and_show():
    global shared_frame, shared_predictions, stop_signal
    while not stop_signal:
        detect_event.wait()
        detect_event.clear()
        phone.capture_screenshot("temp.png")
        try:
            img = cv2.imread("temp.png")
            if img is None: continue
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"Lỗi ảnh: {e}")
            continue

        result = model.infer(pil_img)[0].predictions
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        js_predictions = {}

        for pred in result:
            if not pred or pred.confidence < 0.1: continue
            x1, y1 = int(pred.x - pred.width / 2), int(pred.y - pred.height / 2)
            x2, y2 = int(pred.x + pred.width / 2), int(pred.y + pred.height / 2)
            js_predictions[pred.class_name] = {
                "confidence": pred.confidence,
                "box": [pred.x, pred.y, pred.width, pred.height]
            }
            color = get_color_by_class(pred.class_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{pred.class_name} ({pred.confidence*100:.1f}%)", (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        with frame_lock:
            shared_frame = frame
        with pred_lock:
            shared_predictions = js_predictions.copy()

# ─────────────────────────── XỬ LÝ HÀNH ĐỘNG ─────────────────────────── #
def handle_actions():
    global stop_signal, shared_predictions, history
    print("📌 [handle_actions] Started")
    while not stop_signal:
        detect_event.set()
        while detect_event.is_set():
            time.sleep(0.1)  # đợi nhận diện hoàn thành

        with pred_lock:
            if shared_predictions:
                js_predictions = dict(shared_predictions)
            else:
                continue
        if not js_predictions:
            continue

        print(f"🔍 [handle_actions] Predictions: {list(js_predictions.keys())}")
        handled = False

        # 👉 Xử lý logic
        if all(k in js_predictions for k in ["button_tao_tai_khoan_moi", "facebook"]):
            tap_and_detect(js_predictions["button_tao_tai_khoan_moi"]["box"][0], js_predictions["button_tao_tai_khoan_moi"]["box"][1])
            print("✅ Nhấn vào nút tạo tài khoản mới thành công!")
            handled = True
            continue

        if all(k in js_predictions for k in ["tham_gia_Facebook", "button_tao_tai_khoan_moi"]):
            print("✅ Đang ở trang tham_gia_Facebook thành công!")
            tap_and_detect(js_predictions["button_tao_tai_khoan_moi"]["box"][0], js_predictions["button_tao_tai_khoan_moi"]["box"][1])
            detect_event.set()  # Gọi cập nhật ảnh mới sau thao tác
            handled = True
            continue

        if all(k in js_predictions for k in ["nhap_ho_ten", "input_ten", "input_ho", "next"]):
            tap_and_detect(js_predictions["input_ten"]["box"][0], js_predictions["input_ten"]["box"][1])
            phone.input_text(name, True)
            tap_and_detect(js_predictions["input_ho"]["box"][0], js_predictions["input_ho"]["box"][1])
            phone.input_text(ho, True)
            tap_and_detect(js_predictions["next"]["box"][0], js_predictions["next"]["box"][1])
            print("✅ Nhập họ tên thành công!")
            time.sleep(5)  # ⏳ Delay 5 giây để chờ giao diện load tiếp
            detect_event.set()  # Gọi cập nhật ảnh mới sau thao tác
            handled = True
            continue
        
        if all(k in js_predictions for k in ["form_muon_dung_tao_tai_khoan", "tiep_tuc_tao_tai_khoan", "dung_tao_tai_khoan", "comeback"]):
            tap_and_detect(js_predictions["dung_tao_tai_khoan"]["box"][0], js_predictions["dung_tao_tai_khoan"]["box"][1])
            print("❌ Không muốn tạo tài khoản, thoát!")
            time.sleep(5)  # ⏳ Delay 5 giây để chờ giao diện load tiếp
            detect_event.set()  # Gọi cập nhật ảnh mới sau thao tác
            handled = True
            continue

        if all(k in js_predictions for k in ["ngay_thang_nam_sinh", "input_ngay_thang_nam_sinh"]):
            if all(k in history for k in ["input_ngay_sinh", "input_thang_sinh", "input_nam_sinh", "set"]):
                if all(k in js_predictions for k in ["ngay_thang_nam_sinh", "input_ngay_thang_nam_sinh", "next"]):
                    tap_and_detect(js_predictions["next"]["box"][0], js_predictions["next"]["box"][1])
                    print("✅ Đã nhập ngày tháng năm sinh trước đó, chuyển sang trang khác!")
                    time.sleep(5)  # ⏳ Delay 5 giây để chờ giao diện load tiếp
                    history.clear()
                else:
                    print("Error: Không có nút next để chuyển trang!")
                    continue
                detect_event.set()  # Gọi cập nhật ảnh mới sau thao tác
                handled = True
                continue

            if all(k not in js_predictions for k in ["input_ngay_sinh", "input_thang_sinh", "input_nam_sinh", "set"]):
                tap_and_detect(js_predictions["input_ngay_thang_nam_sinh"]["box"][0], js_predictions["input_ngay_thang_nam_sinh"]["box"][1])
                print("✅ Nhấn vào input_ngay_thang_nam_sinh")
                time.sleep(1)
                detect_event.set()  # Gọi cập nhật ảnh mới sau thao tác
                handled = True
                continue

            if history is None or not isinstance(history, list):
                history = []

            if "input_ngay_sinh" not in history and "input_ngay_sinh" in js_predictions:
                long_press_and_detect(js_predictions["input_ngay_sinh"]["box"][0], js_predictions["input_ngay_sinh"]["box"][1], 3000)
                phone.delete_left(2)
                phone.input_text(ngay_sinh)
                history.append("input_ngay_sinh")
                time.sleep(3)
                detect_event.set()  # Gọi cập nhật ảnh mới sau thao tác
                handled = True
                continue

            if "input_thang_sinh" not in history and "input_thang_sinh" in js_predictions:
                long_press_and_detect(js_predictions["input_thang_sinh"]["box"][0], js_predictions["input_thang_sinh"]["box"][1], 3000)
                phone.delete_left(5)
                phone.input_text("thg " + thang_sinh)
                history.append("input_thang_sinh")
                time.sleep(3)
                detect_event.set()  # Gọi cập nhật ảnh mới sau thao tác
                handled = True
                continue

            if "input_nam_sinh" not in history and "input_nam_sinh" in js_predictions:
                long_press_and_detect(js_predictions["input_nam_sinh"]["box"][0], js_predictions["input_nam_sinh"]["box"][1], 3000)
                phone.delete_left(4)
                phone.input_text(nam_sinh)
                history.append("input_nam_sinh")
                time.sleep(3)
                detect_event.set()  # Gọi cập nhật ảnh mới sau thao tác
                handled = True
                continue

            if "set" in js_predictions:
                tap_and_detect(js_predictions["set"]["box"][0], js_predictions["set"]["box"][1])
                history.append("set")
                print("✅ Nhập ngày tháng năm sinh thành công!")
                time.sleep(3)  # ⏳ Delay 5 giây để chờ giao diện load tiếp
                detect_event.set()  # Gọi cập nhật ảnh mới sau thao tác
                handled = True
                continue
            
        if all(k in js_predictions for k in ["form_gioi_tinh", "form_nu", "form_nam"]):
            if gtinh == "Nam":
                tap_and_detect(js_predictions["form_nam"]["box"][0], js_predictions["form_nam"]["box"][1])
                print("✅ Chọn giới tính Nam thành công!")
            elif gtinh == "Nữ":
                tap_and_detect(js_predictions["form_nu"]["box"][0], js_predictions["form_nu"]["box"][1])
                print("✅ Chọn giới tính Nữ thành công!")
            else:
                print(f"❌ Giới tính '{gtinh}' không hợp lệ!")
                continue
            tap_and_detect(js_predictions["next"]["box"][0], js_predictions["next"]["box"][1])
            time.sleep(1)
            detect_event.set()  # Gọi cập nhật ảnh mới sau thao tác
            handled = True
            continue

        if all(k in js_predictions for k in ["input_so_di_dong", "redirect_email"]):
            if sdt_or_email == "sdt":
                tap_and_detect(js_predictions["input_so_di_dong"]["box"][0], js_predictions["input_so_di_dong"]["box"][1])
                phone.input_text(sdt_or_email, True)
                tap_and_detect(js_predictions["next"]["box"][0], js_predictions["next"]["box"][1])
                time.sleep(1)
                print("✅ Nhập số điện thoại thành công!")
                time.sleep(3)
                tap_and_detect(js_predictions["input_so_di_dong"]["box"][0], js_predictions["input_so_di_dong"]["box"][1] - js_predictions["input_so_di_dong"]["box"][3])
                phone.input_text(matkhau, True)
                tap_and_detect(js_predictions["input_so_di_dong"]["box"][0], js_predictions["input_so_di_dong"]["box"][1] + js_predictions["input_so_di_dong"]["box"][3])
            else:
                tap_and_detect(js_predictions["redirect_email"]["box"][0], js_predictions["redirect_email"]["box"][1])
                print("✅ Chuyển sang email thành công!")
            detect_event.set()  # Gọi cập nhật ảnh mới sau thao tác
            handled = True
            continue

        if all(k in js_predictions for k in ["input_email", "redirect_so_dien_thoai"]):
            if sdt_or_email == "email":
                # Lấy email tạm thời
                if not history or "input_email" not in history:
                    history.append("input_email")
                    global tempEmail
                    if (tempEmail == "") or (tempEmail is None): 
                        tempEmail = get_temp_email()
                    long_press_and_detect(js_predictions["input_email"]["box"][0], js_predictions["input_email"]["box"][1], 5000)
                    phone.input_text(tempEmail["email"], True)
                    detect_event.set()  # Gọi cập nhật ảnh mới sau thao tác
                    time.sleep(1)
                else:
                    tap_and_detect(js_predictions["next"]["box"][0], js_predictions["next"]["box"][1])
                    print("✅ Nhập email thành công!")
                    time.sleep(5)
                    tap_and_detect(js_predictions["input_email"]["box"][0], js_predictions["input_email"]["box"][1] - js_predictions["input_email"]["box"][3])
                    phone.input_text(matkhau, True)
                    tap_and_detect(js_predictions["input_email"]["box"][0], js_predictions["input_email"]["box"][1] + js_predictions["input_email"]["box"][3])
                    history.clear()
                    print("✅ Nhập mật khẩu thành công!")
            else:
                tap_and_detect(js_predictions["redirect_so_dien_thoai"]["box"][0], js_predictions["redirect_so_dien_thoai"]["box"][1])
                print("✅ Chuyển sang số điện thoại thành công!")
            detect_event.set()  # Gọi cập nhật ảnh mới sau thao tác
            handled = True
            continue

        if all(k in js_predictions for k in ["page_dong_y", "dong_y", "ban_da_co_tai_khoan"]):
            tap_and_detect(js_predictions["dong_y"]["box"][0], js_predictions["dong_y"]["box"][1])
            print("✅ Đồng ý các điều khoản thành công!")
            detect_event.set()  # Gọi cập nhật ảnh mới sau thao tác
            handled = True
            continue
            
        if all(k in js_predictions for k in ["nhap_ma_xac_nhan", "input_ma_xac_nhan", "next"]):
            print("🧠 Nhận diện màn hình nhập mã xác nhận")
            tap_and_detect(js_predictions["input_ma_xac_nhan"]["box"][0], js_predictions["input_ma_xac_nhan"]["box"][1])
            ma_xac_nhan = read_email_by_id(tempEmail)
            phone.delete_left(7)
            phone.input_text(ma_xac_nhan)
            tap_and_detect(js_predictions["next"]["box"][0], js_predictions["next"]["box"][1])
            time.sleep(1)
            print(f"✅ Đã nhập mã: {ma_xac_nhan} và nhấn Tiếp theo")
            detect_event.set()  # Gọi cập nhật ảnh mới sau thao tác
            handled = True
            continue

        if all(k in js_predictions for k in ["button_luu_thong_tin_dang_nhap", "luu_thong_tin_dang_nhap"]):
            tap_and_detect(js_predictions["button_luu_thong_tin_dang_nhap"]["box"][0], js_predictions["button_luu_thong_tin_dang_nhap"]["box"][1])
            print("✅ Đã lưu thông tin đăng nhập!")
            detect_event.set()  # Gọi cập nhật ảnh mới sau thao tác
            handled = True
            continue

        if all(k in js_predictions for k in ["tai_khoan_bi_khoa", "help"]):
            tap_and_detect(js_predictions["help"]["box"][0], js_predictions["help"]["box"][1])
            print("❗ Tài khoản bị khóa, cần hỗ trợ!")
            detect_event.set() 
            handled = True
            continue

        if all(k in js_predictions for k in ["page_dang_xuat", "dang_xuat_2"]):
            tap_and_detect(js_predictions["dang_xuat_2"]["box"][0], js_predictions["dang_xuat_2"]["box"][1])
            print("✅ Đã nhận diện trang đăng xuất!")
            detect_event.set() 
            handled = True
            continue
        
        if all(k in js_predictions for k in ["dang_xuat_1", "comeback"]):
                tap_and_detect(js_predictions["dang_xuat_1"]["box"][0], js_predictions["dang_xuat_1"]["box"][1])
                print("❗ Tài khoản bị khóa, cần hỗ trợ!")
                detect_event.set() 
                handled = True
                continue

        if all(k in js_predictions for k in ["login_by_google", "comeback"]):
            tap_and_detect(js_predictions["comeback"]["box"][0], js_predictions["comeback"]["box"][1])
            print("🔄 Nhấn nút comeback để quay lại!")
            detect_event.set()
            handled = True
            continue

        global thoat
        if not handled:
            thoat += 1
            if thoat >= 3:
                if all(k in js_predictions for k in ["thoat"]):
                    print("🛑 Đã nhận diện thao tác thoát!")
                    tap_and_detect(js_predictions["thoat"]["box"][0], js_predictions["thoat"]["box"][1])
                    handled = True
                    thoat = 0
                    time.sleep(1)
                    detect_event.set()  # Gọi cập nhật ảnh mới sau thao tác
                    continue
                if list(js_predictions.keys()) == ['comeback'] or all(k in js_predictions for k in ["page_dong_y", "page_dong_y"]):
                    print("🔄 Nhận diện thao tác comeback!")
                    tap_and_detect(js_predictions["comeback"]["box"][0], js_predictions["comeback"]["box"][1])
                    handled = True
                    thoat = 0
                    time.sleep(1)
                    detect_event.set()
                    continue
                else:
                    print("❌ Không nhận diện được thao tác nào!")
        else:
            thoat = 0

# ───────────────────────────── CHẠY LUỒNG CHÍNH ───────────────────────────── #
if __name__ == "__main__":
    t_display = threading.Thread(target=show_frame_thread, daemon=True)
    t_detect = threading.Thread(target=detect_and_show, daemon=True)

    t_display.start()
    t_detect.start()

    handle_actions()

    if os.path.exists("temp.png"):
        os.remove("temp.png")

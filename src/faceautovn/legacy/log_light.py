import os, re, requests
import subprocess
import time
import cv2
import numpy as np
import pytesseract
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Đường dẫn đến Tesseract (nếu cần, thay đổi tùy hệ điều hành)
# Ví dụ: pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def run_scrcpy_background(display=True):
    """Chạy scrcpy ở chế độ nền."""
    if display:
        process = subprocess.Popen(["scrcpy", "--stay-awake"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    else:
        process = subprocess.Popen(["scrcpy", "--no-display", "--stay-awake"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f"scrcpy chạy ngầm với PID: {process.pid}")
    return process

# Bảng chuyển đổi tiếng Việt có dấu sang kiểu Telex 
telex_map = {
    "à": "af", "á": "as", "ả": "ar", "ã": "ax", "ạ": "aj",
    "è": "ef", "é": "es", "ẻ": "er", "ẽ": "ex", "ẹ": "ej",
    "ì": "if", "í": "is", "ỉ": "ir", "ĩ": "ix", "ị": "ij",
    "ò": "of", "ó": "os", "ỏ": "or", "õ": "ox", "ọ": "oj",
    "ù": "uf", "ú": "us", "ủ": "ur", "ũ": "ux", "ụ": "uj",
    "ỳ": "yf", "ý": "ys", "ỷ": "yr", "ỹ": "yx", "ỵ": "yj",
    "â": "aa", "ầ": "aaf", "ấ": "aas", "ẩ": "aar", "ẫ": "aax", "ậ": "aaj",
    "ê": "ee", "ề": "eef", "ế": "ees", "ể": "eer", "ễ": "eex", "ệ": "eej",
    "ô": "oo", "ồ": "oof", "ố": "oos", "ổ": "oor", "ỗ": "oox", "ộ": "ooj",
    "ơ": "ow", "ờ": "owf", "ớ": "ows", "ở": "owr", "ỡ": "owx", "ợ": "owj",
    "ư": "uw", "ừ": "uwf", "ứ": "uws", "ử": "uwr", "ữ": "uwx", "ự": "uwj",
    "ă": "aw", "ằ": "awf", "ắ": "aws", "ẳ": "awr", "ẵ": "awx", "ặ": "awj",
    "đ": "dd",
    "À": "AF", "Á": "AS", "Ả": "AR", "Ã": "AX", "Ạ": "AJ",
    "È": "EF", "É": "ES", "Ẻ": "ER", "Ẽ": "EX", "Ẹ": "EJ",
    "Ì": "IF", "Í": "IS", "Ỉ": "IR", "Ĩ": "IX", "Ị": "IJ",
    "Ò": "OF", "Ó": "OS", "Ỏ": "OR", "Õ": "OX", "Ọ": "OJ",
    "Ù": "UF", "Ú": "US", "Ủ": "UR", "Ũ": "UX", "Ụ": "UJ",
    "Ỳ": "YF", "Ý": "YS", "Ỷ": "YR", "Ỹ": "YX", "Ỵ": "YJ",
    "Â": "AA", "Ầ": "AAF", "Ấ": "AAS", "Ẩ": "AAR", "Ẫ": "AAX", "Ậ": "AAJ",
    "Ê": "EE", "Ề": "EEF", "Ế": "EES", "Ể": "EER", "Ễ": "EEX", "Ệ": "EEJ",
    "Ô": "OO", "Ồ": "OOF", "Ố": "OOS", "Ổ": "OOR", "Ỗ": "OOX", "Ộ": "OOJ",
    "Ơ": "OW", "Ờ": "OWF", "Ớ": "OWS", "Ở": "OWR", "Ỡ": "OWX", "Ợ": "OWJ",
    "Ă": "AW", "Ằ": "AWF", "Ắ": "AWS", "Ẳ": "AWR", "Ẵ": "AWX", "Ặ": "AWJ",
    "Ư": "UW", "Ừ": "UWF", "Ứ": "UWS", "Ử": "UWR", "Ữ": "UWX", "Ự": "UWJ",
    "Đ": "DD"
}

def convert_to_telex(text):
    """Chuyển văn bản tiếng Việt có dấu sang kiểu Telex"""
    result = ""
    for char in text:
        if char in telex_map:
            result += telex_map[char]
        else:
            result += char
    return result

def clear_input():
    """Xóa nội dung trong ô nhập liệu tại tọa độ (x, y)"""
    try:
        # Di chuyển con trỏ về cuối dòng
        subprocess.run(["adb", "shell", "input", "keyevent", "123"], check=True)  # KEYCODE_MOVE_END
        
        # Xóa toàn bộ nội dung (giả sử tối đa 50 ký tự)
        for _ in range(50):
            subprocess.run(["adb", "shell", "input", "keyevent", "67"], check=True)  # KEYCODE_DEL
        print(f"✅ Đã xóa nội dung.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi xóa input: {e}")

def get_available_keyboards():
    """Lấy danh sách các bàn phím khả dụng bằng ADB."""
    try:
        result = subprocess.run(
            ["adb", "shell", "ime", "list", "-a"],
            check=True,
            capture_output=True,
            text=True
        )
        keyboards = re.findall(r'mId=([^\s]+)', result.stdout)
        print(f"🔍 Danh sách bàn phím khả dụng: {keyboards}")
        return keyboards
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi liệt kê bàn phím: {e.stderr}")
        return []

def set_keyboard(keyboard_id):
    """Đặt phương thức nhập (bàn phím) bằng ADB."""
    global CURRENT_KEYBOARD
    try:
        subprocess.run(
            ["adb", "shell", "ime", "set", keyboard_id],
            check=True,
            capture_output=True,
            text=True
        )
        CURRENT_KEYBOARD = keyboard_id  # Cập nhật bàn phím hiện tại
        print(f"✅ Đã chuyển sang bàn phím: {keyboard_id}")
        time.sleep(1)  # Thời gian chờ để bàn phím được áp dụng
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi chuyển bàn phím: {e.stderr}")
        return False

# Khởi tạo danh sách bàn phím và biến flag
AVAILABLE_KEYBOARDS = get_available_keyboards()
AVOID_KEYBOARD = "com.google.android.tts/com.google.android.apps.speech.tts.googletts.settings.asr.voiceime.VoiceInputMethodService"
AVAILABLE_KEYBOARDS = [kb for kb in AVAILABLE_KEYBOARDS if kb != AVOID_KEYBOARD]
CURRENT_KEYBOARD = None  # Ban đầu chưa có bàn phím nào được chọn

def input_text(text, vi=False):
    """Nhập văn bản qua ADB, tự động chọn bàn phím phù hợp."""
    try:
        # Xóa input hiện tại trước khi nhập
        clear_input()

        # Kiểm tra danh sách bàn phím toàn cục
        if not AVAILABLE_KEYBOARDS:
            print("❌ Không có bàn phím nào khả dụng.")
            return False

        # Xác định bàn phím mong muốn
        preferred_keyboard = "com.google.android.inputmethod.latin/com.android.inputmethod.latin.LatinIME" if vi else next(
            (kb for kb in AVAILABLE_KEYBOARDS if kb != "com.google.android.inputmethod.latin/com.android.inputmethod.latin.LatinIME"), 
            AVAILABLE_KEYBOARDS[0]
        )

        # Chỉ chuyển bàn phím nếu chưa đúng
        if CURRENT_KEYBOARD != preferred_keyboard:
            if preferred_keyboard in AVAILABLE_KEYBOARDS:
                if not set_keyboard(preferred_keyboard):
                    print(f"❌ Không thể chuyển sang bàn phím: {preferred_keyboard}")
                    return False
            else:
                print(f"❌ Bàn phím mong muốn ({preferred_keyboard}) không có trong danh sách: {AVAILABLE_KEYBOARDS}")
                return False

        # Chuyển đổi văn bản nếu cần
        input_text_converted = convert_to_telex(text) if vi else text
        print(f"🔹 Nhập văn bản: {text} (Telex: {input_text_converted})")

        # Thay thế khoảng trắng bằng %s cho ADB
        adb_text = input_text_converted.replace(" ", "%s")

        # Gửi lệnh ADB
        subprocess.run(
            ["adb", "shell", "input", "text", adb_text],
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ Gửi thành công")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = f"❌ Lỗi khi gửi: {e}"
        if e.stderr:
            error_msg += f"\nChi tiết: {e.stderr}"
        print(error_msg)
        return False
    except Exception as e:
        print(f"❌ Lỗi không xác định: {e}")
        return False

def capture_screen(filename=None):
    """Chụp ảnh màn hình từ thiết bị Android."""
    subprocess.run(["adb", "shell", "screencap", "-p", "/sdcard/screen.png"], check=True)
    subprocess.run(["adb", "pull", "/sdcard/screen.png", filename], check=True)
    return filename

def get_screen_size():
    """Lấy kích thước màn hình thiết bị."""
    result = subprocess.run(["adb", "shell", "wm", "size"], capture_output=True, text=True)
    output = result.stdout.strip()
    if "Physical size:" in output:
        size = output.split(": ")[-1]
        width, height = map(int, size.split("x"))
        return width, height
    return 1080, 1920  # Giá trị mặc định

def tap(x, y):
    """Chạm vào màn hình tại vị trí (x, y)."""
    print(f"🔹 Chạm vào ({x}, {y})")
    subprocess.run(["adb", "shell", "input", "tap", str(x), str(y)], check=True)

def swipe_down(width, height, times=1):
    """Vuốt màn hình lên `times` lần để tải nội dung mới."""
    for _ in range(times):
        print("🔄 Vuốt màn hình lên")
        subprocess.run(["adb", "shell", "input", "swipe", str(width // 2), str(height - 300),
                        str(width // 2), "300", "500"], check=True)
        time.sleep(1.5)

def swipe_up(width, height, times=1):
    """Vuốt màn hình xuống `times` lần."""
    for _ in range(times):
        print("🔄 Vuốt màn hình xuống")
        subprocess.run(["adb", "shell", "input", "swipe", str(width // 2), "300",
                        str(width // 2), str(height - 300), "500"], check=True)
        time.sleep(1.5)
        
def find_button(image_name, threshold=0.7):
    screen_path = capture_screen()
    screen = cv2.imread(screen_path, cv2.IMREAD_GRAYSCALE)

    if screen is None:
        print("❌ Lỗi khi đọc ảnh màn hình.")
        return None

    template_path = str(ASSETS_DIR / "templates" / "img_light" / f"{image_name}.png")
    if not os.path.exists(template_path):
        print(f"⚠️ Không tìm thấy mẫu ảnh: {template_path}")
        return None

    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"⚠️ Lỗi khi đọc ảnh mẫu: {template_path}")
        return None
    # Lấy kích thước của screen
    screen_height, screen_width = screen.shape
    best_match = None

    best_value = 0

    for scale in [1.0, 0.9, 0.8, 0.7]:  
        resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
        templ_height, templ_width = resized_template.shape
        
        # Kiểm tra kích thước trước khi gọi matchTemplate
        if templ_height > screen_height or templ_width > screen_width:
            print(f"⚠️ Kích thước mẫu ({templ_width}x{templ_height}) lớn hơn màn hình ({screen_width}x{screen_height}), bỏ qua scale {scale}")
            continue    
        result = cv2.matchTemplate(screen, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_value:
            best_value = max_val
            best_match = (max_loc, resized_template.shape[::-1])  # (x, y) và (w, h)

    if best_match and best_value >= threshold:
        x, y = best_match[0]
        w, h = best_match[1]

        if image_name == "home":
            screen_width, _ = get_screen_size()
            x = screen_width - w - 10  # Dịch về góc phải

        return x + w // 2, y + h // 2  # Trả về tọa độ trung tâm nút

    return None

def find_text(text_to_find="Đăng xuất"):
    """Tìm văn bản trên màn hình bằng OCR và in debug."""
    screen_path = capture_screen()
    screen = cv2.imread(screen_path)

    if screen is None:
        print("❌ Lỗi khi đọc ảnh màn hình.")
        return None

    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Cấu hình nhận diện
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(gray, lang='vie', config=custom_config, output_type=pytesseract.Output.DICT)

    print("📋 Văn bản nhận diện được:")
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

        if text:
            print(f"🔹 '{text}' tại ({x}, {y}) - Kích thước: {w}x{h}")

        if text_to_find.lower() in text.lower():
            print(f"✅ Tìm thấy '{text_to_find}' tại ({x}, {y})")
            return x + w // 2, y + h // 2

    print(f"❌ Không tìm thấy '{text_to_find}' trên màn hình.")
    return None

def control_app(package_name, action="start", activity=".MainActivity"):
    """
    Function để mở hoặc thoát ứng dụng bằng ADB.
    - package_name: Tên gói của ứng dụng (ví dụ: 'com.facebook.katana' cho Facebook)
    - action: 'start' để mở, 'stop' để thoát
    - activity: Tên activity để mở ứng dụng (mặc định là '.MainActivity')
    """
    if action == "start":
        # Lệnh ADB để mở ứng dụng với activity cụ thể
        command = f"adb shell am start -n {package_name}/{package_name}{activity}"
        print(f"Đang mở ứng dụng {package_name} với {activity}...")
        time.sleep(0.5)
    elif action == "stop":
        # Lệnh ADB để thoát ứng dụng
        command = f"adb shell am force-stop {package_name}"
        print(f"Đang thoát ứng dụng {package_name}...")
    else:
        print("Hành động không hợp lệ! Sử dụng 'start' hoặc 'stop'.")
        return
    
    # Thực thi lệnh ADB
    try:
        os.system(command)
        print("Thành công!")
    except Exception as e:
        print(f"Lỗi: {e}")

def read_mail(mail):
    headers = {
        'accept': '*/*',
        'accept-language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
        'application-name': 'web',
        'application-version': '4.0.0',
        'content-type': 'application/json',
        'origin': 'https://temp-mail.io',
        'priority': 'u=1, i',
        'referer': 'https://temp-mail.io/',
        'sec-ch-ua': '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
        'x-cors-header': 'iaWg3pchvFx48fY',
    }

    response = requests.get(f"https://api.internal.temp-mail.io/api/v3/email/{mail}/messages", headers=headers)

    # Kiểm tra mã trạng thái
    if response.status_code == 200:
        # Chuyển đổi JSON thành đối tượng Python
        data = response.json()
        # print(json.dumps(data, indent=4, ensure_ascii=False))
        return data
    else:
        print(f"Yêu cầu thất bại với mã trạng thái: {response.status_code}")
        print(response.text)

def get_mail(min_length=10, max_length=10):
    headers = {
        'accept': '*/*',
        'accept-language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
        'application-name': 'web',
        'application-version': '4.0.0',
        'content-type': 'application/json',
        'origin': 'https://temp-mail.io',
        'priority': 'u=1, i',
        'referer': 'https://temp-mail.io/',
        'sec-ch-ua': '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
        'x-cors-header': 'iaWg3pchvFx48fY',
    }

    json_data = {
        'min_name_length': min_length,
        'max_name_length': max_length,
    }

    response = requests.post('https://api.internal.temp-mail.io/api/v3/email/new', headers=headers, json=json_data)
    if response.status_code == 200 or response.status_code == 201:
        return response.json()['email']
    else:
        print(f"Yêu cầu thất bại với mã trạng thái: {response.status_code}")
        return "False"
    
def get_otp(mail=""):
    solan = 0
    while(True):
        if(solan>=150): break
        boxchat = read_mail(mail)
        for i in range(0,len(boxchat),1):
            if boxchat[i]["from"] == '"Facebook" <registration@facebookmail.com>':
                # print(boxchat[i]["from"])
                if " là mã xác nhận" in boxchat[i]["subject"]:
                    mxn = boxchat[i]["subject"].split(" là mã xác nhận")[0]
                    if mxn:
                        return mxn 
        solan+=1
        time.sleep(0.5)

def input_text_vietnamese(text):
    """Nhập văn bản tiếng Việt bằng cách copy-paste từ clipboard."""
    print(f"⌨️ Nhập: {text}")
    
    # Đặt nội dung vào clipboard
    subprocess.run(["adb", "shell", "am", "broadcast", "-a", "clipper.set", "--es", "text", text], check=True)

    time.sleep(1)  # Chờ một chút để clipboard cập nhật
    
    # Dán nội dung bằng Ctrl+V (KEYCODE_PASTE)
    subprocess.run(["adb", "shell", "input", "keyevent", "279"], check=True)

def auto_register(now=None, firstname="Tài", lastname="Phan Anh", day="31", month="12", year="2003", 
                  sex="male", phone="", email = "False", password="Khoatran2006"):
    width, height = get_screen_size()
    while(email=="False"):
        email = get_mail()
        print(email)
    # Khởi động ứng dụng
    control_app("com.facebook.katana", "stop", activity=".LoginActivity")
    control_app("com.facebook.katana", "start", activity=".LoginActivity")
    time.sleep(5)

    # Xác định trạng thái ban đầu
    if not now:
        while True:
            if find_and_tap("home"):
                now = "home"
                break
            elif find_and_tap("register1"):
                now = "register1"
                break
            else:
                control_app("com.facebook.katana", "stop", activity=".LoginActivity")
                control_app("com.facebook.katana", "start", activity=".LoginActivity")
                time.sleep(5)

    # Đăng xuất nếu ở trang home
    if now == "home":
        find_and_tap("logout1")
        find_and_tap("logout2")
        find_and_tap("register1")

    find_and_tap("register2")

    # Điền form đăng ký
    input_field("register_lastname", lastname, vi=True, max_attempts=3)
    input_field("register_firstname", firstname, vi=True)
    find_and_tap("next")
    
    # Nhập ngày sinh
    input_field("birthday_year", year)
    input_field("birthday_month", f"Th{month}")
    input_field("birthday_day", day)
    find_and_tap("birthday_set")
    find_and_tap("next")
    
    # Chọn giới tính
    find_and_tap("male" if sex == "male" else "female")
    find_and_tap("next")
    
    # Nhập số điện thoại hoặc email
    if phone:
        input_field("phone", phone)
    else:
        find_and_tap("convert_email")
        input_field("email", email)
    find_and_tap("next")
    
    # Nhập mật khẩu
    input_field("password", password)
    find_and_tap("next")
    find_and_tap("next")  # Nhấn "Tiếp" lần nữa nếu cần
    find_and_tap("save")
    find_and_tap("dongy")
    
    print(f"Đang lấy mã từ mail: {email}")
    # Xác nhận mã
    while True:
        ma = get_otp(email)
        print(f"Mã xác nhận là: {ma}")
        input_field("maxacnhan", ma)
        if find_and_tap("next"):
            break  # Thoát vòng lặp nếu mã đúng (giả sử nhấn "Tiếp" thành công)

    # Chụp màn hình và kết thúc
    time.sleep(10)
    capture_screen()
    print(f"✅ Đã tạo thành công email: {email}, Password: {password}")

# Hàm phụ trợ (đã định nghĩa trước đó)
def find_and_tap(image_name, threshold=0.8, max_attempts=5):
    for _ in range(max_attempts):
        position = find_button(image_name, threshold)
        if position:
            tap(*position)
            print(f"✅ Đã nhấn vào '{image_name}' tại {position}")
            time.sleep(0.5)
            return True
        else:
            print(f"🔍 Không tìm thấy '{image_name}', vuốt xuống...")
            width, height = get_screen_size()
            swipe_down(width, height, times=1)
            time.sleep(1)
    print(f"❌ Không thể tìm thấy '{image_name}' sau {max_attempts} lần thử.")
    return False

def input_field(image_name, text, vi=False, threshold=0.8, max_attempts=5):
    for _ in range(max_attempts):
        position = find_button(image_name, threshold)
        if position:
            tap(*position)
            print(f"📝 Đã nhấn vào trường '{image_name}'")
            time.sleep(0.2)
            input_text(text, vi)
            return True
        else:
            width, height = get_screen_size()
            if image_name in ["birthday_year", "birthday_month", "birthday_day"]:
                tap(width//2,height//2)
                print(f"🔍 Không tìm thấy '{image_name}'")
            else:
                print(f"🔍 Không tìm thấy '{image_name}', vuốt xuống...")
                swipe_down(width, height, times=1)
                time.sleep(1)
    print(f"❌ Không thể tìm thấy '{image_name}' sau {max_attempts} lần thử.")
    return False

if __name__ == "__main__":# Chạy scrcpy ngầm
    scrcpy_process = run_scrcpy_background()
    auto_register()
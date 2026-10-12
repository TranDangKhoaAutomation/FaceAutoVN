import cv2
import numpy as np
import subprocess
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def capture_screen(filename=None):
    """Chụp ảnh màn hình từ thiết bị Android."""
    subprocess.run(["adb", "shell", "screencap", "-p", "/sdcard/screen.png"], check=True)
    subprocess.run(["adb", "pull", "/sdcard/screen.png", filename], check=True)
    return filename

def preprocess_image(image):
    """Tiền xử lý ảnh để tăng độ chính xác."""
    # Làm mịn để giảm nhiễu
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # Tăng độ tương phản nhẹ
    image = cv2.convertScaleAbs(image, alpha=1.2, beta=0)
    return image

def find_button_in_image(template_path, image_path, threshold=0.8, scales=[2.0, 1.5, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]):
    """Tìm nút trong ảnh với multi-scale, kiểm tra viền xanh, hiển thị chấm xanh lá ở giữa."""
    # Đọc ảnh ở chế độ màu
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if template is None or image is None:
        return False, 0, None
    
    # Tiền xử lý ảnh
    template = preprocess_image(template)
    image_processed = preprocess_image(image)
    
    # Tách kênh màu và tìm trên từng kênh
    best_val = 0
    best_loc = None
    best_scale = None
    
    for scale in scales:
        scaled_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = scaled_template.shape[:2]
        
        if h > image_processed.shape[0] or w > image_processed.shape[1]:
            continue
        
        # Tìm trên từng kênh màu (B, G, R)
        for channel in range(3):  # BGR
            template_channel = scaled_template[:, :, channel]
            image_channel = image_processed[:, :, channel]
            result = cv2.matchTemplate(image_channel, template_channel, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_scale = scale
    
    if best_val >= threshold:
        # Tính tọa độ dựa trên scale tốt nhất
        h, w = template.shape[:2]
        scaled_w, scaled_h = int(w * best_scale), int(h * best_scale)
        top_left = best_loc
        match_percentage = best_val * 100
        
        # Kiểm tra vùng khớp có viền xanh không (đặc trưng của nút "Tạo tài khoản mới")
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        
        # Cắt vùng khớp từ mask
        x, y = top_left
        region = mask[y:y+scaled_h, x:x+scaled_w]
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        has_blue_border = False
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Đảm bảo contour đủ lớn
                has_blue_border = True
                break
        
        if not has_blue_border:
            print("Vùng khớp không có viền xanh, có thể là false positive.")
            return False, match_percentage, None
        
        # Vẽ hình chữ nhật lên ảnh gốc
        image_with_rectangle = image.copy()
        bottom_right = (top_left[0] + scaled_w, top_left[1] + scaled_h)
        cv2.rectangle(image_with_rectangle, top_left, bottom_right, (0, 255, 0), 2)
        
        # Tính trung tâm của vùng khớp và vẽ chấm xanh lá
        center_x = top_left[0] + scaled_w // 2
        center_y = top_left[1] + scaled_h // 2
        center = (center_x, center_y)
        cv2.circle(image_with_rectangle, center, 5, (0, 255, 0), -1)  # Chấm xanh lá, bán kính 5
        
        # Thu nhỏ ảnh để hiển thị (giảm còn 30% kích thước)
        scale_percent = 30
        width = int(image_with_rectangle.shape[1] * scale_percent / 100)
        height = int(image_with_rectangle.shape[0] * scale_percent / 100)
        resized_image = cv2.resize(image_with_rectangle, (width, height), interpolation=cv2.INTER_AREA)
        
        # Hiển thị ảnh đã thu nhỏ
        cv2.imshow('Result', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Lưu ảnh kết quả để debug
        cv2.imwrite(str(OUTPUT_DIR / "debug_result.png"), image_with_rectangle)
        
        return True, match_percentage, top_left
    return False, best_val * 100, None

# Test
if __name__ == "__main__":
    if(1==21):
        capture_screen()
    else:
        template_path = str(ASSETS_DIR / "templates" / "img_light" / "register1.png")
        image_path = str(OUTPUT_DIR / "screen.png")
        found, percentage, position = find_button_in_image(template_path, image_path, threshold=0.9)
        if found:
            print(f"Tìm thấy với độ khớp: {percentage:.2f}% tại {position}")
        else:
            print(f"Không tìm thấy, độ khớp cao nhất: {percentage:.2f}%")
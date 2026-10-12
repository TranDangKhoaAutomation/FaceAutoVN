import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent))
import adb  # hoặc từ adb import ADBController nếu bạn tách file

device = adb.ADBController(debug=True)

for i in range(1000):
    if input("Nhấn Enter để chụp (hoặc nhập gì đó để dừng): ") == "":
        filename = str(APP_DIR.parents[2] / "outputs" / f"train_{i}.png")
        device.capture_screenshot(filename)
        print(f"✅ Đã lưu: {filename}")
    else:
        print("⛔ Dừng vòng lặp")
        break

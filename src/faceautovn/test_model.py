# -*- coding: utf-8 -*-
"""Test nhanh model YOLO local `data/best.pt`.

Ví dụ:
    python src/faceautovn/test_model.py
    python src/faceautovn/test_model.py --model data/best.pt --image assets/images/imgkiemtra.png --conf 0.2

Script sẽ:
- kiểm tra file model tồn tại
- load YOLO bằng ultralytics
- chạy predict trên ảnh test, mặc định `assets/images/imgkiemtra.png`
- in class/conf/box ra console
- lưu ảnh đã vẽ box vào `outputs/test_model_output.jpg` (hoặc --output)
"""

import argparse
import os
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    app_dir = Path(__file__).resolve().parent
    project_root = app_dir.parents[1]
    parser = argparse.ArgumentParser(description="Test local YOLO best.pt for FaceAutoVN")
    parser.add_argument("--model", default=str(project_root / "data" / "best.pt"), help="Đường dẫn model .pt")
    parser.add_argument("--image", default=str(project_root / "assets" / "images" / "imgkiemtra.png"), help="Ảnh dùng để test")
    parser.add_argument("--conf", type=float, default=0.2, help="Ngưỡng confidence")
    parser.add_argument("--output", default=str(project_root / "outputs" / "test_model_output.jpg"), help="File ảnh output đã vẽ box")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Không thấy model: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Không thấy ảnh test: {image_path}")

    print(f"✅ Load model: {model_path}")
    model = YOLO(str(model_path))

    print(f"✅ Predict image: {image_path}")
    results = model.predict(str(image_path), conf=args.conf, verbose=False)
    result = results[0]

    names = result.names or model.names or {}
    boxes = result.boxes or []
    print(f"✅ Số object phát hiện: {len(boxes)}")

    for i, box in enumerate(boxes, start=1):
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        print(f"{i:02d}. {cls_name} conf={conf:.3f} xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

    annotated = result.plot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    print(f"✅ Đã lưu ảnh kết quả: {output_path}")


if __name__ == "__main__":
    # Giảm log/telemetry không cần thiết khi chạy test local.
    os.environ.setdefault("YOLO_VERBOSE", "False")
    main()

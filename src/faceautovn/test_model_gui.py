# -*- coding: utf-8 -*-
"""FaceAutoVN - Qt6 GUI test model best.pt.

Chạy giao diện:
    python src/faceautovn/test_model_gui.py

Tự kiểm thử/click nút bằng QtTest:
    python src/faceautovn/test_model_gui.py --self-test

Tính năng:
- Giao diện Qt6/PySide6 dark hiện đại
- Chọn model .pt, mặc định data/best.pt
- Chọn ảnh test, mặc định assets/images/imgkiemtra.png
- Chỉnh confidence
- Detect nền bằng thread để UI không đơ
- Xem ảnh đã vẽ box, bảng kết quả, log
- Lưu output ảnh
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

import cv2
from PIL import Image
from ultralytics import YOLO

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODEL = DATA_DIR / "best.pt"
DEFAULT_IMAGE = ASSETS_DIR / "images" / "imgkiemtra.png"
DEFAULT_OUTPUT = OUTPUT_DIR / "test_model_gui_output.jpg"


APP_QSS = """
QMainWindow {
    background: #0b1020;
}
QWidget {
    color: #e5e7eb;
    font-family: Segoe UI, Arial;
    font-size: 13px;
}
QFrame#Card {
    background: #111827;
    border: 1px solid #243047;
    border-radius: 16px;
}
QFrame#Hero {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #172554, stop:0.55 #111827, stop:1 #052e2b);
    border: 1px solid #2f3c59;
    border-radius: 20px;
}
QLabel#Title {
    font-size: 25px;
    font-weight: 800;
    color: #f8fafc;
}
QLabel#Subtitle {
    color: #93c5fd;
    font-size: 13px;
}
QLabel#SectionTitle {
    font-weight: 800;
    font-size: 14px;
    color: #c7d2fe;
}
QLabel#Preview {
    background: #050816;
    border: 1px solid #26334d;
    border-radius: 14px;
    color: #94a3b8;
}
QLineEdit, QDoubleSpinBox {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 8px 10px;
    color: #f8fafc;
    selection-background-color: #2563eb;
}
QLineEdit:focus, QDoubleSpinBox:focus {
    border: 1px solid #60a5fa;
}
QPushButton {
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 10px;
    padding: 9px 13px;
    font-weight: 700;
}
QPushButton:hover {
    background: #293548;
    border-color: #60a5fa;
}
QPushButton:pressed {
    background: #111827;
}
QPushButton#PrimaryButton {
    background: #2563eb;
    border: 1px solid #3b82f6;
    color: white;
}
QPushButton#PrimaryButton:hover {
    background: #1d4ed8;
}
QPushButton#SuccessButton {
    background: #047857;
    border: 1px solid #10b981;
    color: white;
}
QPushButton#SuccessButton:hover {
    background: #065f46;
}
QPushButton#DangerButton {
    background: #7f1d1d;
    border: 1px solid #ef4444;
    color: white;
}
QTableWidget {
    background: #0f172a;
    border: 1px solid #26334d;
    border-radius: 12px;
    gridline-color: #26334d;
    selection-background-color: #1d4ed8;
    alternate-background-color: #111c33;
}
QHeaderView::section {
    background: #1e293b;
    color: #c7d2fe;
    border: 0;
    border-right: 1px solid #334155;
    padding: 8px;
    font-weight: 800;
}
QTextEdit {
    background: #060b18;
    border: 1px solid #26334d;
    border-radius: 12px;
    padding: 8px;
    color: #bfdbfe;
}
QSplitter::handle {
    background: #1e293b;
}
"""


class DetectWorker(QObject):
    finished = Signal(object)
    failed = Signal(str, str)
    progress = Signal(str)

    def __init__(self, model_path, image_path, conf, loaded_model=None, loaded_path=None):
        super().__init__()
        self.model_path = str(model_path)
        self.image_path = str(image_path)
        self.conf = float(conf)
        self.loaded_model = loaded_model
        self.loaded_path = loaded_path

    def run(self):
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Không thấy model: {self.model_path}")
            if not Path(self.image_path).exists():
                raise FileNotFoundError(f"Không thấy ảnh test: {self.image_path}")

            model = self.loaded_model
            if model is None or self.loaded_path != self.model_path:
                self.progress.emit("Đang load model YOLO...")
                model = YOLO(self.model_path)
            else:
                self.progress.emit("Dùng lại model đã load...")

            self.progress.emit("Đang chạy detect...")
            results = model.predict(self.image_path, conf=self.conf, verbose=False)
            result = results[0]
            boxes = result.boxes or []
            names = result.names or model.names or {}
            annotated_bgr = result.plot()

            rows = []
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                score = float(box.conf[0].item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                rows.append(
                    {
                        "class": cls_name,
                        "conf": score,
                        "box": (x1, y1, x2, y2),
                    }
                )

            self.finished.emit(
                {
                    "model": model,
                    "model_path": self.model_path,
                    "annotated_bgr": annotated_bgr,
                    "rows": rows,
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())


class FaceAutoVNModelTester(QMainWindow):
    def __init__(self, self_test=False):
        super().__init__()
        self.self_test = self_test
        self.model = None
        self.model_loaded_path = None
        self.annotated_bgr = None
        self.original_bgr = None
        self.worker_thread = None
        self.worker = None
        self.detect_done_for_self_test = False

        self.setWindowTitle("FaceAutoVN • Qt6 Model Tester")
        self.resize(1280, 820)
        self.setMinimumSize(1040, 680)
        self._build_ui()
        self._load_preview(DEFAULT_IMAGE)

        if self.self_test:
            QTimer.singleShot(800, self._run_self_test_clicks)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        hero = QFrame(objectName="Hero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(22, 18, 22, 18)
        title = QLabel("FaceAutoVN Model Tester")
        title.setObjectName("Title")
        subtitle = QLabel("Qt6 GUI • test data/best.pt • preview boxes • save output")
        subtitle.setObjectName("Subtitle")
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        root.addWidget(hero)

        config_card = self._card()
        grid = QGridLayout(config_card)
        grid.setContentsMargins(18, 16, 18, 16)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.model_edit = QLineEdit(str(DEFAULT_MODEL))
        self.image_edit = QLineEdit(str(DEFAULT_IMAGE))
        self.output_edit = QLineEdit(str(DEFAULT_OUTPUT))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(0.20)
        self.conf_spin.setDecimals(2)

        self._add_path_row(grid, 0, "Model .pt", self.model_edit, self.choose_model)
        self._add_path_row(grid, 1, "Ảnh test", self.image_edit, self.choose_image)
        self._add_path_row(grid, 2, "Output", self.output_edit, self.choose_output)

        grid.addWidget(QLabel("Confidence"), 3, 0)
        grid.addWidget(self.conf_spin, 3, 1)

        actions = QHBoxLayout()
        self.load_btn = QPushButton("Load model")
        self.detect_btn = QPushButton("Chạy detect")
        self.detect_btn.setObjectName("PrimaryButton")
        self.save_btn = QPushButton("Lưu output")
        self.save_btn.setObjectName("SuccessButton")
        self.reset_btn = QPushButton("Reset mặc định")
        self.open_btn = QPushButton("Mở thư mục")
        actions.addWidget(self.load_btn)
        actions.addWidget(self.detect_btn)
        actions.addWidget(self.save_btn)
        actions.addWidget(self.reset_btn)
        actions.addStretch(1)
        actions.addWidget(self.open_btn)
        grid.addLayout(actions, 3, 2, 1, 2)

        self.load_btn.clicked.connect(self.load_model_clicked)
        self.detect_btn.clicked.connect(self.detect_clicked)
        self.save_btn.clicked.connect(self.save_output)
        self.reset_btn.clicked.connect(self.reset_defaults)
        self.open_btn.clicked.connect(self.open_output_folder)
        root.addWidget(config_card)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        preview_card = self._card()
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(14, 14, 14, 14)
        section = QLabel("Preview")
        section.setObjectName("SectionTitle")
        self.preview_label = QLabel("Chọn ảnh test rồi bấm Chạy detect")
        self.preview_label.setObjectName("Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setMinimumSize(520, 420)
        preview_layout.addWidget(section)
        preview_layout.addWidget(self.preview_label, 1)
        splitter.addWidget(preview_card)

        result_card = self._card()
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(14, 14, 14, 14)
        result_title = QLabel("Kết quả")
        result_title.setObjectName("SectionTitle")
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Class", "Conf", "Box xyxy"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(135)
        result_layout.addWidget(result_title)
        result_layout.addWidget(self.table, 1)
        result_layout.addWidget(QLabel("Log"))
        result_layout.addWidget(self.log_box)
        splitter.addWidget(result_card)
        splitter.setSizes([780, 420])

        self.status = QLabel("Sẵn sàng")
        self.status.setStyleSheet("color: #93c5fd; padding-left: 4px;")
        root.addWidget(self.status)

    def _card(self):
        card = QFrame()
        card.setObjectName("Card")
        return card

    def _add_path_row(self, grid, row, label_text, edit, choose_fn):
        label = QLabel(label_text)
        button = QPushButton("Chọn...")
        button.clicked.connect(choose_fn)
        grid.addWidget(label, row, 0)
        grid.addWidget(edit, row, 1, 1, 2)
        grid.addWidget(button, row, 3)

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn model .pt", str(DATA_DIR), "PyTorch model (*.pt);;All files (*.*)")
        if path:
            self.model_edit.setText(path)
            self.model = None
            self.model_loaded_path = None
            self.log(f"Đã chọn model: {path}")

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh test", str(ASSETS_DIR / "images"), "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*.*)")
        if path:
            self.image_edit.setText(path)
            self._load_preview(Path(path))

    def choose_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh output", str(DEFAULT_OUTPUT), "JPEG (*.jpg);;PNG (*.png);;All files (*.*)")
        if path:
            self.output_edit.setText(path)

    def reset_defaults(self):
        self.model_edit.setText(str(DEFAULT_MODEL))
        self.image_edit.setText(str(DEFAULT_IMAGE))
        self.output_edit.setText(str(DEFAULT_OUTPUT))
        self.conf_spin.setValue(0.20)
        self._load_preview(DEFAULT_IMAGE)
        self.log("Đã reset về mặc định")

    def _load_preview(self, image_path):
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            self.preview_label.setText(f"Không đọc được ảnh:\n{image_path}")
            return
        self.original_bgr = bgr
        self.annotated_bgr = None
        self._show_bgr(bgr)
        self.clear_table()
        self.log(f"Đã load ảnh: {image_path}")

    def _show_bgr(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        target = self.preview_label.size()
        if target.width() > 50 and target.height() > 50:
            pix = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.annotated_bgr is not None:
            self._show_bgr(self.annotated_bgr)
        elif self.original_bgr is not None:
            self._show_bgr(self.original_bgr)

    def log(self, message):
        self.status.setText(message)
        self.log_box.append(message)

    def set_busy(self, busy):
        for btn in [self.load_btn, self.detect_btn, self.save_btn, self.reset_btn, self.open_btn]:
            btn.setEnabled(not busy)
        QApplication.setOverrideCursor(Qt.WaitCursor) if busy else QApplication.restoreOverrideCursor()

    def load_model_clicked(self):
        self.detect_only_load = True
        self._start_worker(load_only=True)

    def detect_clicked(self):
        self.detect_only_load = False
        self._start_worker(load_only=False)

    def _start_worker(self, load_only=False):
        model_path = self.model_edit.text().strip()
        image_path = self.image_edit.text().strip()
        conf = self.conf_spin.value()
        if load_only:
            # Worker vẫn chạy predict nhẹ hơn không đáng kể? Không: load riêng để phản hồi nhanh.
            try:
                if not Path(model_path).exists():
                    raise FileNotFoundError(f"Không thấy model: {model_path}")
                self.set_busy(True)
                self.log("Đang load model YOLO...")
                QApplication.processEvents()
                self.model = YOLO(model_path)
                self.model_loaded_path = model_path
                self.log(f"✅ Load model xong: {model_path}")
            except Exception as exc:
                self.show_error(str(exc), traceback.format_exc())
            finally:
                self.set_busy(False)
            return

        self.set_busy(True)
        self.worker_thread = QThread(self)
        self.worker = DetectWorker(model_path, image_path, conf, self.model, self.model_loaded_path)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.log)
        self.worker.finished.connect(self.on_detect_finished)
        self.worker.failed.connect(self.on_detect_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def on_detect_finished(self, payload):
        self.model = payload["model"]
        self.model_loaded_path = payload["model_path"]
        self.annotated_bgr = payload["annotated_bgr"]
        rows = payload["rows"]
        self.populate_table(rows)
        self._show_bgr(self.annotated_bgr)
        self.log(f"✅ Detect xong: {len(rows)} object")
        self.set_busy(False)
        if self.self_test:
            self.detect_done_for_self_test = True
            QTimer.singleShot(300, self._finish_self_test)

    def on_detect_failed(self, message, details):
        self.set_busy(False)
        self.show_error(message, details)
        if self.self_test:
            QTimer.singleShot(300, lambda: QApplication.exit(1))

    def populate_table(self, rows):
        self.clear_table()
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            x1, y1, x2, y2 = row["box"]
            values = [row["class"], f'{row["conf"]:.3f}', f"{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}"]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.table.setItem(i, col, item)

    def clear_table(self):
        self.table.setRowCount(0)

    def save_output(self):
        if self.annotated_bgr is None:
            QMessageBox.warning(self, "Chưa có output", "Hãy bấm Chạy detect trước.")
            return False
        path = Path(self.output_edit.text().strip() or DEFAULT_OUTPUT)
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), self.annotated_bgr)
        if not ok:
            QMessageBox.critical(self, "Lỗi", f"Không lưu được ảnh:\n{path}")
            return False
        self.log(f"✅ Đã lưu output: {path}")
        return True

    def open_output_folder(self):
        path = Path(self.output_edit.text().strip() or DEFAULT_OUTPUT)
        folder = path.parent if path.suffix else path
        os.startfile(str(folder))

    def show_error(self, message, details):
        self.log(f"❌ {message}")
        QMessageBox.critical(self, "Lỗi", f"{message}\n\nChi tiết:\n{details[-2000:]}")

    def _run_self_test_clicks(self):
        self.log("[SELF-TEST] Click Reset mặc định")
        QTest.mouseClick(self.reset_btn, Qt.LeftButton)
        QTimer.singleShot(250, self._self_test_click_detect)

    def _self_test_click_detect(self):
        self.log("[SELF-TEST] Click Chạy detect")
        QTest.mouseClick(self.detect_btn, Qt.LeftButton)

    def _finish_self_test(self):
        self.log("[SELF-TEST] Click Lưu output")
        QTest.mouseClick(self.save_btn, Qt.LeftButton)
        out = Path(self.output_edit.text().strip())
        if out.exists() and out.stat().st_size > 0 and self.table.rowCount() > 0:
            self.log("[SELF-TEST] PASS")
            QTimer.singleShot(300, lambda: QApplication.exit(0))
        else:
            self.log("[SELF-TEST] FAIL")
            QTimer.singleShot(300, lambda: QApplication.exit(2))


def parse_args():
    parser = argparse.ArgumentParser(description="FaceAutoVN Qt6 model tester")
    parser.add_argument("--self-test", action="store_true", help="Tự click các nút để kiểm thử GUI + model rồi thoát")
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("FaceAutoVN Model Tester")
    app.setStyleSheet(APP_QSS)
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    window = FaceAutoVNModelTester(self_test=args.self_test)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

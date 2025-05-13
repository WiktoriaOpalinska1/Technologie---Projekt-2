import sys, os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from PyQt5.QtCore import Qt, QPropertyAnimation, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QGraphicsOpacityEffect, QVBoxLayout, QWidget, \
    QMainWindow, QCheckBox, QHBoxLayout


# --------- MediaPipe Init ---------
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
pose = mp_pose.Pose()
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# --------- Global Config ---------
CLOTHES_CATEGORIES = {'t': 'tops', 's': 'skirts', 'p': 'pants', 'j': 'jumpers'}
ACCESSORY_CATEGORIES = ["hat", "scarf", "glasses", "crown"]
base_clothes = Path("clothes")

# --------- Load Clothes ---------
outfits = {c: list((base_clothes/f).glob("*.png")) for c, f in CLOTHES_CATEGORIES.items()}
idx = {c: 0 for c in CLOTHES_CATEGORIES}
active_clothes = {c: True for c in CLOTHES_CATEGORIES}
active_accessories = {acc: (acc == "glasses") for acc in ACCESSORY_CATEGORIES}

# --------- Load Accessories ---------
accessory_images = {acc: [] for acc in ACCESSORY_CATEGORIES}
current_acc_idx = {acc: 0 for acc in ACCESSORY_CATEGORIES}
selected_accessory = ACCESSORY_CATEGORIES[0]

for acc in ACCESSORY_CATEGORIES:
    acc_path = base_clothes / acc
    if acc_path.exists():
        accessory_images[acc] = [
            str(acc_path / f)
            for f in os.listdir(acc_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

# --------- Helper Functions ---------
def get_body_metrics(lm, w, h):
    P = lambda i: np.array([lm[i].x * w, lm[i].y * h])
    sh_l, sh_r = P(11), P(12)
    hip_l, hip_r = P(23), P(24)
    center_sh = (sh_l + sh_r) / 2
    center_hip = (hip_l + hip_r) / 2
    shoulder_w = np.linalg.norm(sh_r - sh_l)
    torso_h = np.linalg.norm(center_hip - center_sh)
    vec = sh_r - sh_l
    angle = np.arctan2(vec[1], vec[0])
    if angle > np.pi/2: angle -= np.pi
    if angle < -np.pi/2: angle += np.pi
    return shoulder_w, torso_h, angle, center_sh, center_hip

def overlay_rgba(frame, overlay, x, y):
    h, w = frame.shape[:2]
    if overlay is None:
        return frame
    o_h, o_w = overlay.shape[:2]

    # Przytnij jeśli overlay wychodzi poza krawędzie obrazu
    if x + o_w > w:
        o_w = w - x
        overlay = overlay[:, :o_w]
    if y + o_h > h:
        o_h = h - y
        overlay = overlay[:o_h, :]

    if x < 0:
        overlay = overlay[:, -x:]
        o_w += x
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        o_h += y
        y = 0

    if o_w <= 0 or o_h <= 0:
        return frame

    alpha = overlay[:, :, 3] / 255.0 if overlay.shape[2] == 4 else np.ones((o_h, o_w))

    for c in range(3):
        frame[y:y+o_h, x:x+o_w, c] = (
            alpha * overlay[:, :, c] + (1 - alpha) * frame[y:y+o_h, x:x+o_w, c]
        )
    return frame


def place_clothes(frame, outfit, lm, cat):
    h, w = frame.shape[:2]
    sh_w, torso_h, angle, center_sh, center_hip = get_body_metrics(lm, w, h)
    if cat in ('s','p'):
        anchor, y_off = center_hip, -0.10
        hip_w = np.linalg.norm(center_hip - center_sh) * 2
        scale = min((hip_w * 3.2) / outfit.shape[1], (torso_h * 2.7) / outfit.shape[0])
    elif cat == 't':
        anchor, y_off = center_sh, -0.15
        scale = min((sh_w * 1.8) / outfit.shape[1], (torso_h * 1.6) / outfit.shape[0])
    elif cat == 'j':
        anchor, y_off = center_sh, -0.20
        scale = min((sh_w * 2.5) / outfit.shape[1], (torso_h * 2.5) / outfit.shape[0])
    else:
        return frame
    outfit_rs = cv2.resize(outfit, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    oh, ow = outfit_rs.shape[:2]
    M = cv2.getRotationMatrix2D((ow/2, oh/2), -np.degrees(angle), 1.0)
    outfit_rot = cv2.warpAffine(outfit_rs, M, (ow, oh), flags=cv2.INTER_AREA, borderValue=(0,0,0,0))
    anchor_xy = (int(anchor[0] - ow/2), int(anchor[1] + oh * y_off))
    return overlay_rgba(frame, outfit_rot, *anchor_xy)

def place_accessory(frame, landmarks, w, h):
    global selected_accessory

    acc = selected_accessory
    if acc not in accessory_images or not accessory_images[acc]:
        print(f"[Błąd] Brak załadowanych obrazów dla akcesorium: {acc}")
        return frame

    if current_acc_idx[acc] >= len(accessory_images[acc]):
        print(f"[Błąd] Błędny indeks obrazu dla {acc}")
        return frame

    acc_path = accessory_images[acc][current_acc_idx[acc]]
    overlay = cv2.imread(acc_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        print(f"[Błąd] Nie udało się wczytać obrazu: {acc_path}")
        return frame

    try:
        # Landmarki twarzy
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        forehead = landmarks[10]
        chin = landmarks[152]

        lx, ly = int(left_eye.x * w), int(left_eye.y * h)
        rx, ry = int(right_eye.x * w), int(right_eye.y * h)
        fx, fy = int(forehead.x * w), int(forehead.y * h)
        cx, cy = int(chin.x * w), int(chin.y * h)

        face_width = int((rx - lx) * 1.5)
        new_w = int(face_width * 1.25)
        new_h = int(new_w * overlay.shape[0] / overlay.shape[1])
        overlay = cv2.resize(overlay, (new_w, new_h))

        # Pozycje zależne od akcesorium
        if acc == "hat":
            pos_x, pos_y = fx - new_w // 2, fy - new_h + int(new_h * 0.2)
        elif acc == "glasses":
            center_x = lx + (rx - lx) // 2
            pos_x, pos_y = center_x - new_w // 2, ly - new_h // 2
        elif acc == "scarf":
            pos_x, pos_y = cx - new_w // 2, cy + 10
        else:  # crown lub inne
            pos_x, pos_y = fx - new_w // 2, fy - new_h

        return overlay_rgba(frame, overlay, pos_x, pos_y)

    except Exception as e:
        print(f"[Wyjątek] Błąd w place_accessory: {e}")
        return frame

# --------- GUI App ---------
class StartScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Design Fashion")
        self.setFixedSize(600, 400)
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #3a1c71, stop:1 #f67093
                );
            }
            QLabel {
                color: white;
                font-size: 36px;
                font-weight: bold;
            }
            QPushButton {
                background-color: white;
                color: #3a1c71;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #eee;
            }
        """)

        self.label = QLabel("Design Fashion")
        self.label.setAlignment(Qt.AlignCenter)

        self.btn_start = QPushButton("Get Started")
        self.btn_start.setFixedSize(200, 40)
        self.btn_start.clicked.connect(self.open_main_app)

        self.fade_label = QGraphicsOpacityEffect()
        self.label.setGraphicsEffect(self.fade_label)

        self.fade_button = QGraphicsOpacityEffect()
        self.btn_start.setGraphicsEffect(self.fade_button)

        layout = QVBoxLayout()
        layout.addStretch()
        layout.addWidget(self.label)
        layout.addSpacing(40)
        layout.addWidget(self.btn_start, alignment=Qt.AlignCenter)
        layout.addStretch()

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.animate_elements()

    def animate_elements(self):
        self.anim1 = QPropertyAnimation(self.fade_label, b"opacity")
        self.anim1.setDuration(1500)
        self.anim1.setStartValue(0)
        self.anim1.setEndValue(1)
        self.anim1.start()

        self.anim2 = QPropertyAnimation(self.fade_button, b"opacity")
        self.anim2.setDuration(1500)
        self.anim2.setStartValue(0)
        self.anim2.setEndValue(1)
        QTimer.singleShot(500, self.anim2.start)
        self.anim2.start()

    def open_main_app(self):
        self.main_app = VirtualTryOnApp()
        self.main_app.show()
        self.close()


class VirtualTryOnApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Design Fashion")
        self.resize(1000, 700)
        self.initUI()
        self.capture = cv2.VideoCapture(0)
        self.cam_width = 640
        self.cam_height = 480
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def initUI(self):
        from PyQt5.QtWidgets import QGroupBox

        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: black")

        # ===== Akcesoria twarzy =====
        self.accessory_checkboxes = {}
        acc_group = QGroupBox("Accessories")
        acc_layout = QVBoxLayout()
        for acc in ACCESSORY_CATEGORIES:
            box = QCheckBox(acc)
            box.setChecked(acc == "glasses")  # domyślnie tylko glasses
            box.stateChanged.connect(self.toggle_accessory)
            self.accessory_checkboxes[acc] = box
            acc_layout.addWidget(box)
        acc_group.setLayout(acc_layout)

        # ===== Kategorie ubrań =====
        self.checkboxes = {}
        cloth_group = QGroupBox("Clothes")
        cloth_layout = QVBoxLayout()
        for c, label in CLOTHES_CATEGORIES.items():
            box = QCheckBox(label)
            box.setChecked(label in ("tops", "pants", "jumpers"))  # domyślnie zaznaczone
            box.stateChanged.connect(self.toggle_category)
            self.checkboxes[c] = box
            cloth_layout.addWidget(box)
        cloth_group.setLayout(cloth_layout)

        # ===== Przycisk "Następne" =====
        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self.next_item)

        # ===== Sidebar =====
        sidebar_container = QWidget()
        sidebar_container.setStyleSheet("""
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #3a1c71, stop:1 #f67093
            );
        """)

        # Ustawienia stylów dla grup i checkboxów
        acc_group.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        cloth_group.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")

        for checkbox in self.accessory_checkboxes.values():
            checkbox.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")

        for checkbox in self.checkboxes.values():
            checkbox.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")

        self.btn_next.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")

        sidebar = QVBoxLayout(sidebar_container)
        sidebar.addWidget(acc_group)
        sidebar.addWidget(cloth_group)
        sidebar.addSpacing(10)
        sidebar.addWidget(self.btn_next)
        sidebar.addStretch()

        # ===== Główna część =====
        main_layout = QHBoxLayout()
        main_layout.addWidget(sidebar_container, stretch=1)
        main_layout.addWidget(self.image_label, stretch=3)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def change_accessory(self, text):
        global selected_accessory
        selected_accessory = text

    def next_item(self):
        for c in CLOTHES_CATEGORIES:
            if self.checkboxes[c].isChecked() and outfits[c]:
                idx[c] = (idx[c] + 1) % len(outfits[c])
        if accessory_images[selected_accessory]:
            current_acc_idx[selected_accessory] = (current_acc_idx[selected_accessory] + 1) % len(accessory_images[selected_accessory])

    def toggle_category(self):
        for c in CLOTHES_CATEGORIES:
            active_clothes[c] = self.checkboxes[c].isChecked()

    def toggle_accessory(self):
        for acc in ACCESSORY_CATEGORIES:
            active_accessories[acc] = self.accessory_checkboxes[acc].isChecked()

    def keyPressEvent(self, event):
        key = event.text().lower()

        # Obsługa akcesoriów
        acc_keys = {
            'c': 'crown',
            'g': 'glasses',
            'h': 'hat',
            's': 'scarf',
        }
        if key in acc_keys:
            acc = acc_keys[key]
            if active_accessories.get(acc, False) and accessory_images[acc]:
                current_acc_idx[acc] = (current_acc_idx[acc] + 1) % len(accessory_images[acc])
            return

        # Obsługa ubrań
        cloth_keys = {
            'j': 'j',  # jumpers
            'p': 'p',  # pants
            't': 't',  # tops
            'q': 's',  # skirts (bo "s" już zajęte przez scarf)
        }
        if key in cloth_keys:
            c = cloth_keys[key]
            if active_clothes.get(c, False) and outfits[c]:
                idx[c] = (idx[c] + 1) % len(outfits[c])
            return

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret: return
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(rgb)
        face_res = face_mesh.process(rgb)

        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark
            for c in CLOTHES_CATEGORIES:
                if active_clothes[c] and outfits[c]:
                    path = outfits[c][idx[c] % len(outfits[c])]
                    outfit = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    frame = place_clothes(frame, outfit, lm, c)

        if face_res.multi_face_landmarks:
            for fl in face_res.multi_face_landmarks:
                for acc in ACCESSORY_CATEGORIES:
                    if active_accessories.get(acc, False):
                        global selected_accessory
                        selected_accessory = acc
                        frame = place_accessory(frame, fl.landmark, w, h)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), aspectRatioMode=Qt.KeepAspectRatio)

        self.image_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.capture.release()
        pose.close()
        face_mesh.close()
        cv2.destroyAllWindows()



# --------- Run App ---------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    start_screen = StartScreen()
    start_screen.show()
    sys.exit(app.exec_())


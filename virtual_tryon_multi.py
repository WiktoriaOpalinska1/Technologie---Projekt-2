# bez landmarkow

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# ——— MediaPipe Pose ———
# narzedzia do rysowania landmarków
mp_drawing = mp.solutions.drawing_utils
# model detekcji ciała
mp_pose = mp.solutions.pose

# ——— Utility: body metrics - landmarks ———
 
    #Na podstawie listy 33 landmarków (lm) oraz wymiarów obrazu (w,h):
    #- oblicza położenie barków i bioder w pikselach 
    #- center_sh = środek między barkami
    #- center_hip = środek między biodrami
    #- shoulder_w = szerokość barków
    #- torso_h = odległość między środkiem barków a bioder
    #- angle = kąt nachylenia linii barków (w radianach)
    #Zwraca (shoulder_w, torso_h, angle, center_sh, center_hip).
    
def get_body_metrics(lm, w, h):
    P = lambda i: np.array([lm[i].x*w, lm[i].y*h])
    sh_l, sh_r = P(11), P(12)
    hip_l,hip_r= P(23), P(24)

    #środek miedzy barkami i biodrem
    center_sh  = (sh_l + sh_r)/2
    center_hip = (hip_l + hip_r)/2

    # Długości: szerokość barków i wysokość tułowia
    shoulder_w = np.linalg.norm(sh_r - sh_l)
    torso_h    = np.linalg.norm(center_hip - center_sh)

    # Kąt nachylenia linii barków
    vec        = sh_r - sh_l
    angle      = np.arctan2(vec[1], vec[0])

    # Zamknięcie kąta do przedziału [-90°, +90°]
    if angle >  np.pi/2: angle -= np.pi
    if angle < -np.pi/2: angle += np.pi

    return shoulder_w, torso_h, angle, center_sh, center_hip

# ——— nakładanie obrazu RGBA na BGR ———
def overlay_rgba(frame, outfit_rot, anchor, y_off):
    h, w = frame.shape[:2]
    oh, ow = outfit_rot.shape[:2]

    # obliczanie pozycji
    x0 = int(anchor[0] - ow/2)
    y0 = int(anchor[1] + oh * y_off)
    x1, y1 = max(0, x0), max(0, y0)

    hc = min(oh, h - y1)
    wc = min(ow, w - x1)

    # STRAŻNIK: jeśli nie ma co nakładać, wracamy
    if hc <= 0 or wc <= 0:
        return frame

    crop = outfit_rot[0:hc, 0:wc]
    roi  = frame[y1:y1+hc, x1:x1+wc]

    alpha = crop[:, :, 3:4] / 255.0
    roi[:] = (roi * (1 - alpha) + crop[:, :, :3] * alpha).astype(np.uint8)
    frame[y1:y1+hc, x1:x1+wc] = roi
    return frame


# ——— Main overlay function ———
def place_clothes(frame, outfit, lm, cat):
    h, w = frame.shape[:2]
    sh_w, torso_h, angle, center_sh, center_hip = get_body_metrics(lm, w, h)

    # wybór parametrów wg kategorii
    if cat == 't':      # Tops
        anchor, y_off = center_sh, -0.15
        scale_w = (sh_w * 1.8) / outfit.shape[1]
        scale_h = (torso_h * 1.6) / outfit.shape[0]

    elif cat in ('s','p'):  # Skirts / Pants
        hip_w = np.linalg.norm(center_hip - center_sh) * 2  # approximate hip span
        anchor, y_off = center_hip, -0.10
        scale_w = (hip_w * 3.2) / outfit.shape[1]
        scale_h = (torso_h * 2.7) / outfit.shape[0]

    elif cat == 'b':    # bluzy → traktujemy jak Tops
        anchor, y_off = center_sh, -0.20
        scale_w = (sh_w * 2.5) / outfit.shape[1]
        scale_h = (torso_h * 2.5) / outfit.shape[0]

    else:
        return frame

    scale = min(scale_w, scale_h)
    outfit_rs = cv2.resize(outfit, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    oh, ow = outfit_rs.shape[:2]

    M = cv2.getRotationMatrix2D((ow/2, oh/2), -np.degrees(angle), 1.0)
    outfit_rot = cv2.warpAffine(outfit_rs, M, (ow, oh),
                                flags=cv2.INTER_AREA,
                                borderValue=(0,0,0,0))

    return overlay_rgba(frame, outfit_rot, anchor, y_off)

# ——— Prepare lists of PNGs ———
CATS = {'t':'tops', 's':'skirts', 'p':'pants', 'b':'bluzy'}
base = Path("assets/clothes")
outfits = {c:list((base/f).glob("*.png")) for c,f in CATS.items()}
idx = {c:0 for c in CATS}

# ——— Run camera + loop ———
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_pose.Pose(static_image_mode=False,
                  model_complexity=1,
                  enable_segmentation=False,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    cat = 't'
    while True:
        ret, frame = cap.read()
        if not ret: break

        # detect
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); 
        rgb.flags.writeable=False
        res = pose.process(rgb)
        rgb.flags.writeable=True; 
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if res.pose_landmarks:
            path   = outfits[cat][idx[cat] % len(outfits[cat])]
            outfit = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            frame  = place_clothes(frame, outfit, res.pose_landmarks.landmark, cat)

        total = len(outfits[cat])
        if total > 0:
            current = (idx[cat] % total) + 1
        else:
            current = 0
            total = 0

        cv2.putText(frame, f"Mode: {cat.upper()} {idx[cat]+1}/{len(outfits[cat])}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Try-On (t/s/p/d,n,Esc)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:      break
        if key in map(ord, CATS.keys()):
            cat = chr(key); idx[cat]=0
        if key == ord('n') and len(outfits[cat]) > 0:
            # zwiększ i natychmiast zrób modulo długości listy
            idx[cat] = (idx[cat] + 1) % len(outfits[cat])


cap.release()
cv2.destroyAllWindows()

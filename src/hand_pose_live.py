import cv2
import time
import csv
import os
import numpy as np
import mediapipe as mp
from pathlib import Path

# ---------- Settings ----------
CAM_INDEX = 0
MAX_NUM_HANDS = 2
DETECTION_CONF = 0.7
TRACKING_CONF = 0.7

DRAW = True

# Optional: set True to dump landmarks every frame to CSV (for Phase 2)
DUMP_CSV = False
ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT_DIR / "landmarks_dump.csv"
# -----------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def open_csv_writer(path):
    # Columns: ts, frame_idx, hand_idx, handedness, then x0,y0,z0...x20,y20,z20
    header = ["ts", "frame_idx", "hand_idx", "handedness"]
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    is_new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if is_new:
        w.writerow(header)
    return f, w

def put_text(img, text, org, scale=0.7, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness-1, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

    # Try to reduce latency (not always honored)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    csv_file, csv_writer = (None, None)
    if DUMP_CSV:
        csv_file, csv_writer = open_csv_writer(CSV_PATH)
        print(f"[CSV] Dumping landmarks to: {CSV_PATH}")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=1,
        min_detection_confidence=DETECTION_CONF,
        min_tracking_confidence=TRACKING_CONF
    )

    prev_t = time.time()
    fps = 0.0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                continue

            frame_idx += 1
            h, w = frame.shape[:2]

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)
            rgb.flags.writeable = True

            # FPS calc
            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            # Draw + extract
            hand_count = 0
            if res.multi_hand_landmarks and res.multi_handedness:
                for hand_idx, (landmarks, handed) in enumerate(zip(res.multi_hand_landmarks, res.multi_handedness)):
                    hand_count += 1
                    label = handed.classification[0].label  # "Left" / "Right"
                    score = handed.classification[0].score

                    # Landmark array (21x3) normalized to [0..1] for x,y
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)

                    if DUMP_CSV:
                        row = [now, frame_idx, hand_idx, label]
                        row += pts.flatten().tolist()
                        csv_writer.writerow(row)

                    if DRAW:
                        mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                        # Put handedness near wrist (landmark 0)
                        wrist = pts[0]
                        x_px, y_px = int(wrist[0] * w), int(wrist[1] * h)
                        put_text(frame, f"{label} ({score:.2f})", (x_px + 10, y_px - 10), scale=0.6, thickness=2)

            # HUD
            put_text(frame, f"FPS: {fps:.1f}", (10, 30))
            put_text(frame, f"Hands: {hand_count}", (10, 60))
            put_text(frame, "Press Q to quit", (10, 90), scale=0.6, thickness=2)

            cv2.imshow("Hand Pose Live", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()

if __name__ == "__main__":
    main()

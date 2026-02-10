import cv2
import time
import os
import numpy as np
import mediapipe as mp
from pathlib import Path

# ---------------- Settings ----------------
CAM_INDEX = 0
MAX_NUM_HANDS = 2
DETECTION_CONF = 0.7
TRACKING_CONF = 0.7

# Recording settings
MIN_FRAMES = 10            # minimum frames to accept a sample
MAX_SECONDS = 2.0          # safety cap if you forget to stop
ROOT_DIR = Path(__file__).resolve().parents[1]
SAVE_ROOT = ROOT_DIR / "dataset"      # dataset folder
# ------------------------------------------

GESTURES = {
    ord('1'): "STOP_IDLE",
    ord('2'): "SWIPE_LEFT",
    ord('3'): "SWIPE_RIGHT",
    ord('4'): "PUSH",
    ord('5'): "PULL",
    ord('6'): "CIRCLE_CW",
    ord('7'): "CIRCLE_CCW",
}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def put_text(img, text, org, scale=0.7, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness-1, cv2.LINE_AA)

def ensure_dirs():
    for g in set(GESTURES.values()):
        for hand in ["Left", "Right"]:
            (SAVE_ROOT / g / hand).mkdir(parents=True, exist_ok=True)

def pick_primary_hand(res, frame_w, frame_h):
    """
    Returns (label, pts63, wrist_xy_px, score) for the most central detected hand.
    pts63 is np.array shape (63,)
    """
    if not (res.multi_hand_landmarks and res.multi_handedness):
        return None

    candidates = []
    for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
        label = handed.classification[0].label
        score = handed.classification[0].score
        pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)  # (21,3)
        wrist = pts[0]
        wrist_px = (int(wrist[0] * frame_w), int(wrist[1] * frame_h))
        # centrality score (prefer hands near center of frame)
        cx, cy = frame_w / 2, frame_h / 2
        dist2 = (wrist_px[0] - cx) ** 2 + (wrist_px[1] - cy) ** 2
        candidates.append((dist2, label, score, pts, wrist_px, lm))

    candidates.sort(key=lambda x: x[0])  # smallest distance = most central
    _, label, score, pts, wrist_px, lm_obj = candidates[0]
    return label, pts.reshape(-1), wrist_px, score, lm_obj

def save_sample(gesture, hand_label, X, ts, fps_est):
    # filename with timestamp
    t = time.strftime("%Y%m%d_%H%M%S")
    out_dir = SAVE_ROOT / gesture / hand_label
    idx = len([f for f in os.listdir(out_dir) if f.endswith(".npz")])
    out_path = out_dir / f"{t}_{idx:04d}.npz"
    np.savez_compressed(
        out_path,
        X=np.asarray(X, dtype=np.float32),
        ts=np.asarray(ts, dtype=np.float64),
        fps_est=float(fps_est),
        gesture=str(gesture),
        hand=str(hand_label),
    )
    return str(out_path)

def main():
    ensure_dirs()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=1,
        min_detection_confidence=DETECTION_CONF,
        min_tracking_confidence=TRACKING_CONF
    )

    selected_gesture = "STOP_IDLE"
    is_recording = False
    rec_X, rec_ts = [], []
    rec_start = None

    prev_t = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)  # horizontal flip
            if not ret:
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)
            rgb.flags.writeable = True

            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            primary = pick_primary_hand(res, w, h)
            hand_label = None
            if primary:
                hand_label, pts63, wrist_px, score, lm_obj = primary
                mp_draw.draw_landmarks(frame, lm_obj, mp_hands.HAND_CONNECTIONS)
                put_text(frame, f"{hand_label} ({score:.2f})", (wrist_px[0] + 10, wrist_px[1] - 10), scale=0.6)

                # If recording, append
                if is_recording:
                    rec_X.append(pts63.tolist())
                    rec_ts.append(now)

            # Auto-stop if too long
            if is_recording and rec_start and (now - rec_start) > MAX_SECONDS:
                is_recording = False

            # HUD
            put_text(frame, f"FPS: {fps:.1f}", (10, 30))
            put_text(frame, f"Gesture: {selected_gesture}", (10, 60))
            put_text(frame, f"Recording: {is_recording}  Frames: {len(rec_X)}", (10, 90))
            put_text(frame, "Keys: 1-7 select, Q or P start/stop, V quit", (10, 120), scale=0.55, thickness=2)

            cv2.imshow("Record Landmarks", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in GESTURES:
                selected_gesture = GESTURES[key]

            elif key in (ord('q'), ord('Q'), ord('p'), ord('P')):
                if not is_recording:
                    # start
                    is_recording = True
                    rec_X, rec_ts = [], []
                    rec_start = time.time()
                else:
                    # stop + save (only if enough frames and we have a hand label)
                    is_recording = False
                    if len(rec_X) >= MIN_FRAMES and hand_label in ("Left", "Right"):
                        out = save_sample(selected_gesture, hand_label, rec_X, rec_ts, fps)
                        print(f"[Saved] {out}  ({len(rec_X)} frames)")
                    else:
                        print(f"[Discarded] frames={len(rec_X)} hand={hand_label}")

            elif key in (ord('v'), ord('V')):
                break

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

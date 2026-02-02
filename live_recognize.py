import time
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from joblib import load
from features import motion_features, wrist_xy

# ---------------- Settings ----------------
CAM_INDEX = 0
MAX_NUM_HANDS = 2
DETECTION_CONF = 0.7
TRACKING_CONF = 0.7

# Flip webcam horizontally (recommended for laptop selfie view)
FLIP_IMAGE = True

# Sliding window size (seconds) for classification
WINDOW_SECONDS = 1.2

# Prediction smoothing: require same predicted label N times in a row
STREAK_REQUIRED = 3

# Confidence threshold for accepting a prediction
CONF_THRESH = 0.70

# Cooldown to prevent spamming (seconds)
COOLDOWN_SECONDS = 0.6

# Arm/disarm behavior
ARM_BY_STOP = True
STOP_LABEL = "STOP_IDLE"
STOP_ARM_STREAK = 3          # must see STOP this many times to arm/disarm
DISARM_ON_STOP = True        # STOP also disarms (safety reset)

# Model paths
MODEL_PATH = r"models\gesture_rf.joblib"
LABELS_PATH = r"models\labels.txt"
# -----------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def put_text(img, text, org, scale=0.7, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), max(1, thickness - 1), cv2.LINE_AA)


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels


def pick_primary_hand(res, frame_w, frame_h):
    """
    Returns (label, pts63, wrist_px, score, lm_obj) for the most central detected hand.
    """
    if not (res.multi_hand_landmarks and res.multi_handedness):
        return None

    candidates = []
    cx, cy = frame_w / 2, frame_h / 2

    for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
        label = handed.classification[0].label  # "Left" / "Right"
        score = float(handed.classification[0].score)

        pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)  # (21,3)
        wrist = pts[0]
        wrist_px = (int(wrist[0] * frame_w), int(wrist[1] * frame_h))

        dist2 = (wrist_px[0] - cx) ** 2 + (wrist_px[1] - cy) ** 2
        candidates.append((dist2, label, score, pts.reshape(-1), wrist_px, lm))

    candidates.sort(key=lambda x: x[0])
    _, label, score, pts63, wrist_px, lm_obj = candidates[0]
    return label, pts63, wrist_px, score, lm_obj

def motion_score(seq_63):
    """
    A simple 'how much did the hand move' scalar.
    Uses wrist XY speed in normalized image coords.
    """
    wxy = wrist_xy(seq_63)              # (T,2)
    dv = np.diff(wxy, axis=0)           # (T-1,2)
    speed = np.linalg.norm(dv, axis=1)  # (T-1,)
    return float(np.mean(speed)), float(np.max(speed))


# --- Simple command mapping (prints only) ---
def gesture_to_command(gesture, hand_label):
    """
    For now just returns a dict-like string for printing.
    Later we’ll publish this over MQTT to your sim/robot.
    """
    # Example mapping placeholders (you’ll tune this later)
    if gesture == "SWIPE_LEFT":
        return {"cmd": "ROTATE_LEFT", "hand": hand_label}
    if gesture == "SWIPE_RIGHT":
        return {"cmd": "ROTATE_RIGHT", "hand": hand_label}
    if gesture == "PUSH":
        return {"cmd": "FORWARD", "hand": hand_label}
    if gesture == "PULL":
        return {"cmd": "BACKWARD", "hand": hand_label}
    if gesture == "CIRCLE_CW":
        return {"cmd": "CIRCLE_CW_MODE", "hand": hand_label}
    if gesture == "CIRCLE_CCW":
        return {"cmd": "CIRCLE_CCW_MODE", "hand": hand_label}
    if gesture == STOP_LABEL:
        return {"cmd": "STOP_RESET", "hand": hand_label}
    return {"cmd": "UNKNOWN", "hand": hand_label}


def main():
    labels = load_labels(LABELS_PATH)
    clf = load(MODEL_PATH)

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

    # buffers for sliding window: store (ts, pts63)
    buf_ts = deque()
    buf_pts = deque()

    # runtime state
    armed = False
    last_event_time = 0.0

    pred_streak_label = None
    pred_streak_count = 0

    stop_streak = 0

    # FPS
    prev_t = time.time()
    fps = 0.0

    print("\n[Live Recognize] Running.")
    print("Controls: Q quit | (optional) A toggle armed\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            if FLIP_IMAGE:
                frame = cv2.flip(frame, 1)

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
            top_label = None
            top_conf = 0.0

            if primary:
                hand_label, pts63, wrist_px, hand_score, lm_obj = primary

                # draw landmarks + handedness
                mp_draw.draw_landmarks(frame, lm_obj, mp_hands.HAND_CONNECTIONS)
                put_text(frame, f"{hand_label} ({hand_score:.2f})", (wrist_px[0] + 10, wrist_px[1] - 10), scale=0.6)

                # update sliding buffer
                buf_ts.append(now)
                buf_pts.append(pts63)

                # keep only last WINDOW_SECONDS
                while buf_ts and (now - buf_ts[0]) > WINDOW_SECONDS:
                    buf_ts.popleft()
                    buf_pts.popleft()

                # run inference if we have enough frames
                if len(buf_pts) >= 6:
                    seq = np.stack(buf_pts, axis=0)  # (T,63)
                    # --- STOP stability gate (fast arm/disarm, avoids STOP->PUSH confusion) ---
                    mean_spd, max_spd = motion_score(seq)

                    STABLE_MEAN = 0.0025   # tune: 0.002 to 0.004
                    STABLE_MAX  = 0.008    # tune: 0.006 to 0.012

                    stable_like_stop = (mean_spd < STABLE_MEAN) and (max_spd < STABLE_MAX)

                    feat = motion_features(seq)
                    if feat is not None:
                        probs = clf.predict_proba(feat.reshape(1, -1))[0]
                        idx = int(np.argmax(probs))
                        top_label = clf.classes_[idx]
                        top_conf = float(probs[idx])

                        if stable_like_stop and (not armed):
                            # only for display/smoothing; doesn't change the trained classifier itself
                            top_label = STOP_LABEL
                            top_conf = max(top_conf, 0.99)

                        # update smoothing streak
                        if top_label == pred_streak_label:
                            pred_streak_count += 1
                        else:
                            pred_streak_label = top_label
                            pred_streak_count = 1

                        # STOP gating streak (used for arm/disarm)
                        # If the hand is very stable, treat it as STOP-like even if classifier briefly says PUSH.
                        if stable_like_stop:
                            stop_streak += 1
                        elif top_label == STOP_LABEL and top_conf >= CONF_THRESH:
                            stop_streak += 1
                        else:
                            stop_streak = max(0, stop_streak - 1)  # decay instead of hard reset

                        # Decide if we trigger an event
                        cool_ok = (now - last_event_time) >= COOLDOWN_SECONDS
                        confident_ok = top_conf >= CONF_THRESH
                        stable_ok = pred_streak_count >= STREAK_REQUIRED

                        # Arm/disarm logic driven by STOP
                        if ARM_BY_STOP and stop_streak >= STOP_ARM_STREAK:
                            if not armed:
                                armed = True
                                print(f"[ARMED] via {STOP_LABEL}  conf={top_conf:.2f}  hand={hand_label}")
                                last_event_time = now
                            elif DISARM_ON_STOP:
                                armed = False
                                print(f"[DISARM] via {STOP_LABEL}  conf={top_conf:.2f}  hand={hand_label}")
                                last_event_time = now
                            stop_streak = 0  # reset

                        # Trigger non-STOP gestures only when armed
                        if top_label and top_label != STOP_LABEL:
                            if armed and confident_ok and stable_ok and cool_ok:
                                msg = gesture_to_command(top_label, hand_label)
                                print(f"[EVENT] {top_label:10s} conf={top_conf:.2f} hand={hand_label}  -> {msg}")
                                last_event_time = now

            # HUD
            put_text(frame, f"FPS: {fps:.1f}", (10, 30))
            put_text(frame, f"Armed: {armed}", (10, 60))
            if top_label:
                put_text(frame, f"spd mean/max: {mean_spd:.4f}/{max_spd:.4f}", (10, 180), scale=0.55)
            put_text(frame, f"Window: {WINDOW_SECONDS:.1f}s  Thresh: {CONF_THRESH:.2f}", (10, 120), scale=0.55)
            put_text(frame, "Make STOP to arm/disarm. Q quit.", (10, 150), scale=0.55)

            cv2.imshow("Live Recognize", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('a'), ord('A')):
                armed = not armed
                print(f"[TOGGLE] armed={armed}")

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

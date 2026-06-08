import time
import json
import os
import socket
import subprocess
import ipaddress
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from collections import deque
from joblib import load
from features import motion_features, wrist_xy


def resolve_bridge_host():
    env_host = os.environ.get("GESTURE_BRIDGE_HOST")
    if env_host:
        return env_host, "GESTURE_BRIDGE_HOST"

    wsl_distro = os.environ.get("GESTURE_WSL_DISTRO", "Ubuntu-20.04")
    wsl_user = os.environ.get("GESTURE_WSL_USER", "ros_joe")

    def collect_candidates(wsl_args, source):
        candidates = []
        try:
            output = subprocess.check_output(
                [*wsl_args, "sh", "-lc", "ip -4 -o addr"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
            for line in output.splitlines():
                parts = line.split()
                if len(parts) >= 4 and parts[2] == "inet":
                    candidates.append((source, parts[1], parts[3].split("/", 1)[0]))
        except (OSError, subprocess.SubprocessError):
            pass
        return candidates

    candidates = collect_candidates(
        ["wsl", "-d", wsl_distro, "-u", wsl_user],
        f"{wsl_distro}/{wsl_user}",
    )
    if not candidates:
        candidates = collect_candidates(["wsl"], "default WSL")

    def candidate_score(item):
        source, iface, host = item
        try:
            addr = ipaddress.ip_address(host)
        except ValueError:
            return -1

        if (
            addr.version != 4
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_unspecified
        ):
            return -1

        # Prefer WSL's private NAT range over VPN or Wi-Fi addresses.
        if host.startswith("172.") and addr.is_private:
            return 100
        if iface.startswith("eth") and addr.is_private:
            return 80
        if host.startswith("10.") and addr.is_private:
            return 60
        if host.startswith("192.168.") and addr.is_private:
            return 40
        return 10

    ranked = sorted(candidates, key=candidate_score, reverse=True)
    if ranked and candidate_score(ranked[0]) >= 0:
        source, iface, host = ranked[0]
        return host, f"auto WSL {source} {iface}"

    return "127.0.0.1", "fallback"


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
ARM_GESTURES = {"CIRCLE_CW", "CIRCLE_CCW"}
STOP_LABEL = "STOP_IDLE"
STOP_DISARM_STREAK = 3       # must see STOP this many times to disarm

# Model paths
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "gesture_rf.joblib"
LABELS_PATH = ROOT_DIR / "models" / "labels.txt"

# UDP bridge to the ROS 2/Gazebo driver running in WSL.
BRIDGE_HOST, BRIDGE_HOST_SOURCE = resolve_bridge_host()
BRIDGE_PORT = int(os.environ.get("GESTURE_BRIDGE_PORT", "4210"))
HEARTBEAT_SECONDS = 0.2
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


GESTURE_TO_DRIVE_CMD = {
    STOP_LABEL: "STOP",
    "PUSH": "FORWARD",
    "PULL": "BACKWARD",
    "SWIPE_LEFT": "LEFT",
    "SWIPE_RIGHT": "RIGHT",
}


def gesture_to_drive_command(gesture):
    return GESTURE_TO_DRIVE_CMD.get(gesture)


def send_drive_command(sock, addr, cmd, gesture, confidence, hand_label):
    payload = {
        "cmd": cmd,
        "gesture": gesture or "",
        "confidence": float(confidence),
        "hand": hand_label or "",
        "ts": time.time(),
    }
    sock.sendto(json.dumps(payload).encode("utf-8"), addr)


def main():
    labels = load_labels(LABELS_PATH)
    clf = load(MODEL_PATH)
    bridge_addr = (BRIDGE_HOST, BRIDGE_PORT)
    bridge_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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
    active_cmd = "STOP"
    active_gesture = STOP_LABEL
    active_conf = 1.0
    active_hand = ""
    last_bridge_send = 0.0

    # FPS
    prev_t = time.time()
    fps = 0.0

    print("\n[Live Recognize] Running.")
    print("Controls: Q quit | A toggle armed | circles arm | STOP disarms\n")
    print(f"[Gesture Bridge] Sending UDP commands to {BRIDGE_HOST}:{BRIDGE_PORT} ({BRIDGE_HOST_SOURCE})")

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

                        if stable_like_stop:
                            # only for display/smoothing; doesn't change the trained classifier itself
                            top_label = STOP_LABEL
                            top_conf = max(top_conf, 0.99)

                        # update smoothing streak
                        if top_label == pred_streak_label:
                            pred_streak_count += 1
                        else:
                            pred_streak_label = top_label
                            pred_streak_count = 1

                        # STOP gating streak (used for disarm)
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

                        # Circle gestures arm the bridge. STOP only disarms/stops.
                        if top_label in ARM_GESTURES and confident_ok and stable_ok and cool_ok:
                            if not armed:
                                armed = True
                                active_cmd = "STOP"
                                active_gesture = top_label
                                active_conf = top_conf
                                active_hand = hand_label or ""
                                send_drive_command(bridge_sock, bridge_addr, active_cmd, active_gesture, active_conf, active_hand)
                                last_bridge_send = now
                                print(f"[ARMED] via {top_label}  conf={top_conf:.2f}  hand={hand_label}")
                                last_event_time = now

                        if stop_streak >= STOP_DISARM_STREAK:
                            if armed or active_cmd != "STOP":
                                armed = False
                                active_cmd = "STOP"
                                active_gesture = STOP_LABEL
                                active_conf = top_conf
                                active_hand = hand_label or ""
                                send_drive_command(bridge_sock, bridge_addr, active_cmd, active_gesture, active_conf, active_hand)
                                last_bridge_send = now
                                print(f"[DISARM] via {STOP_LABEL}  conf={top_conf:.2f}  hand={hand_label}")
                                last_event_time = now
                            stop_streak = 0  # reset

                        # Trigger non-STOP gestures only when armed
                        if top_label and top_label != STOP_LABEL and top_label not in ARM_GESTURES:
                            if armed and confident_ok and stable_ok and cool_ok:
                                cmd = gesture_to_drive_command(top_label)
                                if cmd:
                                    active_cmd = cmd
                                    active_gesture = top_label
                                    active_conf = top_conf
                                    active_hand = hand_label or ""
                                    send_drive_command(bridge_sock, bridge_addr, active_cmd, active_gesture, active_conf, active_hand)
                                    last_bridge_send = now
                                    print(f"[EVENT] {top_label:10s} conf={top_conf:.2f} hand={hand_label}  -> {active_cmd}")
                                    last_event_time = now

            if (now - last_bridge_send) >= HEARTBEAT_SECONDS:
                send_drive_command(bridge_sock, bridge_addr, active_cmd, active_gesture, active_conf, active_hand)
                last_bridge_send = now

            # HUD
            put_text(frame, f"FPS: {fps:.1f}", (10, 30))
            put_text(frame, f"Armed: {armed}", (10, 60))
            put_text(frame, f"Cmd: {active_cmd}", (10, 90), scale=0.55)
            if top_label:
                put_text(frame, f"spd mean/max: {mean_spd:.4f}/{max_spd:.4f}", (10, 210), scale=0.55)
            put_text(frame, f"Window: {WINDOW_SECONDS:.1f}s  Thresh: {CONF_THRESH:.2f}", (10, 150), scale=0.55)
            put_text(frame, "Circle arms. STOP disarms. Q quit.", (10, 180), scale=0.55)

            cv2.imshow("Live Recognize", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('a'), ord('A')):
                armed = not armed
                active_cmd = "STOP"
                active_gesture = STOP_LABEL
                active_conf = 1.0
                send_drive_command(bridge_sock, bridge_addr, active_cmd, active_gesture, active_conf, active_hand)
                last_bridge_send = now
                print(f"[TOGGLE] armed={armed}")

    finally:
        hands.close()
        cap.release()
        bridge_sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# Hand-Gestured Car: How It Works

This project records hand-landmark sequences, trains a gesture classifier, and runs live gesture recognition from webcam frames.

## High-level pipeline

1. `record_landmarks.py` records labeled sequences into `dataset/<GESTURE>/<HAND>/*.npz`.
2. `train_baseline.py` loads those sequences, converts each one to a fixed-length feature vector, trains a RandomForest model, and saves it to `models/gesture_rf.joblib`.
3. `live_recognize.py` captures a live sliding window, computes the exact same feature vector format, predicts a gesture, smooths predictions, and outputs commands.

## File responsibilities

- `hand_pose_live.py`
  - Debug/visualization utility for live MediaPipe landmarks.
  - Optional CSV dump mode.

- `record_landmarks.py`
  - Data collection tool.
  - You pick a gesture with keys `1..7` and start/stop recording with `Q`/`P`.
  - Saves samples as compressed `.npz` files with:
    - `X`: `(T, 63)` (21 landmarks x 3 coords over time)
    - `ts`: timestamps
    - `fps_est`
    - `gesture`, `hand`

- `train_baseline.py`
  - Loads all samples from `dataset`.
  - Extracts feature vectors per sample via `motion_features` from `features.py`.
  - Splits train/test stratified by gesture.
  - Trains `RandomForestClassifier` and prints report + confusion matrix.
  - Saves model and label list in `models/`.

- `live_recognize.py`
  - Captures webcam frames and tracks landmarks.
  - Maintains a sliding time window (`WINDOW_SECONDS`).
  - Uses shared `motion_features` from `features.py` (same shape/order as training).
  - Applies confidence threshold + streak smoothing + cooldown.
  - Maps predicted gesture to a command payload.

- `features.py`
  - Shared feature extraction module used by both training and live inference.
  - Holds `normalize_landmarks`, `wrist_xy`, `wrist_path_features`, `signed_rotation_features`, and `motion_features`.

## Feature extraction details

`features.py` defines one shared feature layout used by both training and live inference:

1. Mean of normalized landmarks `(63)`
2. Std of normalized landmarks `(63)`
3. Mean of landmark velocities `(63)`
4. Std of landmark velocities `(63)`
5. Velocity-magnitude summary `(3)`
6. `wrist_path_features` `(4)`
7. `signed_rotation_features` `(6)`

Total feature length: `63+63+63+63+3+4+6 = 265`.

### Why normalization is used

`normalize_landmarks` makes features more robust by:
- subtracting wrist position -> translation invariance
- dividing by wrist-to-middle-MCP distance -> scale invariance

This reduces sensitivity to where your hand appears in frame and how close it is to camera.

### Why `signed_rotation_features` helps

`wrist_path_features` captures how curved the path is, but not direction sign strongly enough.

`signed_rotation_features` adds explicit clockwise/counterclockwise signal using signed turn angle over time:
- cross product sign of consecutive velocity vectors gives turn direction
- aggregated signed-angle stats (`sum_ang`, `mean_ang`, positive/negative fractions, etc.) separate CW vs CCW better

This usually improves circle-direction disambiguation (`CIRCLE_CW` vs `CIRCLE_CCW`) and reduces confusion between them.

## Important consistency rule

Training and inference must compute features in the same order and shape. Keeping this logic in `features.py` avoids drift.

## Typical workflow

1. Collect data:
   - `python record_landmarks.py`
2. Train model:
   - `python train_baseline.py`
3. Run live recognition:
   - `python live_recognize.py`

If you change features, retrain before running live recognition.

## Current gesture set

`STOP_IDLE`, `SWIPE_LEFT`, `SWIPE_RIGHT`, `PUSH`, `PULL`, `CIRCLE_CW`, `CIRCLE_CCW`

## Troubleshooting

- If training fails with missing package errors, install dependencies (for example: `scikit-learn`, `numpy`, `joblib`).
- If live recognition behaves oddly after feature edits, retrain and verify model/labels were regenerated.
- If camera does not open, check `CAM_INDEX` in scripts.

## GitHub upload safety

- Local training samples are saved in `dataset/` as landmark arrays (`X`) + metadata (`ts`, `gesture`, `hand`), not raw video frames.
- The repo `.gitignore` excludes `dataset/` and model binaries by default so you do not accidentally push them.
- `models/labels.txt` can stay public (label names only).

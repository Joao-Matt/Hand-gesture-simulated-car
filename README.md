# Hand-Gestured Car

Realtime hand-gesture recognition for driving a Gazebo car simulation. The Windows app uses MediaPipe landmarks and a RandomForest classifier, then sends UDP commands to a ROS 2 Foxy node running in WSL.

## What is included

- Data recording tool: `src/record_landmarks.py`
- Training script: `src/train_baseline.py`
- Live inference script: `src/live_recognize.py`
- Shared feature extraction: `src/features.py`
- ROS/Gazebo integration notes: `GESTURE_GAZEBO_CAR_WORKLOG.md`
- Detailed recognizer architecture docs: `HOW_IT_WORKS.md`

## Gesture controls

- `CIRCLE_CW` / `CIRCLE_CCW`: arm the bridge
- `STOP_IDLE`: disarm and stop
- `PUSH`: forward
- `PULL`: backward
- `SWIPE_LEFT`: turn left
- `SWIPE_RIGHT`: turn right

## Quick start

### Windows recognizer

```bash
python -m pip install -r requirements.txt
python src/record_landmarks.py
python src/train_baseline.py
python src/live_recognize.py
```

### ROS/Gazebo simulation

In WSL Ubuntu-20.04:

```bash
cd ~/HandGestureDrone_ws
source /opt/ros/foxy/setup.bash
source install/setup.bash
ros2 launch car_assembly_2nd gesture_car.launch.py
```

Then start the Windows recognizer:

```powershell
cd "C:\Users\Yoav\Desktop\Personal Projects\hand-gestured-car"
.\.venv\Scripts\activate
python .\src\live_recognize.py
```

The recognizer auto-detects the WSL Ubuntu-20.04 IP for the UDP bridge unless `GESTURE_BRIDGE_HOST` is set manually.

## Privacy and dataset policy

- This repo is configured to **not** commit local training data by default (`dataset/`).
- Trained model binaries are also ignored by default (`models/*`), except `models/labels.txt`.
- Recorded samples store landmarks/time metadata, not raw video frames.
- Local writeups and demo videos are ignored by default.

## Notes

- If you change feature logic, keep training and live inference in sync (this is centralized in `src/features.py`).
- Current gesture classes are listed in `models/labels.txt`.

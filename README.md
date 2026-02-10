# Hand-Gestured Car

Realtime hand-gesture recognition for driving simple car commands with MediaPipe landmarks + a RandomForest classifier.

## What is included

- Data recording tool: `src/record_landmarks.py`
- Training script: `src/train_baseline.py`
- Live inference script: `src/live_recognize.py`
- Shared feature extraction: `src/features.py`
- Detailed architecture docs: `HOW_IT_WORKS.md`

## Quick start

```bash
python -m pip install -r requirements.txt
python src/record_landmarks.py
python src/train_baseline.py
python src/live_recognize.py
```

## Privacy and dataset policy

- This repo is configured to **not** commit local training data by default (`dataset/`).
- Trained model binaries are also ignored by default (`models/*`), except `models/labels.txt`.
- Recorded samples store landmarks/time metadata, not raw video frames.

## Notes

- If you change feature logic, keep training and live inference in sync (this is centralized in `src/features.py`).
- Current gesture classes are listed in `models/labels.txt`.

import os
import glob
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from features import motion_features

DATASET_DIR = "dataset"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

GESTURES = [
    "STOP_IDLE",
    "SWIPE_LEFT",
    "SWIPE_RIGHT",
    "PUSH",
    "PULL",
    "CIRCLE_CW",
    "CIRCLE_CCW",
]

HANDS = ["Left", "Right"]

def load_dataset():
    X, y_g, y_h = [], [], []
    paths = glob.glob(os.path.join(DATASET_DIR, "*", "*", "*.npz"))
    for p in paths:
        data = np.load(p, allow_pickle=True)
        seq = data["X"]  # (T,63)
        gesture = str(data["gesture"])
        hand = str(data["hand"])

        if gesture not in GESTURES or hand not in HANDS:
            continue

        f = motion_features(seq)
        if f is None:
            continue

        X.append(f)
        y_g.append(gesture)
        y_h.append(hand)

    X = np.stack(X, axis=0) if len(X) else np.zeros((0, 1), dtype=np.float32)
    return X, np.array(y_g), np.array(y_h)

def main():
    X, y_g, y_h = load_dataset()
    print(f"Loaded samples: {len(y_g)}")
    print("Gesture counts:", dict(Counter(y_g)))
    print("Hand counts:", dict(Counter(y_h)))

    if len(y_g) < 30:
        print("\nNot enough samples for a meaningful split yet (need ~30+).")
        print("But you can still continue recording and re-run.")
        return

    # Split stratified by gesture (keeps class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_g, test_size=0.25, random_state=42, stratify=y_g
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n=== Classification report (gesture) ===")
    print(classification_report(y_test, y_pred, digits=3))

    print("=== Confusion matrix (gesture order) ===")
    print(GESTURES)
    print(confusion_matrix(y_test, y_pred, labels=GESTURES))

    # Save model + metadata
    dump(clf, os.path.join(OUT_DIR, "gesture_rf.joblib"))
    with open(os.path.join(OUT_DIR, "labels.txt"), "w", encoding="utf-8") as f:
        for g in GESTURES:
            f.write(g + "\n")

    print(f"\nSaved model to: {os.path.join(OUT_DIR, 'gesture_rf.joblib')}")
    print("Saved labels to: models/labels.txt")

if __name__ == "__main__":
    main()

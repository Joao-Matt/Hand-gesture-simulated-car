import numpy as np


def normalize_landmarks(seq_63):
    """
    seq_63: (T,63) landmarks in normalized image coords.
    1) reshape to (T,21,3)
    2) subtract wrist position (landmark 0) => translation invariance
    3) divide by wrist->middle_mcp distance (landmark 9) => scale invariance
    """
    seq = seq_63.reshape(-1, 21, 3).copy()
    wrist = seq[:, 0:1, :]  # (T,1,3)
    seq -= wrist

    middle_mcp = seq[:, 9, :]  # (T,3)
    scale = np.linalg.norm(middle_mcp, axis=1)  # (T,)
    scale = np.maximum(scale, 1e-6)
    seq /= scale[:, None, None]
    return seq.reshape(-1, 63)


def wrist_xy(seq_63):
    seq = seq_63.reshape(-1, 21, 3)
    return seq[:, 0, 0:2]  # wrist x,y in normalized image coords


def wrist_path_features(seq_63):
    wxy = wrist_xy(seq_63)  # (T,2)
    v = np.diff(wxy, axis=0)  # (T-1,2)
    spd = np.linalg.norm(v, axis=1) + 1e-8

    # Direction unit vectors
    u = v / spd[:, None]  # (T-1,2)

    # Angle change between consecutive directions
    dot = np.sum(u[1:] * u[:-1], axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    ang = np.arccos(dot)  # (T-2,)

    # Straight swipe => low mean angle change
    # Circle => higher mean angle change
    return np.array([
        float(np.mean(ang)) if len(ang) else 0.0,
        float(np.max(ang)) if len(ang) else 0.0,
        float(np.mean(spd)),
        float(np.max(spd)),
    ], dtype=np.float32)


def signed_rotation_features(seq_63):
    """
    Encodes CW vs CCW using signed curvature of wrist path.
    Uses cross product of consecutive velocity vectors (2D).
    """
    wxy = wrist_xy(seq_63)  # (T,2)
    v = np.diff(wxy, axis=0)  # (T-1,2)

    if v.shape[0] < 3:
        return np.zeros((6,), dtype=np.float32)

    v1 = v[:-1]
    v2 = v[1:]

    # 2D cross-product scalar: v1_x*v2_y - v1_y*v2_x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]  # (T-2,)
    dot = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]  # (T-2,)

    # Normalize by step lengths so scale changes matter less
    n1 = np.linalg.norm(v1, axis=1) + 1e-8
    n2 = np.linalg.norm(v2, axis=1) + 1e-8
    sin_theta = cross / (n1 * n2)  # approx signed sin(angle)
    cos_theta = dot / (n1 * n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Signed angle change (radians)
    ang = np.arctan2(sin_theta, cos_theta)  # (T-2,)

    # Aggregates
    sum_ang = float(np.sum(ang))
    mean_ang = float(np.mean(ang))
    frac_pos = float(np.mean(ang > 0))
    frac_neg = float(np.mean(ang < 0))
    mean_abs = float(np.mean(np.abs(ang)))
    max_abs = float(np.max(np.abs(ang)))

    return np.array([sum_ang, mean_ang, frac_pos, frac_neg, mean_abs, max_abs], dtype=np.float32)


def motion_features(seq_63):
    """
    Turn a variable-length sequence into a fixed-length feature vector.
    """
    T = seq_63.shape[0]
    if T < 3:
        return None

    nseq = normalize_landmarks(seq_63)  # (T,63)
    vel = np.diff(nseq, axis=0)  # (T-1,63)

    feat = []
    feat += [np.mean(nseq, axis=0), np.std(nseq, axis=0)]
    feat += [np.mean(vel, axis=0), np.std(vel, axis=0)]

    vel_mag = np.linalg.norm(vel, axis=1)  # (T-1,)
    feat += [np.array([
        float(np.mean(vel_mag)),
        float(np.std(vel_mag)),
        float(np.max(vel_mag))
    ], dtype=np.float32)]

    feat += [wrist_path_features(seq_63)]
    feat += [signed_rotation_features(seq_63)]

    return np.concatenate(feat, axis=0).astype(np.float32)

import numpy as np
import pandas as pd
from ultralytics import YOLO

# =========================
# Configuration
# =========================

VIDEO_PATH = "C:\\Users\\pso14\\Desktop\\MLpy\\POSE\\red set\\5s_left_sample.mp4"
MODEL_NAME = "yolov8n-pose.pt"

IDX_SHOULDER = 5
IDX_HIP = 11
IDX_KNEE = 13
IDX_ANKLE = 15

# Runner direction: True if moving left -> right
RUNNER_LEFT_TO_RIGHT = True

# =========================
# Gait templates
# (ankle_angle, knee_angle, hip_flag)
# where hip_flag represents the position of the knee relative to the hips (point types from the back phase have hip_flag 0m the other have 1)
# =========================

POINT_TEMPLATES = {
    1: np.array([110, 160, 0]),  # Toe-Off
    2: np.array([130, 150, 0]),  # Peak Follow Through
    3: np.array([120,  95, 1]),  # Max Heel Recovery
    4: np.array([110, 155, 1]),  # Peak Knee Lift
    5: np.array([100, 150, 1]),  # Foot-Strike
}



def angle_between(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def compute_angles_and_flags(kpts):
    hip = kpts[IDX_HIP][:2]
    knee = kpts[IDX_KNEE][:2]
    ankle = kpts[IDX_ANKLE][:2]
    shoulder = kpts[IDX_SHOULDER][:2]

    # Angles
    knee_angle = angle_between(hip - knee, ankle - knee)
    foot_vec = np.array([1.0, 0.0])
    ankle_angle = angle_between(knee - ankle, foot_vec)

    # Front/back encoding
    if RUNNER_LEFT_TO_RIGHT:
        hip_flag = int(knee[0] < hip[0])
    else:
        hip_flag = int(knee[0] > hip[0])

    return ankle_angle, knee_angle, hip_flag


def classify_point_type(ankle_angle, knee_angle, hip_flag):
    best_type = 0
    best_dist = float("inf")

    for pt, ref in POINT_TEMPLATES.items():
        angle_dist = np.linalg.norm(
            np.array([ankle_angle, knee_angle]) - ref[:2]
        )
        hip_dist = 50 * abs(hip_flag - ref[2])

        dist = angle_dist + hip_dist

        if dist < best_dist:
            best_dist = dist
            best_type = pt

    return best_type


def extract_sequence(video_path):
    model = YOLO(MODEL_NAME)
    results = model.predict(source=video_path, stream=True, verbose=False)

    rows = []

    for frame_idx, result in enumerate(results):
        if result.keypoints is None:
            continue

        kpts_all = result.keypoints.data.cpu().numpy()
        if len(kpts_all) == 0:
            continue

        kpts = kpts_all[0]

        if np.any(kpts[IDX_ANKLE, :2] <= 0):
            continue

        ankle_angle, knee_angle, hip_flag = compute_angles_and_flags(kpts)
        point_type = classify_point_type(
            ankle_angle, knee_angle, hip_flag
        )

        x, y = kpts[IDX_ANKLE, :2]

        rows.append({
            "frame": frame_idx,
            "x": x,
            "y": y,
            "ankle_angle": ankle_angle,
            "knee_angle": knee_angle,
            "hip_flag": hip_flag,
            "point_type": point_type
        })

    return pd.DataFrame(rows)


def compute_average_footpath(df):
    footpath = {}

    for pt in range(1, 6):
        subset = df[df["point_type"] == pt]
        if subset.empty:
            footpath[pt] = (np.nan, np.nan)
        else:
            footpath[pt] = (
                subset["x"].mean(),
                subset["y"].mean()
            )

    return footpath


def footpath_to_df(footpath):
    return pd.DataFrame([
        {"point_type": pt, "avg_x": xy[0], "avg_y": xy[1]}
        for pt, xy in footpath.items()
    ])


if __name__ == "__main__":
    df = extract_sequence(VIDEO_PATH)

    df.to_csv("ankle_sequence_with_angles.csv", index=False)

    footpath = compute_average_footpath(df)
    footpath_df = footpath_to_df(footpath)
    footpath_df.to_csv("footpath_representation.csv", index=False)

    print("\nAveraged Footpath Representation:")
    print(footpath_df)
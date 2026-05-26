import cv2
import mediapipe as mp
from utils.hand.hand_physics import calculate_angle

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmark = mp.tasks.vision.PoseLandmark

MODEL_PATH = "models/pose_landmarker_lite.task"

# Arm landmark indices per side
_ARM_LANDMARKS = {
    "Left":  (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    "Right": (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
}

def create_pose_landmarker(model_path=MODEL_PATH, running_mode=VisionRunningMode.VIDEO):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=running_mode,
    )
    return PoseLandmarker.create_from_options(options)

def get_elbow_angle(pose_landmarks, handedness="Right"):
    """
    Calculates the elbow flexion angle for the arm matching the detected hand.
    Returns 0 (straight) to 180 (fully bent).
    """
    if not pose_landmarks:
        return 0

    side = "Left" if handedness == "Left" else "Right"
    shoulder_idx, elbow_idx, wrist_idx = _ARM_LANDMARKS[side]

    shoulder = pose_landmarks[shoulder_idx]
    elbow = pose_landmarks[elbow_idx]
    wrist = pose_landmarks[wrist_idx]

    raw_angle = calculate_angle(shoulder, elbow, wrist)
    return int(180 - raw_angle)

def draw_arm_landmarks(image, pose_landmarks, handedness="Right"):
    """Draws shoulder-elbow-wrist connections on the image."""
    if not pose_landmarks:
        return image

    h, w = image.shape[:2]
    side = "Left" if handedness == "Left" else "Right"
    shoulder_idx, elbow_idx, wrist_idx = _ARM_LANDMARKS[side]

    points = []
    for idx in (shoulder_idx, elbow_idx, wrist_idx):
        lm = pose_landmarks[idx]
        points.append((int(lm.x * w), int(lm.y * h)))

    for i in range(len(points) - 1):
        cv2.line(image, points[i], points[i + 1], (0, 255, 255), 3, cv2.LINE_AA)
    for pt in points:
        cv2.circle(image, pt, 6, (0, 200, 255), -1)

    return image

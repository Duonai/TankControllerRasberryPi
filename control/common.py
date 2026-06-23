import importlib
from collections import Counter, deque
from typing import Any, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:
    picamera2_module = importlib.import_module("picamera2")
except ImportError:
    picamera2_module = None


UPPER_BODY_LANDMARKS = {
    0,
    7,
    8,
    11,
    12,
    13,
    14,
    15,
    16,
    23,
    24,
}

UPPER_BODY_CONNECTIONS = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (0, 7),
    (0, 8),
)

FINGER_CHAINS = (
    (5, 6, 8),
    (9, 10, 12),
    (13, 14, 16),
    (17, 18, 20),
)


class CameraSource:
    def __init__(self, camera_id: int, width: int, height: int, fps: int) -> None:
        self.mode = "opencv"
        self.picam: Any = None
        self.cap: Optional[cv2.VideoCapture] = None

        if picamera2_module is not None:
            self.mode = "picamera2"
            self.picam = picamera2_module.Picamera2()
            config = self.picam.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"},
                controls={"FrameRate": fps},
            )
            self.picam.configure(config)
            self.picam.start()
            print("[INFO] Camera source: Picamera2")
            return

        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError("웹캠을 열 수 없습니다. /dev/video0 또는 카메라 연결 상태를 확인하세요.")

        print("[INFO] Camera source: OpenCV VideoCapture")

    def read_bgr(self) -> np.ndarray:
        if self.mode == "picamera2":
            assert self.picam is not None
            frame_rgb = self.picam.capture_array()
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        assert self.cap is not None
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("카메라 프레임을 읽지 못했습니다.")
        return frame

    def close(self) -> None:
        if self.picam is not None:
            self.picam.stop()
            self.picam.close()
        if self.cap is not None:
            self.cap.release()


class HandGestureTracker:
    def __init__(self) -> None:
        self.armed = {"left": False, "right": False}
        self.last_state = {"left": "unknown", "right": "unknown"}
        self.history = {
            "left": deque(maxlen=4),
            "right": deque(maxlen=4),
        }

    def smooth_state(self, side: str, state: str) -> str:
        if state == "unknown":
            self.history[side].clear()
            return state

        self.history[side].append(state)
        counts = Counter(self.history[side])
        return counts.most_common(1)[0][0]

    def update(self, side: str, state: str) -> Optional[str]:
        if state == "open":
            self.armed[side] = True

        triggered = None
        if state == "grip" and self.last_state[side] != "grip" and self.armed[side]:
            triggered = f"{side.upper()}_GRAB"
            self.armed[side] = False

        self.last_state[side] = state
        return triggered


def normalized_to_pixel(landmark, width: int, height: int) -> Tuple[int, int]:
    x = min(max(int(landmark.x * width), 0), width - 1)
    y = min(max(int(landmark.y * height), 0), height - 1)
    return x, y


def landmark_visible(landmark, threshold: float = 0.4) -> bool:
    visibility = getattr(landmark, "visibility", 1.0)
    presence = getattr(landmark, "presence", 1.0)
    return visibility >= threshold and presence >= threshold


def point_xy(landmark) -> np.ndarray:
    return np.asarray((landmark.x, landmark.y), dtype=np.float32)


def landmark_has_xy(landmark) -> bool:
    return bool(np.isfinite(landmark.x) and np.isfinite(landmark.y))


def angle_deg_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def palm_center_xy(hand_landmarks) -> np.ndarray:
    indices = (0, 5, 9, 13, 17)
    points = np.asarray([point_xy(hand_landmarks.landmark[idx]) for idx in indices], dtype=np.float32)
    return np.mean(points, axis=0)


def classify_hand_state(hand_landmarks) -> str:
    wrist = point_xy(hand_landmarks.landmark[0])
    palm_span = np.linalg.norm(point_xy(hand_landmarks.landmark[5]) - point_xy(hand_landmarks.landmark[17]))
    if palm_span < 1e-4:
        return "unknown"

    palm_center = palm_center_xy(hand_landmarks)
    fingertip_indices = (4, 8, 12, 16, 20)
    palm_tip_ratios = [
        float(np.linalg.norm(point_xy(hand_landmarks.landmark[idx]) - palm_center) / palm_span)
        for idx in fingertip_indices
    ]
    avg_palm_tip_ratio = float(np.mean(palm_tip_ratios))

    tip_ratios = []
    fold_ratios = []
    pip_ratios = []
    extended_votes = 0
    curled_votes = 0
    for mcp_idx, pip_idx, tip_idx in FINGER_CHAINS:
        mcp = point_xy(hand_landmarks.landmark[mcp_idx])
        pip = point_xy(hand_landmarks.landmark[pip_idx])
        tip = point_xy(hand_landmarks.landmark[tip_idx])
        pip_ratio = float(np.linalg.norm(pip - wrist) / palm_span)
        tip_ratio = float(np.linalg.norm(tip - wrist) / palm_span)
        fold_ratio = float(np.linalg.norm(tip - mcp) / palm_span)
        bend_angle = angle_deg_2d(mcp, pip, tip)

        pip_ratios.append(pip_ratio)
        tip_ratios.append(tip_ratio)
        fold_ratios.append(fold_ratio)

        if tip_ratio > 1.18 or bend_angle > 155.0:
            extended_votes += 1
        if tip_ratio < 1.02 or fold_ratio < 0.88 or bend_angle < 125.0:
            curled_votes += 1

    thumb_tip = point_xy(hand_landmarks.landmark[4])
    thumb_ip = point_xy(hand_landmarks.landmark[3])
    thumb_mcp = point_xy(hand_landmarks.landmark[2])
    thumb_ratio = float(np.linalg.norm(thumb_tip - wrist) / palm_span)
    thumb_fold_ratio = float(np.linalg.norm(thumb_tip - thumb_mcp) / palm_span)
    thumb_extended = angle_deg_2d(thumb_mcp, thumb_ip, thumb_tip) > 145.0 and thumb_ratio > 0.95

    avg_tip_ratio = float(np.mean(tip_ratios))
    avg_fold_ratio = float(np.mean(fold_ratios))
    avg_pip_ratio = float(np.mean(pip_ratios))

    if (extended_votes >= 3 and avg_tip_ratio > 1.08 and avg_palm_tip_ratio > 0.92) or (
        extended_votes >= 2 and avg_palm_tip_ratio > 1.02 and thumb_extended
    ):
        return "open"
    if curled_votes >= 3 and avg_palm_tip_ratio < 0.78 and avg_tip_ratio < 1.18:
        return "grip"
    if curled_votes >= 2 and avg_palm_tip_ratio < 0.86 and avg_pip_ratio < 1.04:
        return "grip"
    if avg_palm_tip_ratio < 0.82 and avg_fold_ratio < 1.02:
        return "grip"
    if thumb_ratio < 1.05 and avg_tip_ratio < 1.16 and avg_palm_tip_ratio < 0.9:
        return "grip"
    if avg_palm_tip_ratio > 0.96 and avg_tip_ratio > 1.02:
        return "open"
    if avg_palm_tip_ratio < 0.9 and (curled_votes >= 1 or thumb_fold_ratio < 1.02):
        return "grip"
    return "other"


def resolve_wrist_landmark(pose_landmarks, wrist_idx: int, hand_landmarks=None):
    wrist = pose_landmarks.landmark[wrist_idx]
    if hand_landmarks is not None and landmark_has_xy(hand_landmarks.landmark[0]):
        return hand_landmarks.landmark[0]
    return wrist


def draw_body_landmarks(frame: np.ndarray, pose_landmarks) -> None:
    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    drawing_utils.draw_landmarks(
        frame,
        pose_landmarks,
        mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
        connection_drawing_spec=drawing_utils.DrawingSpec(color=(80, 220, 255), thickness=2, circle_radius=2),
    )


def highlight_upper_body(frame: np.ndarray, pose_landmarks) -> None:
    height, width = frame.shape[:2]

    for start_idx, end_idx in UPPER_BODY_CONNECTIONS:
        start = pose_landmarks.landmark[start_idx]
        end = pose_landmarks.landmark[end_idx]
        if not (landmark_visible(start) and landmark_visible(end)):
            continue
        start_pt = normalized_to_pixel(start, width, height)
        end_pt = normalized_to_pixel(end, width, height)
        cv2.line(frame, start_pt, end_pt, (0, 200, 255), 2)

    for landmark_idx in UPPER_BODY_LANDMARKS:
        landmark = pose_landmarks.landmark[landmark_idx]
        if not landmark_visible(landmark):
            continue
        point = normalized_to_pixel(landmark, width, height)
        cv2.circle(frame, point, 5, (0, 255, 0), -1)


def draw_hand(frame: np.ndarray, hand_landmarks, connections, landmark_color, connection_color) -> None:
    height, width = frame.shape[:2]

    points = []
    for landmark in hand_landmarks.landmark:
        point = normalized_to_pixel(landmark, width, height)
        points.append(point)
        cv2.circle(frame, point, 3, landmark_color, -1)

    for start_idx, end_idx in connections:
        cv2.line(frame, points[start_idx], points[end_idx], connection_color, 2)
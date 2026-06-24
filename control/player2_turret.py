import argparse
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:
    from .common import CameraSource, HandGestureTracker, classify_hand_state, draw_body_landmarks, draw_hand, highlight_upper_body, landmark_has_xy, point_xy, resolve_wrist_landmark
except ImportError:
    from common import CameraSource, HandGestureTracker, classify_hand_state, draw_body_landmarks, draw_hand, highlight_upper_body, landmark_has_xy, point_xy, resolve_wrist_landmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2P tank turret controller using MediaPipe Pose + Hands")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip", action="store_true", help="Mirror the preview horizontally")
    parser.add_argument("--model-complexity", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--min-detection-conf", type=float, default=0.5)
    parser.add_argument("--min-tracking-conf", type=float, default=0.5)
    parser.add_argument("--yaw-deadzone-deg", type=float, default=12.0, help="Right forearm tilt deadzone in degrees")
    parser.add_argument("--yaw-full-scale-deg", type=float, default=55.0, help="Right forearm tilt that maps to full turret speed")
    parser.add_argument("--pitch-deadzone", type=float, default=0.12, help="Neutral band as a fraction of torso height")
    parser.add_argument("--pitch-full-scale", type=float, default=0.38, help="Left wrist travel for full barrel speed")
    parser.add_argument("--relaxed-ratio", type=float, default=0.42, help="Comfortable wrist height below shoulders")
    return parser.parse_args()


def body_metrics(pose_landmarks) -> Optional[Tuple[np.ndarray, float, float]]:
    left_shoulder = pose_landmarks.landmark[11]
    right_shoulder = pose_landmarks.landmark[12]
    left_hip = pose_landmarks.landmark[23]
    right_hip = pose_landmarks.landmark[24]
    points = (left_shoulder, right_shoulder, left_hip, right_hip)
    if not all(landmark_has_xy(point) for point in points):
        return None

    shoulder_center = np.asarray(
        ((left_shoulder.x + right_shoulder.x) * 0.5, (left_shoulder.y + right_shoulder.y) * 0.5),
        dtype=np.float32,
    )
    torso_height = max(float(abs(((left_hip.y + right_hip.y) * 0.5) - shoulder_center[1])), 1e-4)
    shoulder_width = max(float(abs(right_shoulder.x - left_shoulder.x)), 1e-4)
    return shoulder_center, torso_height, shoulder_width


def split_hands(hand_results):
    left_hand_landmarks = None
    right_hand_landmarks = None

    if hand_results.multi_hand_landmarks is None:
        return left_hand_landmarks, right_hand_landmarks

    handedness_list = hand_results.multi_handedness or []
    for index, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
        label = None
        if index < len(handedness_list) and handedness_list[index].classification:
            label = handedness_list[index].classification[0].label.lower()

        if label == "left":
            left_hand_landmarks = hand_landmarks
        elif label == "right":
            right_hand_landmarks = hand_landmarks

    if left_hand_landmarks is None or right_hand_landmarks is None:
        remaining = list(hand_results.multi_hand_landmarks)
        if left_hand_landmarks is not None:
            remaining = [hand for hand in remaining if hand is not left_hand_landmarks]
        if right_hand_landmarks is not None:
            remaining = [hand for hand in remaining if hand is not right_hand_landmarks]

        remaining.sort(key=lambda hand: hand.landmark[0].x)
        if left_hand_landmarks is None and remaining:
            left_hand_landmarks = remaining.pop(0)
        if right_hand_landmarks is None and remaining:
            right_hand_landmarks = remaining.pop(-1)

    return left_hand_landmarks, right_hand_landmarks


def quantize_axis(value: float, deadzone: float) -> Tuple[str, float]:
    if abs(value) <= deadzone:
        return "STOP", 0.0
    scaled = min((abs(value) - deadzone) / max(1.0 - deadzone, 1e-4), 1.0)
    return ("POS" if value > 0 else "NEG"), float(np.sign(value) * scaled)


def signal_to_axis_value(signal: float, deadzone: float, full_scale: float) -> float:
    if abs(signal) <= deadzone:
        return 0.0

    magnitude = min((abs(signal) - deadzone) / max(full_scale - deadzone, 1e-4), 1.0)
    return float(np.sign(signal) * magnitude)


def compute_forearm_yaw_deg(pose_landmarks, elbow_idx: int, wrist_idx: int, hand_landmarks=None) -> Optional[float]:
    elbow = pose_landmarks.landmark[elbow_idx]
    wrist = resolve_wrist_landmark(pose_landmarks, wrist_idx, hand_landmarks)
    if not landmark_has_xy(elbow) or not landmark_has_xy(wrist):
        return None

    dx = wrist.x - elbow.x
    dy = wrist.y - elbow.y
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None

    return float(np.degrees(np.arctan2(dx, max(abs(dy), 1e-6))))


def yaw_label(value: float) -> str:
    if value > 0.18:
        return "RIGHT"
    if value < -0.18:
        return "LEFT"
    return "STOP"


def pitch_label(value: float) -> str:
    if value > 0.18:
        return "UP"
    if value < -0.18:
        return "DOWN"
    return "STOP"


def draw_status(
    frame: np.ndarray,
    fps: float,
    yaw_value: float,
    yaw_deg: Optional[float],
    pitch_value: float,
    pitch_ref_y: Optional[float],
    fire_text: str,
    left_state: str,
    right_state: str,
) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    yaw_text = "Turret yaw: N/A" if yaw_deg is None else f"Turret yaw: {yaw_label(yaw_value)} {yaw_value:+.2f} ({yaw_deg:+.1f} deg)"
    pitch_text = f"Barrel pitch: {pitch_label(pitch_value)} {pitch_value:+.2f}"
    cv2.putText(frame, yaw_text, (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, pitch_text, (16, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"L hand: {left_state}", (16, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"R hand: {right_state}", (16, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, fire_text, (16, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Right wrist up/down = barrel", (16, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(frame, "Left forearm tilt = turret", (16, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press q or ESC to exit", (16, 266), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def draw_pitch_guides(frame: np.ndarray, pose_landmarks, shoulder_idx: int, relaxed_y: Optional[float], deadzone: float, torso_height: Optional[float]) -> None:
    if relaxed_y is None or torso_height is None:
        return

    height, width = frame.shape[:2]
    shoulder = pose_landmarks.landmark[shoulder_idx]
    if not landmark_has_xy(shoulder):
        return

    shoulder_x = int(shoulder.x * width)
    relaxed_line = int(relaxed_y * height)
    deadzone_px = int(deadzone * torso_height * height)
    x1 = max(shoulder_x - 70, 0)
    x2 = min(shoulder_x + 70, width - 1)
    cv2.line(frame, (x1, relaxed_line), (x2, relaxed_line), (255, 255, 255), 2)
    cv2.line(frame, (x1, relaxed_line - deadzone_px), (x2, relaxed_line - deadzone_px), (0, 220, 0), 1)
    cv2.line(frame, (x1, relaxed_line + deadzone_px), (x2, relaxed_line + deadzone_px), (0, 100, 255), 1)


def main() -> None:
    args = parse_args()
    camera = CameraSource(args.camera_id, args.width, args.height, args.fps)
    previous_tick = cv2.getTickCount()
    pose = mp.solutions.pose
    hands = mp.solutions.hands
    gesture_tracker = HandGestureTracker()
    fire_text = "FIRE: idle"
    fire_until = 0.0
    hand_model_complexity = 0 if args.model_complexity == 0 else 1

    with pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=args.min_detection_conf,
        min_tracking_confidence=args.min_tracking_conf,
    ) as pose_model, hands.Hands(
        static_image_mode=False,
        model_complexity=hand_model_complexity,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_conf,
        min_tracking_confidence=args.min_tracking_conf,
    ) as hand_model:
        try:
            while True:
                frame = camera.read_bgr()
                if args.flip:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                pose_results = pose_model.process(rgb)
                hand_results = hand_model.process(rgb)
                rgb.flags.writeable = True

                left_hand_landmarks, right_hand_landmarks = split_hands(hand_results)

                yaw_value = 0.0
                yaw_deg = None
                pitch_value = 0.0
                pitch_ref_y = None
                torso_height = None
                left_state = "missing"
                right_state = "missing"

                if pose_results.pose_landmarks is not None:
                    draw_body_landmarks(frame, pose_results.pose_landmarks)
                    highlight_upper_body(frame, pose_results.pose_landmarks)
                    metrics = body_metrics(pose_results.pose_landmarks)
                    if metrics is not None:
                        shoulder_center, torso_height, shoulder_width = metrics
                        right_wrist = resolve_wrist_landmark(pose_results.pose_landmarks, 16, right_hand_landmarks)
                        pitch_ref_y = float(shoulder_center[1] + args.relaxed_ratio * torso_height)

                        if landmark_has_xy(right_wrist):
                            right_signal = (pitch_ref_y - float(right_wrist.y)) / torso_height
                            pitch_value = signal_to_axis_value(right_signal, args.pitch_deadzone, args.pitch_full_scale)

                        yaw_deg = compute_forearm_yaw_deg(
                            pose_results.pose_landmarks,
                            13,
                            15,
                            left_hand_landmarks,
                        )
                        if yaw_deg is not None:
                            yaw_value = signal_to_axis_value(yaw_deg, args.yaw_deadzone_deg, args.yaw_full_scale_deg)
                        draw_pitch_guides(frame, pose_results.pose_landmarks, 12, pitch_ref_y, args.pitch_deadzone, torso_height)

                if left_hand_landmarks is not None:
                    draw_hand(frame, left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS, (255, 200, 0), (255, 120, 0))
                    left_state = gesture_tracker.smooth_state("left", classify_hand_state(left_hand_landmarks))
                    fire_command = gesture_tracker.update("left", left_state)
                    if fire_command is not None:
                        fire_text = f"FIRE: {fire_command}"
                        fire_until = time.monotonic() + 1.0

                if right_hand_landmarks is not None:
                    draw_hand(frame, right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS, (255, 0, 200), (180, 120, 255))
                    right_state = gesture_tracker.smooth_state("right", classify_hand_state(right_hand_landmarks))

                if time.monotonic() >= fire_until:
                    fire_text = "FIRE: idle"

                current_tick = cv2.getTickCount()
                elapsed = (current_tick - previous_tick) / cv2.getTickFrequency()
                previous_tick = current_tick
                fps = 1.0 / elapsed if elapsed > 0 else 0.0

                draw_status(
                    frame,
                    fps,
                    yaw_value,
                    yaw_deg,
                    pitch_value,
                    pitch_ref_y,
                    fire_text,
                    left_state,
                    right_state,
                )
                cv2.imshow("2P Turret Control", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
        finally:
            camera.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
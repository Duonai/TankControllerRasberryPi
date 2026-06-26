import argparse
import time
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:
    from .common import CameraSource, HandGestureTracker, classify_hand_state, draw_body_landmarks, draw_hand, highlight_upper_body, landmark_has_xy, point_xy, resolve_wrist_landmark
except ImportError:
    from common import CameraSource, HandGestureTracker, classify_hand_state, draw_body_landmarks, draw_hand, highlight_upper_body, landmark_has_xy, point_xy, resolve_wrist_landmark


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2P tank turret controller using MediaPipe Pose + Hands")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip", action="store_true", default=True, help="Mirror the preview horizontally (default: enabled)")
    parser.add_argument("--no-flip", dest="flip", action="store_false", help="Disable horizontal preview mirroring")
    parser.add_argument("--model-complexity", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--min-detection-conf", type=float, default=0.5)
    parser.add_argument("--min-tracking-conf", type=float, default=0.5)
    parser.add_argument("--yaw-neutral-offset-deg", type=float, default=-15.0, help="Neutral forearm tilt offset toward the torso")
    parser.add_argument("--yaw-deadzone-deg", type=float, default=18.0, help="Forearm tilt deadzone in degrees")
    parser.add_argument("--yaw-full-scale-deg", type=float, default=50.0, help="Forearm tilt that maps to full turret speed")
    parser.add_argument("--pitch-deadzone", type=float, default=0.15, help="Neutral band as a fraction of torso height")
    parser.add_argument("--pitch-full-scale", type=float, default=0.40, help="Wrist travel for full barrel speed")
    parser.add_argument("--relaxed-ratio", type=float, default=0.42, help="Comfortable wrist height below shoulders")
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def serialize_turret_result(
    yaw_value: float,
    yaw_deg: Optional[float],
    pitch_value: float,
    pitch_ref_y: Optional[float],
    fire_text: str,
    left_state: str,
    right_state: str,
    has_pose: bool,
) -> Dict[str, Any]:
    fire_command = "idle"
    if ":" in fire_text:
        fire_command = fire_text.split(":", 1)[1].strip()

    return {
        "has_pose": has_pose,
        "yaw_value": round(yaw_value, 4),
        "yaw_deg": None if yaw_deg is None else round(yaw_deg, 2),
        "yaw_label": yaw_label(yaw_value),
        "pitch_value": round(pitch_value, 4),
        "pitch_ref_y": None if pitch_ref_y is None else round(pitch_ref_y, 4),
        "pitch_label": pitch_label(pitch_value),
        "fire": fire_command,
        "left_hand_state": left_state,
        "right_hand_state": right_state,
    }


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


def wrist_distance_sq(hand_landmarks, wrist_landmark) -> float:
    hand_wrist = hand_landmarks.landmark[0]
    dx = float(hand_wrist.x - wrist_landmark.x)
    dy = float(hand_wrist.y - wrist_landmark.y)
    return dx * dx + dy * dy


def split_hands(hand_results, pose_landmarks=None):
    left_hand_landmarks = None
    right_hand_landmarks = None

    if hand_results.multi_hand_landmarks is None:
        return left_hand_landmarks, right_hand_landmarks

    if pose_landmarks is not None:
        left_pose_wrist = pose_landmarks.landmark[15]
        right_pose_wrist = pose_landmarks.landmark[16]
        if landmark_has_xy(left_pose_wrist) and landmark_has_xy(right_pose_wrist):
            hands_with_costs = []
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hands_with_costs.append(
                    (
                        hand_landmarks,
                        wrist_distance_sq(hand_landmarks, left_pose_wrist),
                        wrist_distance_sq(hand_landmarks, right_pose_wrist),
                    )
                )

            if len(hands_with_costs) == 1:
                hand_landmarks, left_cost, right_cost = hands_with_costs[0]
                if left_cost <= right_cost:
                    left_hand_landmarks = hand_landmarks
                else:
                    right_hand_landmarks = hand_landmarks
                return left_hand_landmarks, right_hand_landmarks

            best_total = None
            best_pair = (None, None)
            for left_index, (left_candidate, left_cost, _) in enumerate(hands_with_costs):
                for right_index, (right_candidate, _, right_cost) in enumerate(hands_with_costs):
                    if left_index == right_index:
                        continue
                    total_cost = left_cost + right_cost
                    if best_total is None or total_cost < best_total:
                        best_total = total_cost
                        best_pair = (left_candidate, right_candidate)

            if best_pair[0] is not None or best_pair[1] is not None:
                return best_pair

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


def compute_forearm_yaw_deg(
    pose_landmarks,
    elbow_idx: int,
    wrist_idx: int,
    hand_landmarks=None,
    neutral_offset_deg: float = 0.0,
) -> Optional[float]:
    elbow = pose_landmarks.landmark[elbow_idx]
    wrist = resolve_wrist_landmark(pose_landmarks, wrist_idx, hand_landmarks)
    if not landmark_has_xy(elbow) or not landmark_has_xy(wrist):
        return None

    dx = wrist.x - elbow.x
    dy = wrist.y - elbow.y
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None

    raw_yaw_deg = float(np.degrees(np.arctan2(dx, -dy)))
    return raw_yaw_deg - neutral_offset_deg


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
    cv2.putText(frame, "Left arm up/down = barrel", (16, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(frame, "Right arm forearm tilt = turret/fire", (16, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press q or ESC to exit", (16, 266), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def draw_pitch_guides(frame: np.ndarray, anchor_x: Optional[float], relaxed_y: Optional[float], deadzone: float, torso_height: Optional[float]) -> None:
    if relaxed_y is None or torso_height is None:
        return

    if anchor_x is None:
        return

    height, width = frame.shape[:2]
    shoulder_x = int(anchor_x * width)
    relaxed_line = int(relaxed_y * height)
    deadzone_px = int(deadzone * torso_height * height)
    x1 = max(shoulder_x - 70, 0)
    x2 = min(shoulder_x + 70, width - 1)
    cv2.line(frame, (x1, relaxed_line), (x2, relaxed_line), (255, 255, 255), 2)
    cv2.line(frame, (x1, relaxed_line - deadzone_px), (x2, relaxed_line - deadzone_px), (0, 220, 0), 1)
    cv2.line(frame, (x1, relaxed_line + deadzone_px), (x2, relaxed_line + deadzone_px), (0, 100, 255), 1)


def draw_yaw_guides(
    frame: np.ndarray,
    pose_landmarks,
    elbow_idx: int,
    wrist_idx: int,
    neutral_offset_deg: float,
    deadzone_deg: float,
) -> None:
    elbow = pose_landmarks.landmark[elbow_idx]
    wrist = pose_landmarks.landmark[wrist_idx]
    if not landmark_has_xy(elbow) or not landmark_has_xy(wrist):
        return

    height, width = frame.shape[:2]
    elbow_px = np.array((int(elbow.x * width), int(elbow.y * height)), dtype=np.int32)
    wrist_px = np.array((int(wrist.x * width), int(wrist.y * height)), dtype=np.int32)
    forearm_length = max(int(np.linalg.norm(wrist_px - elbow_px)), 36)
    guide_radius = min(max(int(forearm_length * 0.75), 42), 90)
    guide_color = (255, 80, 80)
    boundary_color = (255, 0, 255)
    neutral_rad = np.radians(neutral_offset_deg)

    cv2.circle(frame, tuple(elbow_px), guide_radius, guide_color, 1)
    cv2.line(
        frame,
        tuple(elbow_px),
        (
            int(elbow_px[0] + np.sin(neutral_rad) * guide_radius),
            int(elbow_px[1] - np.cos(neutral_rad) * guide_radius),
        ),
        guide_color,
        1,
    )

    for angle_deg, color in ((neutral_offset_deg - deadzone_deg, boundary_color), (neutral_offset_deg + deadzone_deg, boundary_color)):
        angle_rad = np.radians(angle_deg)
        end_x = int(elbow_px[0] + np.sin(angle_rad) * guide_radius)
        end_y = int(elbow_px[1] - np.cos(angle_rad) * guide_radius)
        cv2.line(frame, tuple(elbow_px), (end_x, end_y), color, 2)

    cv2.putText(
        frame,
        f"NEUTRAL {neutral_offset_deg:.0f} deg  STOP +/-{deadzone_deg:.0f}",
        (int(elbow_px[0] - guide_radius), max(int(elbow_px[1] - guide_radius - 8), 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        boundary_color,
        1,
        cv2.LINE_AA,
    )


def run_controller(args: argparse.Namespace, on_result: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
    camera = CameraSource(args.camera_id, args.width, args.height, args.fps)
    previous_tick = cv2.getTickCount()
    frame_id = 0
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

                left_hand_landmarks, right_hand_landmarks = split_hands(hand_results, pose_results.pose_landmarks)

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
                        left_wrist = pose_results.pose_landmarks.landmark[16]
                        right_wrist = pose_results.pose_landmarks.landmark[15]
                        pitch_ref_y = float(shoulder_center[1] + args.relaxed_ratio * torso_height)
                        pitch_anchor_x = None

                        if landmark_has_xy(left_wrist):
                            pitch_anchor_x = float(left_wrist.x)
                            left_signal = (pitch_ref_y - float(left_wrist.y)) / torso_height
                            pitch_value = signal_to_axis_value(left_signal, args.pitch_deadzone, args.pitch_full_scale)
                        else:
                            left_shoulder = pose_results.pose_landmarks.landmark[12]
                            if landmark_has_xy(left_shoulder):
                                pitch_anchor_x = float(left_shoulder.x)

                        yaw_deg = compute_forearm_yaw_deg(
                            pose_results.pose_landmarks,
                            13,
                            15,
                            None,
                            args.yaw_neutral_offset_deg,
                        )
                        if yaw_deg is not None:
                            yaw_value = signal_to_axis_value(yaw_deg, args.yaw_deadzone_deg, args.yaw_full_scale_deg)
                        draw_pitch_guides(frame, pitch_anchor_x, pitch_ref_y, args.pitch_deadzone, torso_height)
                        draw_yaw_guides(frame, pose_results.pose_landmarks, 13, 15, args.yaw_neutral_offset_deg, args.yaw_deadzone_deg)

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
                frame_id += 1

                if on_result is not None:
                    on_result(
                        {
                            "frame_id": frame_id,
                            "fps": round(fps, 2),
                            "result": serialize_turret_result(
                                yaw_value=yaw_value,
                                yaw_deg=yaw_deg,
                                pitch_value=pitch_value,
                                pitch_ref_y=pitch_ref_y,
                                fire_text=fire_text,
                                left_state=left_state,
                                right_state=right_state,
                                has_pose=pose_results.pose_landmarks is not None,
                            ),
                        }
                    )

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


def main() -> None:
    args = parse_args()
    run_controller(args)


if __name__ == "__main__":
    main()
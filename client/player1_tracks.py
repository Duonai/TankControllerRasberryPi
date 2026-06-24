import argparse
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:
    from .common import CameraSource, draw_body_landmarks, highlight_upper_body, landmark_has_xy, normalized_to_pixel, resolve_wrist_landmark
except ImportError:
    from common import CameraSource, draw_body_landmarks, highlight_upper_body, landmark_has_xy, normalized_to_pixel, resolve_wrist_landmark


@dataclass
class TrackOutput:
    left_value: float
    right_value: float
    left_label: str
    right_label: str
    drive_label: str
    torso_height: float
    relaxed_y: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="1P tank track controller using MediaPipe Pose")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip", action="store_true", help="Mirror the preview horizontally")
    parser.add_argument("--model-complexity", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--min-detection-conf", type=float, default=0.5)
    parser.add_argument("--min-tracking-conf", type=float, default=0.5)
    parser.add_argument("--deadzone", type=float, default=0.12, help="Neutral band as a fraction of torso height")
    parser.add_argument("--full-scale", type=float, default=0.42, help="Hand movement for full track command")
    parser.add_argument("--relaxed-ratio", type=float, default=0.42, help="Comfortable wrist height below shoulders")
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def serialize_track_result(output: Optional[TrackOutput]) -> Dict[str, Any]:
    result = {
        "has_pose": output is not None,
        "left_value": 0.0,
        "right_value": 0.0,
        "left_label": "STOP",
        "right_label": "STOP",
        "drive_label": "IDLE",
    }

    if output is not None:
        result.update(
            {
                "left_value": round(output.left_value, 4),
                "right_value": round(output.right_value, 4),
                "left_label": output.left_label,
                "right_label": output.right_label,
                "drive_label": output.drive_label,
            }
        )

    return result


def compute_body_metrics(pose_landmarks) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
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
    hip_center = np.asarray(
        ((left_hip.x + right_hip.x) * 0.5, (left_hip.y + right_hip.y) * 0.5),
        dtype=np.float32,
    )
    torso_height = max(float(abs(hip_center[1] - shoulder_center[1])), 1e-4)
    shoulder_width = max(float(abs(right_shoulder.x - left_shoulder.x)), 1e-4)
    return shoulder_center, hip_center, torso_height, shoulder_width


def signal_to_track_value(signal: float, deadzone: float, full_scale: float) -> float:
    if abs(signal) <= deadzone:
        return 0.0

    magnitude = min((abs(signal) - deadzone) / max(full_scale - deadzone, 1e-4), 1.0)
    return float(np.sign(signal) * magnitude)


def label_from_value(value: float) -> str:
    if value > 0.18:
        return "FWD"
    if value < -0.18:
        return "REV"
    return "STOP"


def drive_label(left_value: float, right_value: float) -> str:
    threshold = 0.18
    avg = 0.5 * (left_value + right_value)
    diff = left_value - right_value

    if abs(left_value) < threshold and abs(right_value) < threshold:
        return "IDLE"
    if left_value > threshold and right_value > threshold:
        return "FORWARD"
    if left_value < -threshold and right_value < -threshold:
        return "BACKWARD"
    if left_value > threshold and right_value < -threshold:
        return "PIVOT_RIGHT"
    if left_value < -threshold and right_value > threshold:
        return "PIVOT_LEFT"
    if avg >= 0.0:
        return "ARC_RIGHT" if diff > 0.12 else "ARC_LEFT"
    return "REVERSE_RIGHT" if diff > 0.12 else "REVERSE_LEFT"


def compute_track_output(pose_landmarks, left_hand_landmarks, right_hand_landmarks, args: argparse.Namespace) -> Optional[TrackOutput]:
    metrics = compute_body_metrics(pose_landmarks)
    if metrics is None:
        return None

    shoulder_center, _hip_center, torso_height, _shoulder_width = metrics
    left_wrist = resolve_wrist_landmark(pose_landmarks, 15, left_hand_landmarks)
    right_wrist = resolve_wrist_landmark(pose_landmarks, 16, right_hand_landmarks)

    if not landmark_has_xy(left_wrist) or not landmark_has_xy(right_wrist):
        return None

    relaxed_y = float(shoulder_center[1] + args.relaxed_ratio * torso_height)
    left_signal = (relaxed_y - float(left_wrist.y)) / torso_height
    right_signal = (relaxed_y - float(right_wrist.y)) / torso_height

    left_value = signal_to_track_value(left_signal, args.deadzone, args.full_scale)
    right_value = signal_to_track_value(right_signal, args.deadzone, args.full_scale)
    left_label = label_from_value(left_value)
    right_label = label_from_value(right_value)

    return TrackOutput(
        left_value=left_value,
        right_value=right_value,
        left_label=left_label,
        right_label=right_label,
        drive_label=drive_label(left_value, right_value),
        torso_height=torso_height,
        relaxed_y=relaxed_y,
    )


def draw_track_guides(frame: np.ndarray, pose_landmarks, output: TrackOutput, deadzone: float) -> None:
    height, width = frame.shape[:2]
    left_shoulder = normalized_to_pixel(pose_landmarks.landmark[11], width, height)
    right_shoulder = normalized_to_pixel(pose_landmarks.landmark[12], width, height)

    relaxed_line = int(output.relaxed_y * height)
    deadzone_px = int(deadzone * output.torso_height * height)

    left_x1 = max(left_shoulder[0] - 70, 0)
    left_x2 = min(left_shoulder[0] + 70, width - 1)
    right_x1 = max(right_shoulder[0] - 70, 0)
    right_x2 = min(right_shoulder[0] + 70, width - 1)

    for x1, x2 in ((left_x1, left_x2), (right_x1, right_x2)):
        cv2.line(frame, (x1, relaxed_line), (x2, relaxed_line), (255, 255, 255), 2)
        cv2.line(frame, (x1, relaxed_line - deadzone_px), (x2, relaxed_line - deadzone_px), (0, 220, 0), 1)
        cv2.line(frame, (x1, relaxed_line + deadzone_px), (x2, relaxed_line + deadzone_px), (0, 100, 255), 1)


def draw_status(frame: np.ndarray, fps: float, output: Optional[TrackOutput]) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    if output is None:
        cv2.putText(frame, "Pose: missing", (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"Drive: {output.drive_label}", (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"L track: {output.left_label} {output.left_value:+.2f}", (16, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"R track: {output.right_label} {output.right_value:+.2f}", (16, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2, cv2.LINE_AA)
        cv2.putText(frame, "Hands near chest = stop", (16, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Raise slightly = forward, lower slightly = reverse", (16, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 2, cv2.LINE_AA)

    cv2.putText(frame, "Press q or ESC to exit", (16, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def run_controller(args: argparse.Namespace, on_result: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
    camera = CameraSource(args.camera_id, args.width, args.height, args.fps)
    previous_tick = cv2.getTickCount()
    frame_id = 0
    pose = mp.solutions.pose

    with pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=args.min_detection_conf,
        min_tracking_confidence=args.min_tracking_conf,
    ) as model:
        try:
            while True:
                frame = camera.read_bgr()
                if args.flip:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = model.process(rgb)
                rgb.flags.writeable = True

                output = None
                if results.pose_landmarks is not None:
                    draw_body_landmarks(frame, results.pose_landmarks)
                    highlight_upper_body(frame, results.pose_landmarks)
                    output = compute_track_output(
                        results.pose_landmarks,
                        None,
                        None,
                        args,
                    )
                    if output is not None:
                        draw_track_guides(frame, results.pose_landmarks, output, args.deadzone)

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
                            "result": serialize_track_result(output),
                        }
                    )

                draw_status(frame, fps, output)
                cv2.imshow("1P Tank Tracks", frame)

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
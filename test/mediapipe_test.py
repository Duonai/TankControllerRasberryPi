import argparse
import importlib
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:
	picamera2_module = importlib.import_module("picamera2")
except ImportError:
	picamera2_module = None


UPPER_BODY_LANDMARKS = {
	0,   # nose
	7,   # left ear
	8,   # right ear
	11,  # left shoulder
	12,  # right shoulder
	13,  # left elbow
	14,  # right elbow
	15,  # left wrist
	16,  # right wrist
	23,  # left hip
	24,  # right hip
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

DEFAULT_MODEL_CANDIDATES = (
	"pose_landmarker.task",
	"pose_landmarker_lite.task",
	"pose_landmarker_full.task",
	"pose_landmarker_heavy.task",
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


def parse_args() -> argparse.Namespace:
	default_model = str(Path(__file__).with_name(DEFAULT_MODEL_CANDIDATES[0]))

	parser = argparse.ArgumentParser(
		description="MediaPipe Pose Landmarker upper-body demo for Raspberry Pi"
	)
	parser.add_argument(
		"--model",
		default=str(default_model),
		help="Path to pose_landmarker.task",
	)
	parser.add_argument("--camera-id", type=int, default=0)
	parser.add_argument("--width", type=int, default=640)
	parser.add_argument("--height", type=int, default=480)
	parser.add_argument("--fps", type=int, default=30)
	parser.add_argument("--min-conf", type=float, default=0.5)
	parser.add_argument("--flip", action="store_true", help="Mirror the preview horizontally")
	return parser.parse_args()


def build_landmarker(model_path: str, min_conf: float):
	BaseOptions = mp.tasks.BaseOptions
	PoseLandmarker = mp.tasks.vision.PoseLandmarker
	PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
	RunningMode = mp.tasks.vision.RunningMode

	options = PoseLandmarkerOptions(
		base_options=BaseOptions(model_asset_path=model_path),
		running_mode=RunningMode.IMAGE,
		num_poses=1,
		min_pose_detection_confidence=min_conf,
		min_pose_presence_confidence=min_conf,
		min_tracking_confidence=min_conf,
		output_segmentation_masks=False,
	)
	return PoseLandmarker.create_from_options(options)


def normalized_to_pixel(landmark, width: int, height: int) -> Tuple[int, int]:
	x = min(max(int(landmark.x * width), 0), width - 1)
	y = min(max(int(landmark.y * height), 0), height - 1)
	return x, y


def landmark_visible(landmark, threshold: float = 0.4) -> bool:
	visibility = getattr(landmark, "visibility", 1.0)
	presence = getattr(landmark, "presence", 1.0)
	return visibility >= threshold and presence >= threshold


def angle_deg(a: Iterable[float], b: Iterable[float], c: Iterable[float]) -> float:
	a_vec = np.asarray(tuple(a), dtype=np.float32)
	b_vec = np.asarray(tuple(b), dtype=np.float32)
	c_vec = np.asarray(tuple(c), dtype=np.float32)
	ba = a_vec - b_vec
	bc = c_vec - b_vec
	denom = np.linalg.norm(ba) * np.linalg.norm(bc)
	if denom < 1e-6:
		return 0.0
	cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
	return float(np.degrees(np.arccos(cosine)))


def draw_upper_body(frame: np.ndarray, pose_landmarks) -> None:
	height, width = frame.shape[:2]

	for start_idx, end_idx in UPPER_BODY_CONNECTIONS:
		start = pose_landmarks[start_idx]
		end = pose_landmarks[end_idx]
		if not (landmark_visible(start) and landmark_visible(end)):
			continue
		start_pt = normalized_to_pixel(start, width, height)
		end_pt = normalized_to_pixel(end, width, height)
		cv2.line(frame, start_pt, end_pt, (0, 200, 255), 2)

	for landmark_idx in UPPER_BODY_LANDMARKS:
		landmark = pose_landmarks[landmark_idx]
		if not landmark_visible(landmark):
			continue
		point = normalized_to_pixel(landmark, width, height)
		cv2.circle(frame, point, 5, (0, 255, 0), -1)


def annotate_elbow_angle(
	frame: np.ndarray,
	pose_landmarks,
	shoulder_idx: int,
	elbow_idx: int,
	wrist_idx: int,
	label: str,
	color: Tuple[int, int, int],
) -> Optional[float]:
	height, width = frame.shape[:2]
	shoulder = pose_landmarks[shoulder_idx]
	elbow = pose_landmarks[elbow_idx]
	wrist = pose_landmarks[wrist_idx]

	if not all(landmark_visible(point) for point in (shoulder, elbow, wrist)):
		return None

	angle = angle_deg(
		(shoulder.x, shoulder.y, shoulder.z),
		(elbow.x, elbow.y, elbow.z),
		(wrist.x, wrist.y, wrist.z),
	)

	elbow_point = normalized_to_pixel(elbow, width, height)
	text_point = (elbow_point[0] + 10, max(elbow_point[1] - 10, 20))
	cv2.putText(
		frame,
		f"{label}: {angle:.1f}",
		text_point,
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		color,
		2,
		cv2.LINE_AA,
	)
	return angle


def put_status(frame: np.ndarray, fps: float, left_angle: Optional[float], right_angle: Optional[float]) -> None:
	cv2.putText(
		frame,
		f"FPS: {fps:.1f}",
		(16, 30),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.8,
		(255, 255, 0),
		2,
		cv2.LINE_AA,
	)

	left_text = "L elbow: N/A" if left_angle is None else f"L elbow: {left_angle:.1f}"
	right_text = "R elbow: N/A" if right_angle is None else f"R elbow: {right_angle:.1f}"

	cv2.putText(
		frame,
		left_text,
		(16, 62),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		(0, 255, 120),
		2,
		cv2.LINE_AA,
	)
	cv2.putText(
		frame,
		right_text,
		(16, 94),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		(0, 180, 255),
		2,
		cv2.LINE_AA,
	)
	cv2.putText(
		frame,
		"Press q or ESC to exit",
		(16, 126),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.65,
		(255, 255, 255),
		2,
		cv2.LINE_AA,
	)


def main() -> None:
	args = parse_args()
	model_path = Path(args.model)

	if not model_path.exists():
		script_dir = Path(__file__).resolve().parent
		detected_model = next(
			(script_dir / name for name in DEFAULT_MODEL_CANDIDATES if (script_dir / name).exists()),
			None,
		)
		if detected_model is not None:
			model_path = detected_model

	if not model_path.exists():
		raise FileNotFoundError(
			f"모델 파일을 찾을 수 없습니다: {model_path}\n"
			"pose_landmarker.task 또는 pose_landmarker_lite/full/heavy.task 파일을 이 스크립트와 같은 폴더에 두거나 --model로 경로를 지정하세요."
		)

	camera = CameraSource(args.camera_id, args.width, args.height, args.fps)
	previous_tick = cv2.getTickCount()

	with build_landmarker(str(model_path), args.min_conf) as landmarker:
		try:
			while True:
				frame = camera.read_bgr()
				if args.flip:
					frame = cv2.flip(frame, 1)

				rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
				result = landmarker.detect(mp_image)

				left_angle = None
				right_angle = None

				if result.pose_landmarks:
					pose_landmarks = result.pose_landmarks[0]
					draw_upper_body(frame, pose_landmarks)
					left_angle = annotate_elbow_angle(
						frame,
						pose_landmarks,
						11,
						13,
						15,
						"L",
						(0, 255, 120),
					)
					right_angle = annotate_elbow_angle(
						frame,
						pose_landmarks,
						12,
						14,
						16,
						"R",
						(0, 180, 255),
					)

				current_tick = cv2.getTickCount()
				elapsed = (current_tick - previous_tick) / cv2.getTickFrequency()
				previous_tick = current_tick
				fps = 1.0 / elapsed if elapsed > 0 else 0.0

				put_status(frame, fps, left_angle, right_angle)
				cv2.imshow("MediaPipe Upper Body", frame)

				key = cv2.waitKey(1) & 0xFF
				if key in (27, ord("q")):
					break
		finally:
			camera.close()
			cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

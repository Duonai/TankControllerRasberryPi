# 2026-06-23 Copilot 작업 맥락

## 작업 개요
- Raspberry Pi 환경에서 MediaPipe 기반 웹캠 추론 코드 작성 및 조정.
- 초기 목표는 상체 body landmark 검출과 팔 각도 표시.
- 이후 Holistic 기반으로 body landmark + hand landmark를 함께 사용하는 테스트 코드로 확장.
- 손 동작은 별도 학습 모델 없이 landmark 간 거리/각도 계산 규칙으로 구현.

## 생성/수정된 주요 파일
- `/home/willtek/main_ws/mediapipe_test.py`
  - Pose Landmarker 기반 상체 landmark 표시.
  - 좌/우 팔꿈치 각도 표시.
- `/home/willtek/main_ws/mediapipe_test_world.py`
  - `pose_world_landmarks` 기준 3D elbow angle 테스트용.
- `/home/willtek/TankControllerRasberryPi/mediapipe_holistic_test.py`
  - Holistic 기반 body + hand landmark 통합 테스트 코드.
  - 현재 주 작업 파일.

## 모델 관련 메모
- Pose Landmarker 모델 다운로드 경로 확인 완료.
- 사용 가능한 모델 예시:
  - `pose_landmarker_lite.task`
  - `pose_landmarker_full.task`
  - `pose_landmarker_heavy.task`
- Raspberry Pi에서는 보통 `lite` 또는 `full`부터 테스트.

## Holistic 코드 진행 내용
### 1. Body landmark 표시
- 처음에는 상체 일부만 수동으로 그려서 body landmark가 잘 안 보였음.
- 이후 MediaPipe pose skeleton 전체를 그리도록 수정.
- 상체 landmark는 별도로 강조 표시하도록 유지.

### 2. 손 landmark 표시
- Holistic의 `left_hand_landmarks`, `right_hand_landmarks`를 사용해 양손 21개 포인트와 연결선 표시.

### 3. 손 동작 분류
- 목표 상태는 `OPEN`, `GRIP`, `OTHER`.
- 초기에는 finger angle + wrist distance 위주 규칙이었으나 손바닥/손등 방향 편차가 커서 반복 조정.
- 이후 `palm center` 대비 fingertip 거리 기반 규칙으로 변경.
- 현재는 다음 요소를 조합:
  - palm center 대비 fingertip 평균 거리
  - wrist 대비 fingertip 거리
  - MCP 대비 fingertip 거리
  - finger bend angle
- `HandGestureTracker`로 상태를 짧게 smoothing.
- `open -> grip` 상태 전이가 발생하면 `LEFT_GRAB` 또는 `RIGHT_GRAB` 명령 트리거.
- 손 분류 로직은 사용자가 “여기서 종료”라고 한 상태이며, 더 미세 조정은 이후 필요 시 진행.

### 4. 전완 각도 표시
- 처음에는 오른팔만 표시.
- 화면상 2D 기준으로 전완(`elbow -> wrist`) 각도 계산.
- 기준:
  - 수평이면 `0도`
  - 수직이면 `90도`
- pose wrist 대신 hand wrist landmark를 우선 사용하도록 변경해 `N/A` 감소 시도.
- pose visibility gate 제거로 상체만 보여도 계산 가능하도록 수정.
- 이후 왼팔도 같은 방식으로 추가.
- 현재 오버레이 항목:
  - `L forearm 2D`
  - `R forearm 2D`
  - `L hand`
  - `R hand`
  - `CMD`

## 파일/폴더 이동 이력
- 작업 초반 파일 위치는 `/home/willtek/main_ws/` 중심.
- 이후 프로젝트 폴더를 `/home/willtek/TankControllerRasberryPi/`로 사용.
- 현재 Holistic 관련 주 파일은:
  - `/home/willtek/TankControllerRasberryPi/mediapipe_holistic_test.py`

## 실행/검증 메모
- 문법 검증은 반복적으로 아래 방식으로 수행:
  - `python -m py_compile mediapipe_holistic_test.py`
- 실행은 보통 아래 명령 사용:
  - `python mediapipe_holistic_test.py --flip`
- 종료 코드 `130`은 사용자가 `Ctrl+C` 또는 종료 키로 끈 경우로 보임.

## 사용자가 중요하게 본 요구사항
- body landmark는 실제로 화면에서 잘 보여야 함.
- 손 동작은 손바닥/손등 방향 모두에서 동작해야 함.
- `GRIP`, `OPEN`, `OTHER` 외 복잡한 분류는 현재 불필요.
- 팔 각도는 3D 관절 굽힘이 아니라 화면상 2D 전완 방향 각도가 중요.
- 상태 텍스트는 왼쪽 위에 모아서 보기 쉽게 표시.

## 현재 상태 요약
- Holistic 통합 코드는 동작하는 상태로 유지.
- body + hand landmark 표시 기능 있음.
- hand state 및 `GRAB` 트리거 로직 포함.
- 좌/우 전완 2D 각도 표시 로직 포함.
- 추가 조정이 필요하면 앞으로는 `mediapipe_holistic_test.py`를 기준으로 이어서 작업하면 됨.

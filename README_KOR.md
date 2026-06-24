# TankControllerRasberryPi

MediaPipe 랜드마크를 활용한 Raspberry Pi용 카메라 기반 탱크 제어 프로젝트입니다.

이 저장소는 역할을 분리한 런타임 아키텍처를 사용합니다:

- 제어 모듈: 추론 및 제어 연산만 담당
- 런처 스크립트: 런타임 진입점 및 네트워크 스트리밍 담당
- PC 서버: 두 Raspberry Pi 노드의 결과를 수신하는 멀티 클라이언트 서버

## 파일 구조

- `client/player1_tracks.py`: 1P 본체 및 무한궤도 제어 모듈
- `client/player2_turret.py`: 2P 포탑 및 발사 제어 모듈
- `client/common.py`: MediaPipe 카메라 및 랜드마크 공통 헬퍼
- `client/result_transport.py`: Pi → PC 결과 스트리밍용 JSON-over-TCP 헬퍼
- `client/runtime_stream.py`: 런타임 설정 로더 및 재연결 송신기
- `test/mediapipe_holistic_test.py`: holistic 랜드마크 테스트베드
- `test/mediapipe_test.py`: pose 랜드마커 상체 테스트
- `test/mediapipe_test_world.py`: pose world 랜드마크 팔꿈치 각도 테스트
- `server/pc_result_server.py`: Raspberry Pi로부터 추론 결과를 수신하는 PC 측 TCP 서버
- `run_pc_server.py`: PC 서버 런처 (config/profile 기반)
- `run_rpi1_tracks.py`: player1 무한궤도 + 스트리밍 런처
- `run_rpi2_turret.py`: player2 포탑 + 스트리밍 런처
- `config/runtime_config.json`: IP/포트/프로필/디바이스 런타임 설정
- `contexts/*.md`: 작업 노트

## 참고 사항

- 이 프로젝트는 Raspberry Pi 웹캠 또는 Picamera2 입력에 맞게 조정되어 있습니다.
- MediaPipe 모델 파일은 저장소에 포함되지 않습니다. 필요 시 직접 다운로드하세요.
- `mediapipe_*.py`는 레거시/테스트 스크립트로, 일반 런타임에서는 불필요합니다.

## 런타임 아키텍처

각 노드를 각자의 머신에서 독립적으로 실행합니다:

1. **PC**: TCP 서버(`pc_result_server.py`)를 실행하여 두 클라이언트의 결과를 수신합니다.
2. **Raspberry Pi #1**: 무한궤도 컨트롤러(`run_rpi1_tracks.py`)를 실행하고 결과를 스트리밍합니다.
3. **Raspberry Pi #2**: 포탑 컨트롤러(`run_rpi2_turret.py`)를 실행하고 결과를 스트리밍합니다.

각 송신자는 `role` 필드(`player1_tracks` / `player2_turret`)를 포함한 줄바꿈 구분 JSON을 전송합니다.

설계 원칙:

- `client/player1_tracks.py`와 `client/player2_turret.py`는 추론/제어 연산에만 집중합니다.
- 소켓 연결 및 전송은 런처 레이어(`run_rpi1_tracks.py`, `run_rpi2_turret.py`)에서 처리합니다.
- IP/포트/디바이스 설정은 `config/runtime_config.json`에서 관리합니다.

## PC 서버 실행

권장 방법 (config/profile 기반):

```bash
python run_pc_server.py
```

다른 네트워크 프로필 사용:

```bash
python run_pc_server.py --profile alternate
```

핵심 모듈 직접 실행:

```bash
python server/pc_result_server.py --host 192.168.0.12 --port 5000
```

대체 네트워크:

```bash
python server/pc_result_server.py --host 10.56.130.242 --port 5000
```

`WinError 10049` 등 bind 실패 시 확인 사항:

- 선택한 호스트 IP가 현재 PC의 네트워크 어댑터에 실제로 존재하는지 확인
- 방화벽에서 해당 포트의 인바운드 TCP를 허용하는지 확인
- 동일한 포트를 이미 사용 중인 프로세스가 없는지 확인

## Raspberry Pi #1 (무한궤도)

`config/runtime_config.json` 기본값 사용:

```bash
python run_rpi1_tracks.py
```

다른 네트워크 프로필 사용:

```bash
python run_rpi1_tracks.py --profile alternate
```

네트워크 설정은 config에서 가져오되 컨트롤러 옵션만 오버라이드:

```bash
python run_rpi1_tracks.py --profile alternate --camera-id 0 --width 640 --height 480 --flip
```

## Raspberry Pi #2 (포탑)

`config/runtime_config.json` 기본값 사용:

```bash
python run_rpi2_turret.py
```

다른 네트워크 프로필 사용:

```bash
python run_rpi2_turret.py --profile alternate
```

네트워크 설정은 config에서 가져오되 컨트롤러 옵션만 오버라이드:

```bash
python run_rpi2_turret.py --profile alternate --camera-id 0 --width 640 --height 480 --flip
```

`config/runtime_config.json` 예시:

```json
{
    "network_profiles": {
        "primary": { "pc_host": "192.168.0.12", "pc_port": 5000, "pc_bind_host": "0.0.0.0" },
        "alternate": { "pc_host": "10.56.130.242", "pc_port": 5000, "pc_bind_host": "0.0.0.0" }
    },
    "nodes": {
        "pc_server": { "profile": "primary", "bind_host": "0.0.0.0", "bind_port": 5000 },
        "player1_tracks": {
            "profile": "primary",
            "device_id": "rpi1",
            "send_interval": 0.05,
            "use_fake_signal": false,
            "fake_signal_interval": 0.05
        },
        "player2_turret": {
            "profile": "primary",
            "device_id": "rpi2",
            "send_interval": 0.05,
            "use_fake_signal": false,
            "fake_signal_interval": 0.05
        }
    }
}
```

팁:

- 각 Pi가 PC에 접근하는 경로가 다를 경우, `nodes.player1_tracks.profile`과 `nodes.player2_turret.profile`을 독립적으로 설정하세요.
- `--profile`은 임시 오버라이드 용도로만 사용하세요.
- 카메라 추론 없이 빈 페이로드만 스트리밍하려면 해당 노드의 `use_fake_signal`을 `true`로 설정하세요.

## Fake Signal 모드

카메라나 MediaPipe 없이 네트워크 연결 및 수신 측만 테스트하고 싶을 때 사용합니다.

1. `config/runtime_config.json`에서 `nodes.player1_tracks.use_fake_signal` 또는 `nodes.player2_turret.use_fake_signal`을 `true`로 설정합니다.
2. 동일한 런처 명령어(`run_rpi1_tracks.py` 또는 `run_rpi2_turret.py`)를 실행합니다.
3. 런처가 `fake_signal_interval` 간격으로 기본값(0 또는 `"STOP"`)이 채워진 동일한 스키마의 페이로드를 전송합니다.

# TankControllerRasberryPi

Camera-based tank control project for Raspberry Pi using MediaPipe landmarks.

This repository now uses a separated runtime architecture:

- control modules: inference/control computation only
- launcher scripts: runtime entry and network streaming
- PC server: multi-client result receiver (2 Raspberry Pi nodes)

## Current Structure

- `client/player1_tracks.py`: 1P base and track control prototype
- `client/player2_turret.py`: 2P turret and firing control prototype
- `client/common.py`: shared MediaPipe camera and landmark helpers
- `client/result_transport.py`: JSON-over-TCP helpers for Pi-to-PC result streaming
- `client/runtime_stream.py`: runtime config loader and resilient sender
- `test/mediapipe_holistic_test.py`: holistic landmark testbed
- `test/mediapipe_test.py`: pose landmarker upper-body test
- `test/mediapipe_test_world.py`: pose world-landmark elbow-angle test
- `server/pc_result_server.py`: PC-side TCP server that receives inference results from Raspberry Pi
- `run_pc_server.py`: launcher for PC server (config/profile based)
- `run_rpi1_tracks.py`: launcher for player1 tracks + streaming
- `run_rpi2_turret.py`: launcher for player2 turret + streaming
- `config/runtime_config.json`: IP/port/profile/device runtime settings
- `contexts/*.md`: working notes

## Notes

- The project is tuned for Raspberry Pi webcam or Picamera2 input.
- MediaPipe model files are intentionally not tracked. Download them locally when needed.
- `mediapipe_*.py` are kept as legacy/test scripts and are not required in normal runtime.

## Runtime Architecture

Run each node independently on its own machine:

1. PC: run TCP server (`pc_result_server.py`) to receive both clients.
2. Raspberry Pi #1: run tracks controller (`run_rpi1_tracks.py`) and stream output.
3. Raspberry Pi #2: run turret controller (`run_rpi2_turret.py`) and stream output.

Each sender pushes newline-delimited JSON with a `role` field (`player1_tracks` / `player2_turret`).

Design note:

- `client/player1_tracks.py` and `client/player2_turret.py` focus on inference/control computation.
- Socket connection and sending are handled in launcher layer (`run_rpi1_tracks.py`, `run_rpi2_turret.py`).
- IP/port/device settings are managed in `config/runtime_config.json`.

## PC Server

Recommended (config/profile based):

```bash
python run_pc_server.py
```

Use alternate network profile:

```bash
python run_pc_server.py --profile alternate
```

Direct run (core module):

```bash
python server/pc_result_server.py --host 192.168.0.12 --port 5000
```

Alternate network:

```bash
python server/pc_result_server.py --host 10.56.130.242 --port 5000
```

If bind fails with `WinError 10049` or similar, check that:

- the selected host IP actually exists on the current PC network adapter
- firewall allows inbound TCP for the selected port
- no other process is already using the same port

## Raspberry Pi #1 (Tracks)

Use `config/runtime_config.json` defaults (profile from config):

```bash
python run_rpi1_tracks.py
```

Use alternate network profile:

```bash
python run_rpi1_tracks.py --profile alternate
```

Override controller options while keeping network from config:

```bash
python run_rpi1_tracks.py --profile alternate --camera-id 0 --width 640 --height 480 --flip
```

## Raspberry Pi #2 (Turret)

Use `config/runtime_config.json` defaults (profile from config):

```bash
python run_rpi2_turret.py
```

Use alternate network profile:

```bash
python run_rpi2_turret.py --profile alternate
```

Override controller options while keeping network from config:

```bash
python run_rpi2_turret.py --profile alternate --camera-id 0 --width 640 --height 480 --flip
```

`config/runtime_config.json` example:

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

Tip:

- set `nodes.player1_tracks.profile` and `nodes.player2_turret.profile` independently if each Pi uses a different route to the PC
- use `--profile` only for temporary override
- set `use_fake_signal: true` for a node when you want to stream zero-filled payloads without camera inference

Fake signal mode:

1. In `config/runtime_config.json`, set `nodes.player1_tracks.use_fake_signal` or `nodes.player2_turret.use_fake_signal` to `true`.
2. Run the same launcher command (`run_rpi1_tracks.py` or `run_rpi2_turret.py`).
3. The launcher sends the same schema with zero/default values at `fake_signal_interval`.
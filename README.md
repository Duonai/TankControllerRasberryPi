# TankControllerRasberryPi

Camera-based tank control experiments for Raspberry Pi using MediaPipe landmarks.

## Current Structure

- `control/player1_tracks.py`: 1P base and track control prototype
- `control/player2_turret.py`: 2P turret and firing control prototype
- `control/common.py`: shared MediaPipe camera and landmark helpers
- `mediapipe_holistic_test.py`: holistic landmark testbed
- `mediapipe_test.py`: pose landmarker upper-body test
- `mediapipe_test_world.py`: pose world-landmark elbow-angle test
- `contexts/0623copilot.md`: working notes from the current setup session

## Notes

- The project is tuned for Raspberry Pi webcam or Picamera2 input.
- MediaPipe model files are intentionally not tracked. Download them locally when needed.
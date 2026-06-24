import argparse
import time

from client.player1_tracks import build_arg_parser, run_controller
from client.runtime_stream import ResilientJsonSender, load_runtime_config, resolve_sender_config


def build_fake_track_result() -> dict:
    return {
        "has_pose": True,
        "left_value": 0.0,
        "right_value": 0.0,
        "left_label": "STOP",
        "right_label": "STOP",
        "drive_label": "IDLE",
    }


def parse_launcher_args() -> tuple[argparse.Namespace, argparse.Namespace]:
    launcher = argparse.ArgumentParser(description="Run player1 tracks with external JSON-configured streaming.")
    launcher.add_argument("--config", default="config/runtime_config.json", help="Path to runtime JSON config")
    launcher.add_argument("--profile", default="", help="Override network profile name")
    launcher_args, remaining = launcher.parse_known_args()

    controller_args = build_arg_parser().parse_args(remaining)
    return launcher_args, controller_args


def main() -> None:
    launcher_args, controller_args = parse_launcher_args()
    config = load_runtime_config(launcher_args.config)
    sender_conf = resolve_sender_config(config, "player1_tracks", launcher_args.profile)

    sender = ResilientJsonSender(
        host=sender_conf["host"],
        port=sender_conf["port"],
        role="player1_tracks",
        device_id=sender_conf["device_id"],
        send_interval=sender_conf["send_interval"],
    )

    print(
        "[RUN] player1_tracks "
        f"profile={sender_conf['profile']} -> {sender_conf['host']}:{sender_conf['port']} "
        f"device_id={sender_conf['device_id']} fake={sender_conf['use_fake_signal']}"
    )

    try:
        if sender_conf["use_fake_signal"]:
            frame_id = 0
            interval = sender_conf["fake_signal_interval"]
            fake_fps = 1.0 / interval if interval > 0 else 0.0
            print(f"[RUN] player1_tracks fake signal mode interval={interval:.3f}s")
            while True:
                frame_id += 1
                sender.send_result(frame_id, fake_fps, build_fake_track_result())
                time.sleep(interval)
        else:
            run_controller(
                controller_args,
                on_result=lambda packet: sender.send_result(packet["frame_id"], packet["fps"], packet["result"]),
            )
    except KeyboardInterrupt:
        print("[RUN] player1_tracks stopped")
    finally:
        sender.close()


if __name__ == "__main__":
    main()

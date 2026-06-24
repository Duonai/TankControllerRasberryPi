import argparse
import time

from client.player2_turret import build_arg_parser, run_controller
from client.runtime_stream import ResilientJsonSender, load_runtime_config, resolve_sender_config


def build_fake_turret_result() -> dict:
    return {
        "has_pose": True,
        "yaw_value": 0.0,
        "yaw_deg": 0.0,
        "yaw_label": "STOP",
        "pitch_value": 0.0,
        "pitch_ref_y": None,
        "pitch_label": "STOP",
        "fire": "idle",
        "left_hand_state": "missing",
        "right_hand_state": "missing",
    }


def parse_launcher_args() -> tuple[argparse.Namespace, argparse.Namespace]:
    launcher = argparse.ArgumentParser(description="Run player2 turret with external JSON-configured streaming.")
    launcher.add_argument("--config", default="config/runtime_config.json", help="Path to runtime JSON config")
    launcher.add_argument("--profile", default="", help="Override network profile name")
    launcher_args, remaining = launcher.parse_known_args()

    controller_args = build_arg_parser().parse_args(remaining)
    return launcher_args, controller_args


def main() -> None:
    launcher_args, controller_args = parse_launcher_args()
    config = load_runtime_config(launcher_args.config)
    sender_conf = resolve_sender_config(config, "player2_turret", launcher_args.profile)

    sender = ResilientJsonSender(
        host=sender_conf["host"],
        port=sender_conf["port"],
        role="player2_turret",
        device_id=sender_conf["device_id"],
        send_interval=sender_conf["send_interval"],
    )

    print(
        "[RUN] player2_turret "
        f"profile={sender_conf['profile']} -> {sender_conf['host']}:{sender_conf['port']} "
        f"device_id={sender_conf['device_id']} fake={sender_conf['use_fake_signal']}"
    )

    try:
        if sender_conf["use_fake_signal"]:
            frame_id = 0
            interval = sender_conf["fake_signal_interval"]
            fake_fps = 1.0 / interval if interval > 0 else 0.0
            print(f"[RUN] player2_turret fake signal mode interval={interval:.3f}s")
            while True:
                frame_id += 1
                sender.send_result(frame_id, fake_fps, build_fake_turret_result())
                time.sleep(interval)
        else:
            run_controller(
                controller_args,
                on_result=lambda packet: sender.send_result(packet["frame_id"], packet["fps"], packet["result"]),
            )
    except KeyboardInterrupt:
        print("[RUN] player2_turret stopped")
    finally:
        sender.close()


if __name__ == "__main__":
    main()

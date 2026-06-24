import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

try:
    from .result_transport import create_client_socket, send_json_line
except ImportError:
    from result_transport import create_client_socket, send_json_line


def load_runtime_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_sender_config(config: Dict[str, Any], node_name: str, profile_override: str = "") -> Dict[str, Any]:
    profiles = config.get("network_profiles", {})
    nodes = config.get("nodes", {})
    node = nodes.get(node_name, {})

    profile_name = profile_override or node.get("profile", "primary")
    profile = profiles.get(profile_name)
    if not profile:
        raise ValueError(f"Unknown network profile: {profile_name}")

    host = profile.get("pc_host")
    port = int(node.get("server_port", profile.get("pc_port", 5000)))
    device_id = str(node.get("device_id", node_name))
    send_interval = float(node.get("send_interval", 0.05))
    use_fake_signal = bool(node.get("use_fake_signal", False))
    fake_signal_interval = float(node.get("fake_signal_interval", send_interval))

    if not host:
        raise ValueError(f"Missing pc_host in profile: {profile_name}")

    return {
        "host": host,
        "port": port,
        "device_id": device_id,
        "send_interval": max(send_interval, 0.01),
        "use_fake_signal": use_fake_signal,
        "fake_signal_interval": max(fake_signal_interval, 0.01),
        "profile": profile_name,
    }


def resolve_pc_server_config(config: Dict[str, Any], profile_override: str = "") -> Dict[str, Any]:
    profiles = config.get("network_profiles", {})
    nodes = config.get("nodes", {})
    node = nodes.get("pc_server", {})

    profile_name = profile_override or node.get("profile", "primary")
    profile = profiles.get(profile_name)
    if not profile:
        raise ValueError(f"Unknown network profile: {profile_name}")

    bind_host = str(node.get("bind_host", profile.get("pc_bind_host", "0.0.0.0")))
    bind_port = int(node.get("bind_port", profile.get("pc_port", 5000)))

    return {
        "host": bind_host,
        "port": bind_port,
        "profile": profile_name,
    }


class ResilientJsonSender:
    def __init__(self, host: str, port: int, role: str, device_id: str, send_interval: float) -> None:
        self.host = host
        self.port = port
        self.role = role
        self.device_id = device_id
        self.send_interval = max(send_interval, 0.01)

        self.socket = None
        self.next_send_at = 0.0
        self.next_retry_at = 0.0

    def close(self) -> None:
        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                pass
            self.socket = None

    def _connect_if_needed(self, now: float) -> None:
        if self.socket is not None:
            return
        if now < self.next_retry_at:
            return

        try:
            sock = create_client_socket(self.host, self.port, timeout=2.0)
            sock.settimeout(0.2)
            self.socket = sock
            print(f"[NET] Connected to PC {self.host}:{self.port} ({self.role}/{self.device_id})")
        except OSError as exc:
            print(f"[NET] Connect failed: {exc}")
            self.next_retry_at = now + 1.0

    def send_result(self, frame_id: int, fps: float, result: Dict[str, Any]) -> None:
        now = time.monotonic()
        if now < self.next_send_at:
            return

        self._connect_if_needed(now)
        if self.socket is None:
            return

        payload = {
            "role": self.role,
            "device_id": self.device_id,
            "frame_id": frame_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fps": round(float(fps), 2),
            "result": result,
        }

        try:
            send_json_line(self.socket, payload)
        except (ConnectionError, OSError, socket.timeout) as exc:
            print(f"[NET] Send failed: {exc}")
            self.close()
            self.next_retry_at = now + 1.0
        finally:
            self.next_send_at = now + self.send_interval

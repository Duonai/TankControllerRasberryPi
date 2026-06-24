import argparse
import socket
import threading
from datetime import datetime
from pprint import pformat
from typing import Any, Dict, Tuple

from client.result_transport import JsonLineStream, create_server_socket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive inference results from multiple Raspberry Pi clients over TCP.")
    parser.add_argument("--host", default="0.0.0.0", help="PC bind IP address")
    parser.add_argument("--port", type=int, default=5000, help="PC bind port")
    return parser.parse_args()


def client_label(payload: Dict[str, Any], address: Tuple[str, int]) -> str:
    role = str(payload.get("role", "unknown"))
    device_id = str(payload.get("device_id", "unknown"))
    return f"{role}/{device_id}@{address[0]}:{address[1]}"


def handle_client(
    connection: socket.socket,
    address: Tuple[str, int],
    latest_by_role: Dict[str, Dict[str, Any]],
    lock: threading.Lock,
) -> None:
    stream = JsonLineStream(connection)
    endpoint = f"{address[0]}:{address[1]}"
    print(f"[SERVER] Client connected: {endpoint}")

    try:
        while True:
            payload = stream.recv_json_line()
            if payload is None:
                print(f"[SERVER] Client disconnected: {endpoint}")
                return

            role = str(payload.get("role", "unknown"))
            received_at = datetime.now().isoformat(timespec="seconds")

            with lock:
                latest_by_role[role] = payload
                snapshot = {
                    key: {
                        "device_id": value.get("device_id"),
                        "frame_id": value.get("frame_id"),
                        "timestamp": value.get("timestamp"),
                    }
                    for key, value in latest_by_role.items()
                }

            print(
                f"[SERVER] {received_at} {client_label(payload, address)}\n"
                f"{pformat(payload, sort_dicts=False)}\n"
                f"[SERVER] Latest snapshot by role: {pformat(snapshot, sort_dicts=False)}"
            )
    except (ConnectionError, OSError, ValueError) as exc:
        print(f"[SERVER] Client error {endpoint}: {exc}")
    finally:
        connection.close()


def main() -> None:
    args = parse_args()

    run_server(args.host, args.port)


def run_server(host: str, port: int) -> None:
    latest_by_role: Dict[str, Dict[str, Any]] = {}
    lock = threading.Lock()

    with create_server_socket(host, port) as server_socket:
        # Keep accept() interruptible on Windows so Ctrl+C can stop promptly.
        server_socket.settimeout(1.0)
        print(f"[SERVER] Listening on {host}:{port}")
        try:
            while True:
                try:
                    connection, address = server_socket.accept()
                except socket.timeout:
                    continue

                thread = threading.Thread(
                    target=handle_client,
                    args=(connection, address, latest_by_role, lock),
                    daemon=True,
                )
                thread.start()
        except KeyboardInterrupt:
            print("\n[SERVER] Shutdown requested by user (Ctrl+C)")


if __name__ == "__main__":
    main()
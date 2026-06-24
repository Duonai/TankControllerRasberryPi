import json
import socket
from typing import Any, Dict, Optional


DEFAULT_BUFFER_SIZE = 4096


def create_server_socket(host: str, port: int, backlog: int = 5) -> socket.socket:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(backlog)
    return server_socket


def create_client_socket(server_host: str, port: int, timeout: float = 5.0) -> socket.socket:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(timeout)
    client_socket.connect((server_host, port))
    return client_socket


def send_json_line(sock: socket.socket, payload: Dict[str, Any]) -> None:
    message = json.dumps(payload, ensure_ascii=True).encode("utf-8") + b"\n"
    sock.sendall(message)


def recv_json_line(sock: socket.socket, buffer_size: int = DEFAULT_BUFFER_SIZE) -> Optional[Dict[str, Any]]:
    stream = JsonLineStream(sock, buffer_size=buffer_size)
    return stream.recv_json_line()


class JsonLineStream:
    def __init__(self, sock: socket.socket, buffer_size: int = DEFAULT_BUFFER_SIZE) -> None:
        self.sock = sock
        self.buffer = bytearray()
        self.buffer_size = buffer_size

    def recv_json_line(self) -> Optional[Dict[str, Any]]:
        while True:
            newline_index = self.buffer.find(b"\n")
            if newline_index >= 0:
                line = bytes(self.buffer[:newline_index])
                del self.buffer[: newline_index + 1]
                if not line:
                    continue
                return json.loads(line.decode("utf-8"))

            data = self.sock.recv(self.buffer_size)
            if not data:
                if not self.buffer:
                    return None

                line = bytes(self.buffer)
                self.buffer.clear()
                if not line:
                    return None
                return json.loads(line.decode("utf-8"))

            self.buffer.extend(data)
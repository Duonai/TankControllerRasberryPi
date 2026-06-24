import argparse

from client.runtime_stream import load_runtime_config, resolve_pc_server_config
from server.pc_result_server import run_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PC server with runtime JSON config.")
    parser.add_argument("--config", default="config/runtime_config.json", help="Path to runtime JSON config")
    parser.add_argument("--profile", default="", help="Override network profile name")
    parser.add_argument("--host", default="", help="Optional bind host override")
    parser.add_argument("--port", type=int, default=0, help="Optional bind port override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.config)
    server_conf = resolve_pc_server_config(config, args.profile)

    host = args.host or server_conf["host"]
    port = args.port or server_conf["port"]

    print(f"[RUN] pc_server profile={server_conf['profile']} bind={host}:{port}")
    run_server(host, port)


if __name__ == "__main__":
    main()

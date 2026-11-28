"""CLI entry point: python -m DNDSR.Cauchy.monitor."""

from __future__ import annotations

import argparse

from .app import serve_hub


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cauchy-hub",
        description="Multi-platform web monitor and live steering hub for DNDSR Cauchy runs.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument(
        "--port",
        type=int,
        default=53100,
        help="Starting port for auto-discovery (default: 53100; 0 uses an OS ephemeral port)",
    )
    parser.add_argument("--scan-count", type=int, default=100, help="Sequential ports to scan from --port")
    parser.add_argument(
        "--watch",
        action="append",
        default=[],
        help="Directory to scan recursively for *-control.json and *-run-manifest.json (repeatable)",
    )
    parser.add_argument("--registry-dir", default=None, help="Override run-registry directory (~/.dndsr/runs)")
    parser.add_argument("--state-file", default=None, help="Override persistent hub catalog JSON path")
    parser.add_argument("--auth-token", default=None, help="Optional hub access token when exposing beyond loopback")
    args = parser.parse_args(argv)
    serve_hub(
        host=args.host,
        port=args.port,
        scan_count=args.scan_count,
        watch_dirs=args.watch,
        registry_dir=args.registry_dir,
        state_file=args.state_file,
        auth_token=args.auth_token,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

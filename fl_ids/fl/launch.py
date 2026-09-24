"""Starting real multi-process Flower runs reliably (components 4, 8, 11).

Clients started before the server's gRPC port is listening get
"Connection refused", and this Flower version's legacy gRPC client exits
on that rather than retrying -- the server then waits for them forever.
The server takes longer to start than a client (it imports LightGBM for
the boosting broadcast/update), so launching everything at once lost this
race intermittently. Launchers call `wait_for_server` between starting
the server and starting clients.
"""

from __future__ import annotations

import socket
import subprocess
import time

SERVER_STARTUP_TIMEOUT_SECONDS = 60.0
_POLL_SECONDS = 0.2


def wait_for_server(
    server_address: str,
    server_proc: subprocess.Popen | None = None,
    timeout: float = SERVER_STARTUP_TIMEOUT_SECONDS,
) -> None:
    """Block until `server_address` accepts TCP connections.

    Args:
        server_address: `"host:port"` the Flower server binds.
        server_proc: The server process, so an early exit fails fast
            instead of waiting out the timeout.
        timeout: Seconds to wait.

    Raises:
        RuntimeError: If the server exits, or isn't listening within `timeout`.
    """
    host, port = server_address.rsplit(":", 1)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server_proc is not None and server_proc.poll() is not None:
            raise RuntimeError(f"Flower server exited with code {server_proc.returncode} before listening")
        try:
            with socket.create_connection((host, int(port)), timeout=1.0):
                return
        except OSError:
            time.sleep(_POLL_SECONDS)
    raise RuntimeError(f"Flower server at {server_address} wasn't listening within {timeout}s")

"""Socket server lifecycle for MCP extension."""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, Callable, Dict

import omni


class SocketServer:
    """TCP socket server that schedules command execution on Kit main thread."""

    def __init__(self, host: str, port: int, on_command: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        self._host = host
        self._port = port
        self._on_command = on_command
        self._running = False
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self._host, self._port))
        self._socket.listen(1)
        self._thread = threading.Thread(target=self._server_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _server_loop(self) -> None:
        assert self._socket is not None
        self._socket.settimeout(1.0)
        while self._running:
            try:
                client, _ = self._socket.accept()
            except socket.timeout:
                continue
            except Exception:
                time.sleep(0.2)
                continue
            t = threading.Thread(target=self._handle_client, args=(client,), daemon=True)
            t.start()

    def _handle_client(self, client: socket.socket) -> None:
        buffer = b""
        try:
            while self._running:
                data = client.recv(16384)
                if not data:
                    break
                buffer += data
                try:
                    command = json.loads(buffer.decode("utf-8"))
                    buffer = b""
                except json.JSONDecodeError:
                    continue
                response = self._run_on_main_thread(command)
                try:
                    client.sendall(json.dumps(response).encode("utf-8"))
                except Exception:
                    break
        finally:
            try:
                client.close()
            except Exception:
                pass

    def _run_on_main_thread(self, command: Dict[str, Any]) -> Dict[str, Any]:
        result_holder: list[Dict[str, Any] | None] = [None]
        done = threading.Event()
        sub_holder = [None]

        def _callback(_event: Any) -> None:
            try:
                result_holder[0] = self._on_command(command)
            except Exception as exc:
                result_holder[0] = {"status": "error", "message": str(exc)}
            finally:
                done.set()
                if sub_holder[0] is not None:
                    try:
                        sub_holder[0].unsubscribe()
                    except Exception:
                        pass

        stream = omni.kit.app.get_app().get_update_event_stream()
        sub_holder[0] = stream.create_subscription_to_pop(_callback, name="mcp_execute_command")
        done.wait(timeout=60.0)
        if sub_holder[0] is not None:
            try:
                sub_holder[0].unsubscribe()
            except Exception:
                pass
        if result_holder[0] is None:
            return {"status": "error", "message": "Command execution timed out"}
        return result_holder[0]

"""Command dispatcher with explicit response contract."""

from __future__ import annotations

from typing import Any, Callable, Dict


Handler = Callable[..., Dict[str, Any]]


class CommandDispatcher:
    """Maps command names to handlers and validates handler responses."""

    def __init__(self) -> None:
        self._handlers: Dict[str, Handler] = {}

    def register(self, command: str, handler: Handler) -> None:
        self._handlers[command] = handler

    def dispatch(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        handler = self._handlers.get(command)
        if handler is None:
            return {"status": "error", "message": f"Unknown command type: {command}"}
        try:
            result = handler(**params)
            if not isinstance(result, dict) or "status" not in result:
                return {
                    "status": "error",
                    "message": f"Handler '{command}' returned invalid response contract",
                }
            return result
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

"""FlashAI Server - REST API and webhook server."""

from flashai.server.api import create_app, run_server

__all__ = ["create_app", "run_server"]

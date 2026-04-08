"""
server/app.py
=============
Entry point required by OpenEnv spec.
Imports and exposes the FastAPI app from main.py.
"""
from server.main import app

__all__ = ["app"]
"""Launch the PikMatrix AI Inference Dashboard Web Server.

Run this script directly:
    .\\.venv\\Scripts\\python.exe run_server.py
"""

import asyncio
import sys

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

import uvicorn

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("PIKMATRIX AI SPORTS INFERENCE DASHBOARD")
    print("Serving on: http://127.0.0.1:8000")
    print("=" * 80)

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=["app", "templates"],
        reload_excludes=["*.db", "*.db-wal", "*.db-shm", "*.log", "*.png", "*.jpg", ".venv", "scratch", "captures"],
    )

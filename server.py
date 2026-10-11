"""
server.py — Application Server Entry Point
==========================================
Launches the FastAPI backend and RupeeLens web user interface.

Usage:
    python server.py
"""

import sys
import uvicorn

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.app import app, get_local_ip

if __name__ == "__main__":
    local_ip = get_local_ip()
    port = 8000

    print("\n" + "=" * 65)
    print(" 🚀 RUPEELENS INDIAN CURRENCY RECOGNITION SERVER")
    print("=" * 65)
    print(f"  Local Desktop Access : http://localhost:{port}")
    print(f"  Mobile Phone Access   : http://{local_ip}:{port}")
    print(f"  API Docs (Swagger)   : http://localhost:{port}/docs")
    print("-" * 65)
    print("  To access from your mobile phone:")
    print("  1. Ensure phone and PC are connected to the same Wi-Fi network.")
    print(f"  2. Open browser on your phone and go to: http://{local_ip}:{port}")
    print("=" * 65 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port)

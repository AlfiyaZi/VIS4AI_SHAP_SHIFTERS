"""Convenience launcher for the Pair A interactive dashboard."""

import sys
from pathlib import Path

# Add dashboard folder to sys.path
dash_dir = Path(__file__).resolve().parent / "dashboard"
sys.path.insert(0, str(dash_dir))

from app import app

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting Pair A AI-Aided Phase I Visual Analytics Dashboard...")
    print("Serving locally at: http://127.0.0.1:8050")
    print("Press Ctrl+C to stop.")
    print("=" * 60 + "\n")
    app.run(debug=False, host="127.0.0.1", port=8050)

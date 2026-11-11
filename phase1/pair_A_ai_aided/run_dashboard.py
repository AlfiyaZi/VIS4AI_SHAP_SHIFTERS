"""Convenience launcher for the Pair A interactive dashboard."""

import argparse
import sys
from pathlib import Path

# Add dashboard folder to sys.path
dash_dir = Path(__file__).resolve().parent / "dashboard"
sys.path.insert(0, str(dash_dir))

from app import app, data


def main():
    parser = argparse.ArgumentParser(description="Launch Pair A AI-Aided Phase I Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Port to serve on (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Run with Dash debug mode enabled")
    args = parser.parse_args()

    print("\n" + "=" * 68)
    print("  VIS4AI: SHAP_SHIFTERS - Phase I AI-Aided Coordinated Dashboard")
    print("=" * 68)
    print(f"  * Cohort Size:      {len(data.df):,} encounters ({data.df['patient_nbr'].nunique():,} unique patients)")
    print(f"  * Model A Accuracy: {(data.df['model_a_pred_pathway'] == data.df['true_next_pathway']).mean():.1%}")
    print(f"  * Model B Accuracy: {(data.df['model_b_pred_pathway'] == data.df['true_next_pathway']).mean():.1%}")
    print(f"  * Majority Base:    {(data.df['majority_pred_pathway'] == data.df['true_next_pathway']).mean():.1%}")
    print(f"  * Serving URL:      http://{args.host}:{args.port}")
    print("  * Press Ctrl+C to stop the server.")
    print("=" * 68 + "\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

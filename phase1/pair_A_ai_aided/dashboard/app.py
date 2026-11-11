"""
Main Dash application for Pair A (AI-aided) Phase I analysis.
Modular architecture delegating layout to layout.py and callbacks to callbacks.py.
"""

from __future__ import annotations

import dash

import callbacks
import data_layer
import layout

app = dash.Dash(
    __name__,
    title="SHAP_SHIFTERS - Phase I Explorer",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# Load data contract from shared/data
data = data_layer.load()

# Construct UI layout and register coordinated cross-filtering callbacks
app.layout = layout.build_layout(data)
callbacks.register_callbacks(app, data)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Serving Pair A Phase I Dashboard locally at: http://127.0.0.1:8050")
    print("=" * 60 + "\n")
    app.run(debug=True, port=8050)

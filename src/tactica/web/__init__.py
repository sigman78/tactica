"""Web dashboard for the tactica sandbox (optional ``web`` extra).

``tactica web`` serves a local single-page dashboard: run tournaments,
SPRT validations, skill curves and noise floors from editable presets,
watch live charts/logs over SSE, and step through replays on an SVG board.
The simulator core stays framework-free; everything here is a thin layer
over :mod:`tactica.eval`.
"""

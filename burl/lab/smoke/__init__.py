"""In-process smoke drivers for burl.lab.

These exist to exercise the full spine (phases → drive → engine → tools) on
real harvested decisions without going through the FastAPI server.  Faster
iteration, isolates the real-MLX surface from the wire layer.
"""
from __future__ import annotations

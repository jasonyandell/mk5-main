#!/usr/bin/env python3
"""Simple HTTP server for E[Q] 3D visualization.

Usage:
    cd forge/analysis/results
    python serve.py

Then open http://localhost:8000/web/eq_surface_3d.html
"""

import http.server
import socketserver

PORT = 8000

Handler = http.server.SimpleHTTPRequestHandler
Handler.extensions_map.update({
    '.jsonl': 'application/json',
})

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at http://localhost:{PORT}")
    print(f"Open: http://localhost:{PORT}/web/eq_surface_3d.html")
    httpd.serve_forever()

"""
web/dashboard.py

Windows-side dashboard host: serves the React UI locally and points it at
the robot server running on the Pi. Standard library only — nothing to
install on the Windows machine.

    python web/dashboard.py --pi 192.168.137.50:8000
    python web/dashboard.py --pi raspberrypi.local:8000 --port 8080

Opens http://localhost:8080/?pi=<addr> in your browser; the UI remembers
the Pi address (localStorage), so later runs can omit --pi.

The browser talks to the Pi DIRECTLY (fetch/SSE/MJPEG straight to
http://<pi>:8000, CORS-enabled there) — this host only serves the static
files, it is not a proxy, so it adds zero latency to the control path.
"""
import argparse
import os
import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):      # silence per-request noise
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")   # always fresh app.js
        super().end_headers()

    def do_GET(self):
        # Serve the SPA for "/" (query string like ?pi=... is handled
        # client-side by app.js).
        if self.path.split("?", 1)[0] == "/":
            self.path = "/index.html"
        # The UI references /static/... so it also works when served by the
        # Pi's Flask server; here the files ARE the root, so strip the prefix.
        if self.path.startswith("/static/"):
            self.path = self.path[len("/static"):]
        return super().do_GET()


def main():
    ap = argparse.ArgumentParser(description="AutoBin dashboard (Windows host)")
    ap.add_argument("--pi", default="", help="robot server address, e.g. 192.168.137.50:8000")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    url = f"http://localhost:{args.port}/" + (f"?pi={args.pi}" if args.pi else "")
    handler = partial(QuietHandler, directory=STATIC_DIR)
    server = HTTPServer(("127.0.0.1", args.port), handler)

    print(f"dashboard at {url}")
    print("robot server on the Pi:  python web/server.py")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()

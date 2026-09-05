"""web/dashboard.py: Windows-side host serving the UI, stdlib only.
The browser talks to the Pi directly (CORS); this host only serves static files, no proxying."""
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
        # Serve the SPA for "/"; ?pi=... is handled client-side by app.js.
        if self.path.split("?", 1)[0] == "/":
            self.path = "/index.html"
        # /static/... also works when served by the Pi; here files ARE the root, so strip the prefix.
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

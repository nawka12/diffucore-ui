"""Diffucore UI — entry point."""

import argparse

import uvicorn

from server import app

parser = argparse.ArgumentParser(description="Diffucore UI")
parser.add_argument(
    "--listen",
    action="store_true",
    help="Bind to 0.0.0.0 so the UI is reachable from other machines on the "
         "network (default: localhost only).",
)
parser.add_argument(
    "--port",
    type=int,
    default=7860,
    help="Port to serve the UI on (default: 7860).",
)
parser.add_argument(
    "--share",
    action="store_true",
    help="Expose the UI over a public Cloudflare quick tunnel and print a "
         "trycloudflare.com URL (downloads cloudflared on first use).",
)
parser.add_argument(
    "--autolaunch",
    action="store_true",
    help="Open the UI in the default web browser once the server starts "
         "(launch.sh / launch.bat pass this by default). Suppressed when "
         "--share or --listen is set, since those serve remote clients.",
)
args = parser.parse_args()

if args.share:
    import share
    share.start(args.port)

if args.autolaunch and not (args.share or args.listen):
    import threading
    import webbrowser

    threading.Timer(
        1.5, webbrowser.open, args=(f"http://127.0.0.1:{args.port}",)
    ).start()

uvicorn.run(
    app,
    host="0.0.0.0" if args.listen else "127.0.0.1",
    port=args.port,
)

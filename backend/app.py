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
args = parser.parse_args()

uvicorn.run(
    app,
    host="0.0.0.0" if args.listen else "127.0.0.1",
    port=args.port,
)

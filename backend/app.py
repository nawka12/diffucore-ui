"""Diffucore UI — entry point."""

import argparse

parser = argparse.ArgumentParser(description="Diffucore UI")
parser.add_argument(
    "--listen",
    action="store_true",
    help="Bind to 0.0.0.0 so the UI is reachable from other machines on the "
         "network (default: localhost only).",
)
parser.add_argument(
    "--host",
    default=None,
    help="Bind to a specific interface IP (overrides --listen). Useful on "
         "multi-interface machines (VPN + LAN).",
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
         "trycloudflare.com URL (downloads cloudflared on first use). "
         "Auth is enabled automatically with a token printed to the terminal.",
)
parser.add_argument(
    "--auth-token",
    default=None,
    help="Require this token to access the UI (cookie-based login page). "
         "Implied by --share; recommended with --listen on an untrusted LAN. "
         "If omitted under --share, a token is generated and saved to "
         ".auth_token.",
)
parser.add_argument(
    "--log-file",
    default=None,
    help="Append structured logs to this file (rotating, chmod 600) in addition "
         "to stderr. A per-process run-id is stamped into every line. Useful "
         "for triaging 'it broke' reports. Secrets (auth token, share URL) are "
         "never written to the log.",
)
parser.add_argument(
    "--log-level",
    default="INFO",
    choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    help="Logging level for the structured logger (default: INFO).",
)
parser.add_argument(
    "--autolaunch",
    action="store_true",
    help="Open the UI in the default web browser once the server starts "
         "(launch.sh / launch.bat pass this by default). Suppressed when "
         "--share or --listen is set, since those serve remote clients.",
)
args = parser.parse_args()

# Configure structured logging BEFORE importing server: server.py emits log
# lines at import time (offload default, FastAPI app creation), so the loggers
# must be wired up first to capture them. Tokens / share URLs stay on print()
# (stdout only) so they never land in the log file.
import log_setup
log_setup.configure(log_file=args.log_file, level=args.log_level)

import uvicorn

from server import app, configure_auth
from auth import load_or_create_token

# ── auth ──────────────────────────────────────────────────────────────
# --share publishes the UI to the public internet, so the gate goes on by
# default with an auto-generated token. --listen is LAN-only; we leave auth
# opt-in (via --auth-token) but warn, since a trusted home network may not want
# a login step. An explicit --auth-token always wins.
share_token = None
if args.share or args.auth_token:
    token = args.auth_token or load_or_create_token()
    secure = args.share  # the tunnel is HTTPS; a plain-http LAN can't use Secure
    configure_auth(token=token, enabled=True, secure=secure)
    share_token = token if args.share else None
    if args.auth_token:
        print(f"[auth] token gate enabled (token: {token})", flush=True)
    else:
        print(f"[auth] --share: token gate enabled. Token: {token}", flush=True)
        print(f"[auth] saved to .auth_token (chmod 600). The share URL below "
              f"includes ?token=… for one-click access.", flush=True)
elif args.listen:
    print("[auth] --listen without --auth-token: the UI is open to anyone on "
          "this network. Pass --auth-token <token> to gate it.", flush=True)

if args.share:
    import share
    share.start(args.port, token=share_token)

if args.autolaunch and not (args.share or args.listen):
    import threading
    import webbrowser

    threading.Timer(
        1.5, webbrowser.open, args=(f"http://127.0.0.1:{args.port}",)
    ).start()

host = args.host or ("0.0.0.0" if args.listen else "127.0.0.1")
uvicorn.run(
    app,
    host=host,
    port=args.port,
)

# Diffucore UI — Extensions

Diffucore UI has an extension platform in the spirit of AUTO1111's extensions
and ComfyUI's custom nodes. Each extension is a folder under `extensions/`
with an `extension.json` manifest, a Python entry point, and (optionally) a
`web/` directory of JS that gets injected into the UI. Extensions can add API
endpoints, hook into generation and model loading, queue jobs on the shared
worker, broadcast SSE events, store their own settings, and add tabs and
settings panels to the frontend.

A reference extension ships with the app at
[`../extensions/example-watermark/`](../extensions/example-watermark/) — read
its `extension.py` and `web/example.js` alongside this document.

## Layout

```
extensions/
└── your-extension/
    ├── extension.json      # manifest (required)
    ├── extension.py        # Python entry point (name from manifest "entry")
    ├── requirements.txt    # optional — pip-installed on install
    └── web/                # optional — every .js here is loaded into the UI
        └── ui.js
```

The folder name **must** match the manifest's `name` field after install (the
installer renames the folder to the manifest name, sanitized to
`[A-Za-z0-9._-]`). The loader keys everything — registry, state, URL prefix,
on-disk folder — on that name, so they always line up.

### Manifest (`extension.json`)

```json
{
  "name": "your-extension",
  "title": "Your Extension",
  "version": "0.1.0",
  "author": "Your Name",
  "description": "One line shown in the Extensions panel.",
  "entry": "extension.py",
  "web": "web",
  "min_ui_version": "0.1.0",
  "default_enabled": true
}
```

Only `name` is required. Defaults: `entry` is `extension.py`, `web` is `web`,
`version` is `0.0.0`, `default_enabled` is `true`. Unknown fields are ignored,
so a newer manifest never breaks an older loader.

`default_enabled` controls whether a freshly-discovered extension (no
`state.json` entry yet) loads on startup. Most extensions leave it `true` so a
dropped-in folder works immediately. Set it to `false` for an example or
opt-in extension — it shows up in Settings → Extensions (marked "default off")
but doesn't load until the user turns it on. The shipped
`example-watermark` uses this so the reference code lives in the repo without
auto-running on every install. Once the user toggles it, their choice is
persisted in `state.json` and wins over the manifest default.

## The Python entry point

The loader imports `entry` and calls `setup(api)` if it exists, where `api` is
an `ExtensionAPI`. Everything the extension does flows through `api` so the
loader can unwind it cleanly on disable or reload.

```python
def setup(api):
    api.on("post_generate", my_hook)
    api.add_api_router(my_router)
    ...
```

### `ExtensionAPI` reference

| Method / attribute | Description |
|---|---|
| `api.on(event, handler)` | Register a hook (see [Hooks](#hooks)). |
| `api.add_api_router(router, *, prefix="")` | Mount a `fastapi.APIRouter` at `/api/ext/<name><prefix>`. |
| `api.serve_static(path, directory)` | Serve a directory at `/ext-static/<name>/<path>`. |
| `api.add_web_scripts(files)` | Override the auto-discovered JS file list (relative to the `web/` dir). By default every `*.js` directly in `web/` is injected. |
| `api.enqueue_job(label, run, *, kind="ext")` | Queue a callable on the **shared background worker** (one at a time with generation, so it shares the GPU safely). `run` receives the server's `Job` object. Returns the job id. |
| `api.broadcast(event_dict)` | Push an event dict to every connected SSE client (same stream as progress/preview). |
| `api.get_setting(key, default=None)` | Read a persisted per-extension setting (stored in `extensions/state.json`). |
| `api.set_setting(key, value)` | Write a persisted per-extension setting. |
| `api.engine` | The `Engine` singleton (read model state: `loaded_name`, `loaded_family`, `last_seed`, …). Don't reload models directly — use `enqueue_job` so it serializes with generation. |
| `api.root_dir` | Project root `Path`. |
| `api.ext_dir` | This extension's own directory `Path`. |

## Hooks

Register a handler with `api.on(event, handler)`. The handler receives a
`HookContext` and may mutate it; the server reads the relevant fields back
after running all handlers for that event.

```python
from dataclasses import dataclass

@dataclass
class HookContext:
    event: str
    payload: Any = None     # the request model (GeneratePayload / LoadPayload)
    image: Any = None       # PIL.Image (post_generate / post_save)
    info: str = ""          # the base gen info string
    path: Any = None        # saved file Path (post_save)
    status: str = ""        # load result string (post_load)
    extra: dict = ...       # scratch dict for the extension's own use
```

| Event | When | Fields set | Typical use |
|---|---|---|---|
| `startup` | Once, after all extensions load | — | Open resources, warm caches. |
| `pre_generate` | Before the engine runs, after the "model loaded" check | `payload` (`GeneratePayload`) | Tweak the prompt, seed, steps, etc. Mutations land on the payload in place. |
| `post_generate` | After generation + detailer + upscaler, **before** the PNG is saved | `payload`, `image` (PIL), `info` | Post-process the image (watermark, filter, composite). Replace `ctx.image` to change what gets saved. |
| `post_save` | After the PNG is written to `outputs/` | `payload`, `image`, `path` (`Path`) | Mirror the file, log it, push an SSE event. |
| `pre_load` | Before a model load, after the request is queued | `payload` (`LoadPayload`) | Observe/adjust the load request. |
| `post_load` | After a model load returns | `payload`, `status` (str) | React to a successful or failed load. `status` starts with `"Loaded"` on success. |
| `shutdown` | On server shutdown | — | Release resources. |

A handler that raises is logged and skipped — a buggy extension can't abort a
generation or a load. The failure is recorded on the extension and shown in the
Extensions panel.

### Example: stamp the seed onto every image

```python
from PIL import Image, ImageDraw, ImageFont

def setup(api):
    def on_post_generate(ctx):
        if not api.get_setting("enabled", True):
            return
        text = f"seed:{api.engine.last_seed}"
        img = ctx.image.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = ImageFont.load_default()
        draw.text((img.width - 80, img.height - 14), text, font=font, fill=api.get_setting("color", "#e8a065"))
        ctx.image = Image.alpha_composite(img, overlay).convert("RGB")

    api.on("post_generate", on_post_generate)
```

## Custom API endpoints

Mount a `fastapi.APIRouter` with `api.add_api_router(router)`. It's served at
`/api/ext/<name>`, so a route defined as `""` resolves to that exact path and
`"/status"` resolves to `/api/ext/<name>/status`.

```python
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class Ping(BaseModel):
    msg: str = ""

@router.get("")
def status():
    return {"ok": True, "model": api.engine.loaded_name}

@router.post("/ping")
def ping(p: Ping):
    return {"echo": p.msg}

def setup(api):
    api.add_api_router(router)
```

Pydantic models and all the usual FastAPI features work normally. The frontend
reaches these with plain `fetch('/api/ext/your-extension/...')`.

## Static assets

`api.serve_static(path, directory)` serves a directory at
`/ext-static/<name>/<path>`. Your extension's `web/` directory is **always**
served at `/ext-static/<name>/` automatically (so the injected script tags
resolve) — use `serve_static` only for additional asset directories.

## Custom jobs

`api.enqueue_job(label, run)` queues work on the same single-worker thread that
runs generation. This is the right way to do any GPU work or anything that
mustn't race with a generation. The job's result dict is broadcast as a `done`
SSE event, just like a generation.

```python
def setup(api):
    def do_thing(job):
        # runs on the worker thread, serialized with generation
        ...
        return {"info": "done"}

    @router.post("/run")
    def run():
        job_id = api.enqueue_job("do thing", do_thing)
        return {"job": job_id}
```

The job is visible in the shared queue UI and can be cancelled from any device,
identical to a generation job.

## SSE broadcast

`api.broadcast({...})` pushes an event to every connected SSE client (the same
`/api/events` stream that carries progress, previews, and queue changes). Use
a `type` that won't collide with the built-ins (`progress`, `preview`, `done`,
`error`, `cancelled`, `status`, `queue`, `snapshot`). A `type: "ext:<name>"`
prefix is conventional.

```python
api.broadcast({"type": "ext:your-extension", "path": str(ctx.path)})
```

The frontend can listen for it on the existing `EventSource` — extensions
typically add their own listener in their injected JS.

## Extension settings

`api.get_setting(key, default)` and `api.set_setting(key, value)` read/write a
per-extension key-value store persisted in `extensions/state.json`. Use it for
anything your extension needs to remember between sessions (toggles, colors,
last-used values). It's separate from the app's global `Settings` model, so an
extension can't break the core settings round-trip.

The example extension exposes its settings through its own API endpoints and a
settings-panel UI — see `example-watermark/extension.py` and
`web/example.js`.

## The frontend bridge

Every `*.js` file in your `web/` directory is loaded into the index page (in
sorted order, deferred, before Alpine initializes). Register UI through the
global `window.DiffucoreExt`:

```js
// A top-level tab (button in the main nav + a content area).
window.DiffucoreExt.registerTab({
  id: 'your-extension',
  title: 'My Ext',
  mount(el)   { el.innerHTML = '<div x-data="...">...</div>'; },
  unmount(el) { /* optional: clean up listeners, etc. */ },
});

// A panel under Settings → Extensions.
window.DiffucoreExt.registerSettingsPanel({
  id: 'your-extension',
  title: 'My Ext',
  mount(el)   { /* el is a <div> inside the Extensions settings section */ },
  unmount(el) { },
});
```

`mount(el)` receives a container element the extension owns entirely — fill it
with `innerHTML`, attach listeners, instantiate Alpine components with
`x-data`, whatever you need. `unmount(el)` is called when the user leaves the
tab / closes the panel, so you can drop listeners.

`registerTab` / `registerSettingsPanel` must be called at the top level of your
script (not inside an `alpine:init` handler) — the Alpine `app` component reads
`DiffucoreExt.tabs` during `init()`, which runs before any later listener.

The bridge also exposes `DiffucoreExt.tabs` and `DiffucoreExt.settingsPanels`
(the registered arrays) for introspection.

## Installation

Users install extensions from **Settings → Extensions → Install**:

- a **git URL** (e.g. `https://github.com/you/your-ext.git`) — `git clone --depth 1`,
- or a **.zip archive URL** (e.g. a GitHub release asset) — downloaded and extracted.

The installer normalizes the layout: the extension may live at the archive's
root or one level down (the common "release zip" layout), and the folder is
moved to `extensions/<manifest-name>`. If the source includes a
`requirements.txt`, it's `pip install -r`'d into the current venv
(best-effort). The extension is loaded immediately on successful install.

Installing from a URL also works for local development — point it at a local
`file://` .zip, or just drop the folder into `extensions/` and restart.

For development, use **Reload** in the panel (or `POST /api/extensions/reload`)
to re-import the entry module after an edit, without restarting the server.
The loader drops the old hooks/routes/statics first so nothing doubles up.

## Enable / disable / uninstall

- **Enable/disable** toggles whether the extension's module is imported and its
  hooks fire. Disabling stops its hooks and stops injecting its scripts; the
  frontend script tags refresh on the next page load. Note that an extension's
  API routes and static mounts cannot be removed without a server **restart**
  (Starlette has no unmount), so a disabled extension's endpoints keep serving
  until you restart. Installing, enabling, or reloading attaches new routes
  live — no restart needed.
- **Uninstall** deletes the extension's folder and drops its hooks, routes, and
  persisted state.
- A **broken extension** (one whose `setup()` raises, or whose manifest is
  invalid) is shown with its error in the panel and is otherwise inert — it
  never blocks the app or other extensions.

## Management API

| Endpoint | Method | Body / Query | Description |
|---|---|---|---|
| `/api/extensions` | GET | — | List every discovered extension with load state. |
| `/api/extensions/web` | GET | — | Script URLs injected into the index page. |
| `/api/extensions/install` | POST | `{"url": "..."}` | Install from a git/zip URL. |
| `/api/extensions/toggle` | POST | `{"name": "...", "enabled": bool}` | Enable/disable. |
| `/api/extensions/reload` | POST | `?name=...` | Re-import one extension. |
| `/api/extensions/uninstall` | POST | `{"name": "..."}` | Delete the folder + state. |

## Safety notes

- Loading a Python extension runs arbitrary code, same as AUTO1111 / ComfyUI.
  Only install extensions from sources you trust.
- Each extension is imported in isolation; a failure during `setup()` or a hook
  is caught and recorded, never propagated to the generation or load path.
- Manifest `name` values are sanitized to `[A-Za-z0-9._-]` and path-escape
  attempts are refused, so a malicious manifest can't write outside
  `extensions/` or escape via the API URL prefix.
- `api.engine` is the live singleton. Inspect it freely, but **do not** call
  `load_model` / `generate_*` directly from a request handler — that would race
  with the worker. Use `api.enqueue_job(...)` so the work serializes.

## Versioning

`min_ui_version` in the manifest is checked against the app's version on load;
a mismatch is logged as a warning (not a hard failure) so an extension built
against an older UI keeps working where possible. Bump your extension's
`version` on each release — it's shown in the panel.

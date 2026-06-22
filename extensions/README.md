# Extensions

Each subdirectory here is one Diffucore UI extension, declared by an
`extension.json` manifest. The loader scans this folder at startup and loads
every enabled extension. Install new ones from the **Settings → Extensions**
panel (git URL or .zip archive URL), or drop a folder here and restart.

`example-watermark/` ships with the app as a reference — read its
`extension.py` and `web/example.js` alongside [`../docs/EXTENSIONS.md`](../docs/EXTENSIONS.md)
for the full API.

Installed extensions and their per-extension settings live in `state.json`
(gitignored). A broken extension is shown in the panel and never blocks the app.

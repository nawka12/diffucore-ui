// Example Diffucore UI extension — frontend side.
//
// This script is injected into the index page by the backend (every .js file
// in the extension's web/ dir is auto-loaded). It runs before Alpine inits,
// so it can register a tab and a settings panel through the window.DiffucoreExt
// bridge. Read alongside extension.py and docs/EXTENSIONS.md.

(function () {
  // A small helper for talking to this extension's own API. The backend mounts
  // the router at /api/ext/<name>; the name matches extension.json's "name".
  const NAME = 'example-watermark';
  const API = `/api/ext/${NAME}`;

  async function getStatus() {
    try {
      const r = await fetch(API);
      return await r.json();
    } catch (e) {
      return { error: String(e) };
    }
  }

  async function saveSettings(s) {
    const r = await fetch(`${API}/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(s),
    });
    return r.ok;
  }

  // ── settings panel (shown under Settings → Extensions) ───────────
  // mount(el) is handed a container <div>; the extension owns the DOM inside.
  window.DiffucoreExt.registerSettingsPanel({
    id: NAME,
    title: 'Watermark',
    mount(el) {
      el.innerHTML = `
        <h3 style="margin:0 0 6px;font-size:14px;color:var(--txt)">Watermark (example)</h3>
        <p class="sub" style="margin:0 0 10px;color:var(--txt-3);font-size:12px">
          Stamps the seed into the bottom-right corner of every generated image.
        </p>
        <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;font-size:13px">
          <label style="display:flex;align-items:center;gap:6px">
            <input type="checkbox" id="wm-on"> enabled
          </label>
          <label style="display:flex;align-items:center;gap:6px">
            prefix
            <input type="text" id="wm-prefix" value="seed:"
              style="background:var(--surface-2);border:1px solid var(--line);color:var(--txt);
                     border-radius:6px;padding:4px 6px;font-family:var(--mono);font-size:12px;width:90px">
          </label>
          <label style="display:flex;align-items:center;gap:6px">
            color
            <input type="color" id="wm-color" value="#e8a065" style="width:32px;height:24px;border:none;background:none">
          </label>
          <button class="btn small primary" id="wm-save">Save</button>
          <span id="wm-msg" style="font-size:11px;color:var(--teal)"></span>
        </div>`;

      const on = el.querySelector('#wm-on');
      const prefix = el.querySelector('#wm-prefix');
      const color = el.querySelector('#wm-color');
      const msg = el.querySelector('#wm-msg');

      // Seed the form from the backend's current settings.
      getStatus().then(s => {
        if (s && !s.error) {
          on.checked = !!s.enabled;
          prefix.value = s.prefix || 'seed:';
          color.value = s.color || '#e8a065';
        }
      });

      el.querySelector('#wm-save').addEventListener('click', async () => {
        msg.textContent = 'saving…';
        const ok = await saveSettings({
          enabled: on.checked,
          prefix: prefix.value,
          color: color.value,
        });
        msg.textContent = ok ? 'saved' : 'failed';
        setTimeout(() => (msg.textContent = ''), 1500);
      });
    },
  });

  // ── tab (shown in the main nav) ──────────────────────────────────
  // mount(el) receives the full tab content area; unmount(el) cleans up.
  window.DiffucoreExt.registerTab({
    id: NAME,
    title: 'Watermark',
    mount(el) {
      el.innerHTML = `
        <div style="max-width:560px">
          <h2 style="font-family:var(--serif);font-weight:400;margin:0 0 8px">Watermark <em style="color:var(--accent)">example</em></h2>
          <p class="hint">This tab is rendered by the example extension's own JS.
            It calls the extension's <code>/api/ext/${NAME}</code> endpoint:</p>
          <pre class="meta" id="wm-tab-status" style="min-height:80px">loading…</pre>
          <button class="btn small" id="wm-tab-refresh">Refresh</button>
        </div>`;
      const pre = el.querySelector('#wm-tab-status');
      const refresh = async () => {
        pre.textContent = JSON.stringify(await getStatus(), null, 2);
      };
      refresh();
      el.querySelector('#wm-tab-refresh').addEventListener('click', refresh);
    },
    unmount(el) {
      // Nothing to clean up here — the innerHTML is replaced on next mount.
    },
  });
})();

// Diffucore UI — Alpine state + streaming. No build step.

// ── extension bridge ──────────────────────────────────────────────
// Set up before Alpine inits so extension scripts (injected between this file
// and alpine.min.js) can register tabs/settings panels by calling
// window.DiffucoreExt.registerTab / registerSettingsPanel. The Alpine app
// reads .tabs during init() and renders them in the main nav.
window.DiffucoreExt = (function () {
  const tabs = [];
  const settingsPanels = [];
  function registerTab(spec) {
    if (!spec || !spec.id || !spec.title || typeof spec.mount !== 'function') {
      console.warn('[DiffucoreExt] registerTab needs {id, title, mount(el)}');
      return;
    }
    if (!tabs.find(t => t.id === spec.id)) tabs.push(spec);
  }
  function registerSettingsPanel(spec) {
    if (!spec || !spec.id || !spec.title || typeof spec.mount !== 'function') {
      console.warn('[DiffucoreExt] registerSettingsPanel needs {id, title, mount(el)}');
      return;
    }
    if (!settingsPanels.find(s => s.id === spec.id)) settingsPanels.push(spec);
  }
  return { tabs, settingsPanels, registerTab, registerSettingsPanel };
})();

// ── fetch helper ──────────────────────────────────────────────────
// fetch + JSON parse that fails loudly. A non-2xx response (e.g. a 500 whose
// body is an HTML/text error page, not JSON) would otherwise blow up inside
// .json() with a cryptic "Unexpected token '<'" and silently blank the calling
// state. This throws a real Error carrying the status plus the server's FastAPI
// `detail` (or a short body snippet), so callers' catch blocks can surface
// something actionable.
async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let detail = '';
    try {
      const body = await r.text();
      try { detail = JSON.parse(body).detail || ''; }
      catch (_) { detail = body.slice(0, 200); }
    } catch (_) { /* body unreadable */ }
    throw new Error(detail ? `${r.status}: ${detail}` : `Request failed (${r.status} ${r.statusText})`);
  }
  return r.json();
}

document.addEventListener('alpine:init', () => {
  Alpine.data('app', () => ({
    // ── model rack ──────────────────────────────────────────────
    modelType: 'SD/SDXL',
    checkpoints: [], dits: [], vaes: [], tes: [], loras: [], detailers: [], upscalers: [],
    checkpoint: '', dit: '', vae: '', te: '', clip: '', fluxCheckpoint: '',
    perf: { compile: false, cudaGraphs: false, channelsLast: true, tf32: false, fp16Acc: false, fa2Attn: false, offload: 'full' },
    fa2Available: false,
    recommendedOffload: 'full',   // GPU-VRAM-based default from the backend (set on init)
    status: 'No model loaded',
    modelLoaded: false,
    loadingModel: false,
    // Set when the user edits the model rack (family/checkpoint/file selectors)
    // but hasn't loaded yet. A shared SSE broadcast (status from another tab's
    // load) must not clobber that in-progress selection; cleared on a successful
    // local load, so untouched tabs still sync to a model loaded elsewhere.
    loadFormDirty: false,
    animaApplied: false,
    fluxApplied: false,
    uiId: 'diffucore-ui', diffId: 'diffucore',

    // ── shared option sets ──────────────────────────────────────
    samplersSd: [],
    samplersAnima: [],
    samplersFlux: [],
    schedulersSd: ['karras'],
    schedulersAnima: ['flow'],
    schedulersFlux: ['flux'],
    paramTypes: ['None'],

    // ── navigation ──────────────────────────────────────────────
    tab: 'generate',
    mode: 't2i',

    // ── generate form (shared across modes) ─────────────────────
    form: {
      prompt: '', neg: '',
      sampler: 'dpmpp_2m', scheduler: 'karras',
      steps: 25, cfg: 6.0, seed: -1,
      width: 1024, height: 1024,
      strength: 0.6, shift: 3.0,
      teacacheOn: false, teacache: 0.15, teacacheCalibrated: true,
      deepcacheOn: false, deepcache: 2,
    },
    // Batch count: >1 submits N generate jobs at once. With a pinned seed each
    // job gets seed+i; with seed=-1 each gets a fresh backend random. The queue
    // runs them one at a time (as it does for any job); progress/preview route
    // to whichever batch member is currently running.
    batchCount: 1,

    // ── <lora:…> autocomplete in the prompt ─────────────────────
    loraAC: { open: false, items: [], index: 0, start: -1, key: 'prompt', el: null, set: null, wrap: null },

    inputImage: null,
    maskImage: null,
    dragKey: null,
    maskBrush: 40,
    maskTool: 'brush',   // brush | eraser | rect
    maskPainted: false,
    maskMax: false,      // fullscreen the input & mask editor
    maskZoom: 1,         // display zoom while maximized (1 = fit)

    // ── detailer (ADetailer-style passes after generate) ────────
    // `models` is a stack of {model, prompt} run in sequence; rest is shared.
    detail: {
      enabled: false, neg: '',
      models: [{ model: '', prompt: '' }],
      confidence: 0.3, strength: 0.4,
      dilation: 4, padding: 32, blur: 4, maxDet: 0,
      teacache: false,
    },

    // ── upscaler (tiled, after generate) ─────────────────────────
    // teacache is independent of the main slider (0 = off): the refine pass
    // runs its whole trajectory in the detail regime, so caching softens it.
    upscale: {
      enabled: false, scale: 2.0, denoise: 0.35,
      tile: 1024, overlap: 128, prompt: '', teacache: 0.0, base: '',
    },

    // ── standalone upscale popover (from result / lightbox) ──────
    upscalePopover: { open: false, busy: false },
    upscaleForm: { scale: 2.0, denoise: 0.35, tile: 1024, overlap: 128, prompt: '', teacache: 0.0, base: '' },

    // ── generation output ───────────────────────────────────────
    busy: false,
    cancelling: false,
    progress: { step: 0, total: 0 },
    resultUrl: null,
    previewUrl: null,
    preview: true,
    info: '',
    lastSeed: -1,
    _titleBase: '',       // original tab title, captured on init
    _titleDone: false,    // a job finished while this tab was hidden → badge until looked at

    // ── shared queue (broadcast over /api/events to every device) ──
    queue: [],            // [{id, kind, label, status}] — running first, then pending
    runningJob: null,     // id of the job currently on the GPU (any device)
    runProg: { step: 0, total: 0 },  // progress of the running job (for the queue panel)
    myJobId: null,        // id of the single job THIS device submitted and is watching
    _myBatchIds: [],      // ids of in-flight batch jobs (empty for single-job flows)
    _jobWaiters: {},      // id -> resolve fn, fulfilled by the terminal SSE event

    // ── SSE connection state (#7) ────────────────────────────────
    // 'connected' | 'reconnecting' | 'down'. A stalled stream freezes the queue
    // panel silently; the banner tells the user the UI is no longer live.
    connState: 'connected',
    connDownSince: null,        // epoch ms when the connection went 'down'
    CONN_DOWN_TIMEOUT: 8000,    // reconnect grace before we banner the outage
    _connTimer: null,

    // ── OSS calibration ─────────────────────────────────────────
    calibrating: false,
    ossCalibrated: null,          // null = unknown, true/false = checked
    ossInfo: '',

    // ── settings panel (global, non-per-image knobs) ────────────
    settingsOpen: false,
    settingsTab: 'teacache',
    settings: { curvature: 0.25, eta_max: 1.0, beta_alpha: 0.6, beta_beta: 0.6, lq_threshold: 0.025, cfg_interval_start: 0.0, cfg_interval_end: 1.0, vae_tiling: 'auto', gen_defaults: null },
    teacacheStatus: { loaded: false, calibratable: false, family: null, coefficients: null },
    calibratingTea: false,

    // ── extensions ──────────────────────────────────────────────
    // `extensions` mirrors /api/extensions (list of installed exts). `extTabs`
    // is filled by extensions calling window.DiffucoreExt.registerTab from
    // their injected JS. `_mountedExtTab`/`_mountedExtSettings` track the
    // currently mounted panel so we unmount it cleanly on switch.
    extensions: [],
    extTabs: [],
    extInstallUrl: '',
    extInstallPip: false,  // opt-in: pip install -r requirements.txt on install (RCE risk — off by default)
    extBusy: false,
    _mountedExtTab: null,
    _mountedExtSettings: null,

    // ── gallery ─────────────────────────────────────────────────
    gallery: [],
    galleryGroups: [],
    galleryLimit: 60,   // chunked rendering: only this many thumbs live in the DOM
    galleryQuery: '',   // substring filter applied via /api/gallery?q=
    gallerySearching: false,
    selected: null,
    selectedMeta: '',
    selectedFields: {},
    lightbox: { open: false, index: 0, info: false },
    deleteConfirm: false,   // two-click confirm in the lightbox Delete button

    // ── metadata reader ─────────────────────────────────────────
    metaPreview: null,
    metaText: '',
    metaFields: null,

    // ── x/y/z sweep (txt2img only — reuses the shared form for base params) ──
    xyzSweep: false,
    // Start every axis empty — preset values would be sent as-is and, after a
    // type switch (e.g. to Checkpoint), become bogus values that error the grid.
    axes: {
      x: { type: 'Sampler', text: '', list: [] },
      y: { type: 'Steps', text: '', list: [] },
      z: { type: 'None', text: '', list: [] },
    },
    xyzGrids: [],
    xyzInfo: '',

    toast: '',
    toastKind: 'info',     // info | success | error — drives the toast border colour

    // ── computed ────────────────────────────────────────────────
    get samplers() {
      if (this.modelType === 'Anima') return this.samplersAnima;
      if (this.modelType === 'FLUX') return this.samplersFlux;
      return this.samplersSd;
    },
    get schedulers() {
      if (this.modelType === 'Anima') return this.schedulersAnima;
      if (this.modelType === 'FLUX') return this.schedulersFlux;
      return this.schedulersSd;
    },
    // Shift is a flow-only knob, and even then only some schedulers honour the
    // passed value: Anima uses it everywhere except flow_dyn (resolution-aware),
    // FLUX only on the plain flow scheduler (flux/sgm_uniform/simple derive
    // their own). SD/SDXL ignore it entirely.
    get isFlowModel() {
      return this.modelType === 'Anima' || this.modelType === 'FLUX';
    },
    get shiftHonored() {
      if (this.modelType === 'Anima') return this.form.scheduler !== 'flow_dyn';
      if (this.modelType === 'FLUX') return this.form.scheduler === 'flow';
      return false;
    },
    get offloadOptions() {
      // "stream" streams the backbone's blocks (ComfyUI --lowvram analog): the
      // FLUX DiT, the SD/SDXL UNet, and the Anima DiT all support it.
      return ['stream', 'full', 'encoders', 'none'];
    },
    get sweeping() {
      return this.mode === 't2i' && this.xyzSweep;
    },

    // X/Y/Z axes whose values come from a known set get a multi-select; numeric
    // axes (Steps / CFG / Seed) keep a free-text comma list.
    axisIsList(axis) { return axis.type === 'Sampler' || axis.type === 'Scheduler' || axis.type === 'Checkpoint'; },
    axisOptions(axis) {
      if (axis.type === 'Scheduler') return this.schedulers;
      // Anima is split-file: its "checkpoint" is the DiT (VAE + TE stay fixed).
      if (axis.type === 'Checkpoint') return this.modelType === 'Anima' ? this.dits : this.checkpoints;
      return this.samplers;
    },
    axisValues(axis) {
      if (axis.type === 'None') return '';   // a disabled axis carries no values
      return this.axisIsList(axis) ? axis.list.join(', ') : axis.text;
    },
    // Switching an axis's type makes its old values meaningless — and, for a
    // Checkpoint switch, harmful (stale sampler names load as bogus checkpoints
    // and abort the grid). Clear both stores so each type starts fresh.
    clearAxisValues(axis) { axis.list = []; axis.text = ''; },
    get progressPct() {
      const t = this.progress.total;
      return t > 0 ? Math.round((this.progress.step / t) * 100) : 0;
    },
    get progressLabel() {
      const t = this.progress.total;
      if (t <= 0) return 'Starting…';
      const steps = `${this.progress.step} / ${t}  (${this.progressPct}%)`;
      // X/Y/Z carries a cell index so the bar reads "image N/total" too.
      return this.progress.cells ? `Image ${this.progress.cell}/${this.progress.cells} · ${steps}` : steps;
    },
    get checkpointChoices() { return this.choices(this.checkpoints, 'models/checkpoints/'); },
    get ditChoices()        { return this.choices(this.dits, 'models/diffusion-models/'); },
    get vaeChoices()        { return this.choices(this.vaes, 'models/vae/'); },
    get teChoices()         { return this.choices(this.tes, 'models/text-encoders/'); },
    get detailerChoices()   { return this.choices(this.detailers, 'models/detailers/'); },

    choices(list, where) {
      return list.length ? list : [`(none in ${where})`];
    },

    // ── init ────────────────────────────────────────────────────
    async init() {
      // Pull in tabs registered by already-loaded extension scripts.
      this.extTabs = window.DiffucoreExt ? [...window.DiffucoreExt.tabs] : [];
      await this.refreshModels();
      await this.loadSettings();
      this.applyGenDefaults();
      this._initTitle();
      this.connectEvents();
      this.refreshExtensions();
    },

    // ── browser-tab title reflects this device's job state ──────
    // While a job runs the tab title shows a spinner + %, so the user can
    // tell from the tab alone — even sitting on another tab — whether their
    // generation is still going. When one finishes while the tab is hidden
    // we leave a "done" badge until they look back.
    _initTitle() {
      this._titleBase = document.title;   // "Diffucore"
      const apply = () => {
        if (this.busy) {
          this._titleDone = false;
          const t = this.progress.total;
          document.title = t > 0
            ? `⏳ ${this.progressPct}% · ${this._titleBase}`
            : `⏳ ${this._titleBase}`;
        } else {
          document.title = this._titleDone ? `✓ ${this._titleBase}` : this._titleBase;
        }
      };
      this.$watch('busy', (now, was) => {
        if (was && !now && document.hidden) this._titleDone = true;
        apply();
      });
      this.$watch('progress', apply);
      document.addEventListener('visibilitychange', () => {
        if (!document.hidden && this._titleDone) { this._titleDone = false; apply(); }
      });
    },

    // ── shared events (one SSE stream per device) ───────────────
    // The browser auto-reconnects an EventSource on error, but a stalled stream
    // freezes the queue panel silently — the user thinks a job is still running.
    // We surface it: onerror bumps connState to 'reconnecting', and if no
    // reconnect lands within CONN_DOWN_TIMEOUT we escalate to 'down' and show a
    // banner (IMPROVE.md #7). A fresh snapshot on reconnect re-syncs everything.
    connectEvents() {
      const es = new EventSource('/api/events');
      this._connEscalate = () => {
        if (this.connState === 'down') return;
        this.connState = 'down';
        this.connDownSince = Date.now();
      };
      const armEscalator = () => {
        clearTimeout(this._connTimer);
        this._connTimer = setTimeout(this._connEscalate, this.CONN_DOWN_TIMEOUT);
      };
      const clearEscalator = () => {
        clearTimeout(this._connTimer);
        this._connTimer = null;
        this.connState = 'connected';
        this.connDownSince = null;
      };
      es.onopen = () => clearEscalator();
      es.onmessage = (e) => {
        if (this.connState !== 'connected') clearEscalator();
        this.onServerEvent(JSON.parse(e.data));
      };
      es.onerror = () => {
        if (this.connState === 'connected') {
          this.connState = 'reconnecting';
          armEscalator();
        }
      };
      this._es = es;
    },

    onServerEvent(ev) {
      switch (ev.type) {
        case 'snapshot':
          this.applyState(ev);
          this.queue = ev.jobs; this.runningJob = ev.running;
          if (ev.progress) this.runProg = { step: ev.progress.step, total: ev.progress.total };
          // A terminal event (done/error/cancelled) is lost if the SSE drops while
          // the job finishes. The reconnect snapshot lists every live job, so any
          // waiter whose job is absent already ended off-stream — resolve it instead
          // of leaving busy stuck forever.
          {
            const live = new Set((ev.jobs || []).map((j) => String(j.id)));
            for (const id of Object.keys(this._jobWaiters)) {
              if (!live.has(id)) {
                const w = this._jobWaiters[id];
                delete this._jobWaiters[id];
                const bi = this._myBatchIds.indexOf(+id);
                if (bi !== -1) this._myBatchIds.splice(bi, 1);
                if (String(this.myJobId) === id) this.myJobId = null;
                w({ type: 'error', message: 'Lost connection during the job. Check the gallery for the result.' });
              }
            }
          }
          break;
        case 'queue':
          this.queue = ev.jobs; this.runningJob = ev.running;
          break;
        case 'status':
          this.applyState(ev);
          break;
        case 'progress':
          // Running-job progress feeds the queue panel; the main bar only
          // tracks THIS device's own job (or one of its in-flight batch jobs),
          // so a queued device isn't shown another device's progress.
          this.runProg = { step: ev.step, total: ev.total };
          if (ev.job === this.myJobId || this._myBatchIds.includes(ev.job))
            this.progress = { step: ev.step, total: ev.total, cell: ev.cell, cells: ev.cells };
          break;
        case 'preview':
          if (ev.job === this.myJobId || this._myBatchIds.includes(ev.job))
            this.previewUrl = ev.image;
          break;
        case 'done':
        case 'error':
        case 'cancelled': {
          const w = this._jobWaiters[ev.job];
          if (w) { delete this._jobWaiters[ev.job]; w(ev); }
          const bi = this._myBatchIds.indexOf(ev.job);
          if (bi !== -1) this._myBatchIds.splice(bi, 1);
          // Clear myJobId only when THIS job ends. A job queued behind it has
          // already moved myJobId on at submit time, so an earlier job's
          // completion must not null it out — otherwise the queued job's
          // progress/preview events stop routing and the bar sticks on "Starting…".
          if (ev.job === this.myJobId) this.myJobId = null;
          break;
        }
      }
    },

    // Submit a job, then resolve once its terminal event arrives on the stream.
    // Live progress/preview are handled globally by onServerEvent.
    async submitJob(url, payload) {
      const r = await fetchJSON(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      this.myJobId = r.job;
      return new Promise((resolve) => { this._jobWaiters[r.job] = resolve; });
    },

    // Submit N copies of a job for batch generation. A pinned seed is offset by
    // i per copy so the N images differ; seed=-1 is forwarded as-is so the
    // backend picks a fresh random per job (its existing behavior). Each job's
    // terminal event resolves one promise; Promise.all in the caller waits for
    // every copy. Unlike submitJob, this leaves myJobId null — onServerEvent
    // routes progress/preview through _myBatchIds instead.
    async submitBatch(url, payload, count) {
      const baseSeed = payload.seed;
      const promises = [];
      try {
        for (let i = 0; i < count; i++) {
          const body = { ...payload };
          if (baseSeed !== -1 && baseSeed != null) body.seed = baseSeed + i;
          const r = await fetchJSON(url, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
          });
          this._myBatchIds.push(r.job);
          promises.push(new Promise((resolve) => { this._jobWaiters[r.job] = resolve; }));
        }
      } catch (e) {
        // Fetch failed partway: cancel any jobs already enqueued so they don't
        // run orphaned, drop their waiters, then surface the error to the caller.
        for (const id of [...this._myBatchIds]) {
          delete this._jobWaiters[id];
          try {
            await fetch('/api/cancel', {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ job: id }),
            });
          } catch (_) { /* the worker still drops the job from its queue */ }
        }
        this._myBatchIds = [];
        throw e;
      }
      return promises;
    },

    // Reflect server-side model-load state (shared across devices/refresh).
    applyState(s) {
      this.status = s.status;
      this.modelLoaded = !!s.loaded;
      if (s.last_seed !== undefined) this.lastSeed = s.last_seed;
      // Don't overwrite a selection the user is editing locally (loadFormDirty);
      // a broadcast from another tab's load would otherwise revert it.
      if (s.load_form && !this.loadFormDirty) this.restoreLoadForm(s.load_form);
    },

    restoreLoadForm(f) {
      this.modelType = f.model_type;
      if (f.model_type === 'FLUX') this.fluxCheckpoint = f.checkpoint || '';
      else if (f.checkpoint) this.checkpoint = f.checkpoint;
      if (f.dit) this.dit = f.dit;
      if (f.vae) this.vae = f.vae;
      if (f.te) this.te = f.te;
      if (f.clip) this.clip = f.clip;
      if (f.offload) this.perf.offload = f.offload;
      this.perf.compile = !!f.compile;
      this.perf.cudaGraphs = !!f.cuda_graphs;
      this.perf.channelsLast = !!f.channels_last;
      this.perf.tf32 = !!f.tf32;
      this.perf.fp16Acc = !!f.fp16_accumulation;
      this.perf.fa2Attn = f.attention === 'fa2_turing';
      this.syncSampler();
      this.syncScheduler();
    },

    async refreshModels() {
      const m = await fetchJSON('/api/models');
      this.checkpoints = m.checkpoints; this.dits = m.dits;
      this.vaes = m.vaes; this.tes = m.tes; this.loras = m.loras;
      this.detailers = m.detailers || [];
      this.upscalers = m.upscalers || [];
      this.samplersSd = m.samplers_sd;
      this.samplersAnima = m.samplers_anima;
      this.samplersFlux = m.samplers_flux;
      this.schedulersSd = m.schedulers_sd;
      this.schedulersAnima = m.schedulers_anima;
      this.schedulersFlux = m.schedulers_flux;
      this.paramTypes = m.xyz_param_types;
      this.recommendedOffload = m.recommended_offload || 'full';
      this.fa2Available = !!m.fa2_available;
      this.uiId = m.ui_id; this.diffId = m.diff_id;
      // First fetch seeds every selector; a later Refresh (picking up newly
      // dropped files) must NOT clobber selections the user already made —
      // only replace ones whose file vanished from the new list.
      if (!this._modelsFetched) {
        this._modelsFetched = true;
        this.perf.offload = this.recommendedOffload;
        this.checkpoint = this.checkpointChoices[0];
        this.dit = this.ditChoices[0];
        this.vae = this.vaeChoices[0];
        this.te = this.teChoices[0];
        this.clip = this.teChoices[0];
        this.detail.models[0].model = this.detailerChoices[0];
      } else {
        const keep = (cur, list) => (list.includes(cur) ? cur : list[0]);
        this.checkpoint = keep(this.checkpoint, this.checkpointChoices);
        this.dit = keep(this.dit, this.ditChoices);
        this.vae = keep(this.vae, this.vaeChoices);
        this.te = keep(this.te, this.teChoices);
        // '' is a valid CLIP pick ("(none)", FLUX.2); a gone file falls to none.
        if (this.clip !== '' && !this.teChoices.includes(this.clip)) this.clip = '';
        for (const dm of this.detail.models) dm.model = keep(dm.model, this.detailerChoices);
      }
      this.syncSampler();
      this.syncScheduler();
      // Hydrate from server-side load state last, so a model already loaded by
      // another device (or before a refresh) restores its exact selections.
      this.applyState(m);
    },

    setModelType(type) {
      this.modelType = type;
      this.loadFormDirty = true;
      this.syncSampler();
      this.syncScheduler();
      // FLUX must stream its DiT to fit a 24 GB card. SD/SDXL and Anima use the
      // VRAM-based default the backend recommended at startup (which is "stream"
      // on a ≤6 GB card — backbone-block streaming that fits SDXL/Anima on ~4 GB).
      this.perf.offload = (type === 'FLUX') ? 'stream' : this.recommendedOffload;
      // channels_last only helps the conv backbones (SD/SDXL UNet + VAE); it's a
      // no-op for the DiT families, so default it on only for SD/SDXL.
      this.perf.channelsLast = (type === 'SD/SDXL');
      if (type === 'Anima' && !this.animaApplied) {
        // sensible Anima defaults, applied once
        this.animaApplied = true;
        this.form.sampler = 'er_sde';
        this.form.steps = 30;
        this.form.cfg = 4.0;
        // Anima's shift=3 flow schedule turns a given strength into far more
        // effective noise than SDXL's EDM, so 0.4 over-regenerates faces; ~0.25 refines.
        this.detail.strength = 0.25;
      }
      if (type === 'FLUX' && !this.fluxApplied) {
        // sensible FLUX-dev defaults, applied once (cfg = distilled guidance)
        this.fluxApplied = true;
        this.form.sampler = 'euler';
        this.form.steps = 20;
        this.form.cfg = 3.5;
      }
    },

    syncSampler() {
      const list = this.samplers;
      if (!list.includes(this.form.sampler)) this.form.sampler = list[0];
    },

    syncScheduler() {
      const list = this.schedulers;
      if (!list.includes(this.form.scheduler)) this.form.scheduler = list[0];
    },

    async loadModel() {
      this.loadingModel = true;
      this.status = 'Loading…';
      try {
        const body = {
          model_type: this.modelType,
          checkpoint: this.modelType === 'FLUX' ? this.fluxCheckpoint : this.checkpoint,
          dit: this.dit, vae: this.vae, te: this.te, clip: this.clip,
          offload: this.perf.offload,
          compile: this.perf.compile,
          cuda_graphs: this.perf.cudaGraphs,
          channels_last: this.perf.channelsLast,
          tf32: this.perf.tf32,
          fp16_accumulation: this.perf.fp16Acc,
          // fa2 only applies to the DiT families; never send it for SD/SDXL so
          // a leftover checked chip can't tag an SD load with a no-op flag.
          attention: (this.perf.fa2Attn && this.modelType !== 'SD/SDXL') ? 'fa2_turing' : 'sdpa',
        };
        // Load is queued like any job; it waits for a running generation to
        // finish. The server also broadcasts the new state to every device.
        const ev = await this.submitJob('/api/load', body);
        if (ev.type === 'done') {
          this.status = ev.status;
          this.modelLoaded = !!ev.loaded;
          // Our selection is now the loaded/persisted form — let broadcasts sync again.
          if (this.modelLoaded) this.loadFormDirty = false;
          if (!this.modelLoaded) this.flash(ev.status);
        } else if (ev.type === 'cancelled') {
          this.status = 'Load cancelled';
        } else if (ev.type === 'error') {
          this.status = 'Error: ' + ev.message;
          this.flash(ev.message);
        }
      } catch (e) {
        this.status = 'Error: ' + e;
        this.flash('' + e);
      } finally {
        this.loadingModel = false;
      }
    },

    // ── files ───────────────────────────────────────────────────
    onFile(evt, key) { this.readImage(evt.target.files[0], key); },
    onDrop(evt, key) { this.dragKey = null; this.readImage(evt.dataTransfer.files[0], key); },
    readImage(f, key) {
      if (!f) return;
      // Drops bypass the file input's accept filter; refuse non-images loudly
      // instead of silently doing nothing.
      if (f.type && !f.type.startsWith('image/')) { this.flash('Not an image file'); return; }
      const r = new FileReader();
      r.onload = () => {
        this[key] = r.result;
        if (key === 'inputImage') this.syncOutputSize(r.result);
      };
      r.readAsDataURL(f);
    },

    // Match the output size to the source image (rounded to ×8, the VAE's latent
    // grid) so an img2img/inpaint input isn't stretched by default.
    syncOutputSize(dataUrl) {
      const img = new Image();
      img.onload = () => {
        const r8 = n => Math.max(8, Math.round(n / 8) * 8);
        this.form.width = r8(img.naturalWidth);
        this.form.height = r8(img.naturalHeight);
      };
      img.src = dataUrl;
    },

    // ── inpaint mask painting ───────────────────────────────────
    // The offscreen `_mask` buffer is the source of truth: transparent where
    // untouched, opaque white where masked. The visible canvas is always a
    // pure redraw of `base + orange-tinted mask`, so the brush, eraser,
    // rectangle, undo and invert tools all just mutate `_mask` and redraw.
    // Export flattens the mask onto black (the white-on-black PNG the engine
    // expects). All buffers live on the canvas element so Alpine never proxies
    // the DOM nodes.
    initMask(c) {
      if (!c || !this.inputImage) return;
      const img = new Image();
      img.onload = () => {
        c.width = img.naturalWidth;
        c.height = img.naturalHeight;
        const blank = (w, h) => {
          const x = document.createElement('canvas');
          x.width = w; x.height = h;
          return x;
        };
        c._mask = blank(img.naturalWidth, img.naturalHeight);
        c._tint = blank(img.naturalWidth, img.naturalHeight);
        c._base = img;
        c._painting = false;
        c._dragging = false;
        c._undo = [];
        this.maskPainted = false;
        this.redrawMask(c);
        this.applyMaskZoom();
      };
      img.src = this.inputImage;
    },

    toggleMaskMax() {
      this.maskMax = !this.maskMax;
      this.maskZoom = 1;
      this.$nextTick(() => this.applyMaskZoom());
    },
    zoomMask(d) {
      this.maskZoom = Math.min(4, Math.max(1, Math.round((this.maskZoom + d) * 100) / 100));
      this.applyMaskZoom();
    },
    // Size the maximized canvas to a fit-to-viewport base × zoom and let the
    // stage scroll to pan. maskPos() reads getBoundingClientRect, so painting
    // stays pixel-accurate at any display size.
    applyMaskZoom() {
      const c = this.$refs.maskCanvas;
      if (!c || !c._base) return;
      if (!this.maskMax) { c.style.width = ''; c.style.height = ''; return; }
      const stage = c.parentElement;
      if (!stage) return;
      const aspect = c.width / c.height;
      const fit = Math.min(stage.clientWidth, stage.clientHeight * aspect);
      const w = fit * this.maskZoom;
      c.style.width = w + 'px';
      c.style.height = (w / aspect) + 'px';
    },

    // Repaint the visible canvas: base image, then the mask tinted orange.
    redrawMask(c) {
      const ctx = c.getContext('2d');
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.drawImage(c._base, 0, 0);
      const t = c._tint, tctx = t.getContext('2d');
      tctx.clearRect(0, 0, t.width, t.height);
      tctx.drawImage(c._mask, 0, 0);
      tctx.globalCompositeOperation = 'source-in';
      tctx.fillStyle = '#e8a065';
      tctx.fillRect(0, 0, t.width, t.height);
      tctx.globalCompositeOperation = 'source-over';
      ctx.save();
      ctx.globalAlpha = 0.5;
      ctx.drawImage(t, 0, 0);
      ctx.restore();
    },

    maskDown(e) {
      const c = e.currentTarget;
      if (!c._mask) return;
      // Capture the pointer so move/up keep firing on the canvas even when the
      // cursor strays outside its bounds — lets a stroke run off the edge.
      try { c.setPointerCapture(e.pointerId); } catch (_) {}
      const p = this.maskPos(c, e);
      if (this.maskTool === 'rect') {
        c._dragging = true;
        c._dragStart = p;
      } else {
        this.pushUndo(c);
        c._painting = true;
        c._last = p;
        this.paintSeg(c, p, p);
      }
    },
    maskMove(e) {
      const c = e.currentTarget;
      const p = this.maskPos(c, e);
      if (c._dragging) {
        this.redrawMask(c);
        const ctx = c.getContext('2d');
        ctx.fillStyle = 'rgba(232,160,101,0.5)';
        ctx.fillRect(c._dragStart.x, c._dragStart.y, p.x - c._dragStart.x, p.y - c._dragStart.y);
      } else if (c._painting) {
        this.paintSeg(c, c._last, p);
        c._last = p;
        this.strokeBrushRing(c, p);
      } else if (this.maskTool !== 'rect' && c._mask) {
        // Hover preview: repaint, then draw the ring on top so the user sees
        // the brush footprint at the current size before committing a stroke.
        this.redrawMask(c);
        this.strokeBrushRing(c, p);
      }
    },
    // Clear the hover ring when the pointer leaves the canvas (unless a stroke
    // or rectangle drag is in progress — those manage their own repaint).
    maskLeave(e) {
      const c = e.currentTarget;
      if (c._mask && !c._painting && !c._dragging) this.redrawMask(c);
    },
    // Draw the brush footprint as a ring in canvas coordinates, so it scales
    // with the CSS-displayed canvas automatically. A dark halo under a light
    // ring keeps it visible over both the image and the orange mask tint.
    strokeBrushRing(c, p) {
      const rect = c.getBoundingClientRect();
      const scale = rect.width ? c.width / rect.width : 1; // canvas px per display px
      const ctx = c.getContext('2d');
      ctx.save();
      ctx.beginPath();
      ctx.arc(p.x, p.y, this.maskBrush / 2, 0, Math.PI * 2);
      ctx.lineWidth = 2 * scale;
      ctx.strokeStyle = 'rgba(0,0,0,0.65)';
      ctx.stroke();
      ctx.lineWidth = 1 * scale;
      ctx.strokeStyle = 'rgba(255,255,255,0.95)';
      ctx.stroke();
      ctx.restore();
    },
    maskUp(e) {
      const c = e.currentTarget;
      if (c._dragging) {
        c._dragging = false;
        this.fillRectMask(c, c._dragStart, this.maskPos(c, e));
      }
      c._painting = false;
    },

    maskPos(c, e) {
      const r = c.getBoundingClientRect();
      return {
        x: (e.clientX - r.left) * (c.width / r.width),
        y: (e.clientY - r.top) * (c.height / r.height),
      };
    },
    // Stroke white (brush) or erase (eraser) along a segment of the mask.
    paintSeg(c, a, b) {
      const mctx = c._mask.getContext('2d');
      mctx.save();
      mctx.globalCompositeOperation = this.maskTool === 'eraser' ? 'destination-out' : 'source-over';
      mctx.strokeStyle = '#fff';
      mctx.lineWidth = this.maskBrush;
      mctx.lineCap = 'round';
      mctx.lineJoin = 'round';
      mctx.beginPath();
      mctx.moveTo(a.x, a.y);
      mctx.lineTo(b.x, b.y);
      mctx.stroke();
      mctx.restore();
      if (this.maskTool !== 'eraser') this.maskPainted = true;
      this.redrawMask(c);
    },
    fillRectMask(c, a, b) {
      const w = Math.abs(b.x - a.x), h = Math.abs(b.y - a.y);
      if (!w || !h) { this.redrawMask(c); return; }
      this.pushUndo(c);
      const mctx = c._mask.getContext('2d');
      mctx.fillStyle = '#fff';
      mctx.fillRect(Math.min(a.x, b.x), Math.min(a.y, b.y), w, h);
      this.maskPainted = true;
      this.redrawMask(c);
    },
    // Snapshot the mask before a mutating op so it can be undone. The first
    // snapshot of any chain is the empty mask, so a non-empty undo stack
    // means something is painted (drives maskPainted after undo).
    pushUndo(c) {
      const mctx = c._mask.getContext('2d');
      c._undo.push(mctx.getImageData(0, 0, c._mask.width, c._mask.height));
      if (c._undo.length > 30) c._undo.shift();
    },
    undoMask() {
      const c = this.$refs.maskCanvas;
      if (!c || !c._undo || !c._undo.length) return;
      c._mask.getContext('2d').putImageData(c._undo.pop(), 0, 0);
      this.maskPainted = c._undo.length > 0;
      this.redrawMask(c);
    },
    invertMask() {
      const c = this.$refs.maskCanvas;
      if (!c || !c._mask) return;
      this.pushUndo(c);
      const m = c._mask, mctx = m.getContext('2d');
      const inv = document.createElement('canvas');
      inv.width = m.width; inv.height = m.height;
      const ictx = inv.getContext('2d');
      ictx.fillStyle = '#fff';
      ictx.fillRect(0, 0, inv.width, inv.height);
      ictx.globalCompositeOperation = 'destination-out';
      ictx.drawImage(m, 0, 0);
      mctx.clearRect(0, 0, m.width, m.height);
      mctx.drawImage(inv, 0, 0);
      this.maskPainted = true;
      this.redrawMask(c);
    },
    clearMask() {
      const c = this.$refs.maskCanvas;
      if (!c || !c._base) return;
      this.pushUndo(c);
      c._mask.getContext('2d').clearRect(0, 0, c._mask.width, c._mask.height);
      this.maskPainted = false;
      this.redrawMask(c);
    },
    // Flatten the transparent mask onto black → the white-on-black PNG the
    // engine expects (decoded as RGB, so transparency must be made opaque).
    exportMask() {
      const c = this.$refs.maskCanvas;
      if (!c || !c._mask) return null;
      const o = document.createElement('canvas');
      o.width = c._mask.width; o.height = c._mask.height;
      const octx = o.getContext('2d');
      octx.fillStyle = '#000';
      octx.fillRect(0, 0, o.width, o.height);
      octx.drawImage(c._mask, 0, 0);
      return o.toDataURL('image/png');
    },

    // ── auto-growing textareas ──────────────────────────────────
    autogrow(el) {
      if (!el || el.offsetParent === null) return;   // skip while hidden
      el.style.height = 'auto';
      el.style.height = el.scrollHeight + 'px';
    },
    resizeTextareas() {
      this.$nextTick(() => {
        this.$root.querySelectorAll('textarea.autosize').forEach((el) => this.autogrow(el));
      });
    },

    // ── <lora:…> autocomplete ───────────────────────────────────
    // Typing `<` opens a list of available LoRAs; the fragment after the last
    // unclosed `<` (optionally past a `lora:` prefix) filters it. Selecting one
    // inserts `<lora:name:1.0>`. Names are the exact filenames the backend
    // expects (see lora_path); the dropdown hides the .safetensors suffix.
    loraLabel(name) {
      return name.replace(/\.safetensors$/i, '');
    },
    // Trigger on `<`, insert <lora:name:1.0>. Used by the prompt (default) and
    // the X/Y/Z "Prompt S/R" fields, which pass {key, set} to write back to the
    // axis text — S/R then operates on the whole literal tag, so name swaps,
    // weight sweeps, and removal all stay valid <lora:…> strings.
    loraAutocomplete(el, opts) {
      const o = opts || { key: 'prompt', set: (v) => { this.form.prompt = v; } };
      const before = el.value.slice(0, el.selectionStart);
      const lt = before.lastIndexOf('<');
      const m = lt === -1 ? null : before.slice(lt + 1).match(/^(?:lora:)?([^:>]*)$/i);
      if (!m) { this.loraAC.open = false; return; }
      const frag = m[1];
      const items = this.loras.filter((n) => n.toLowerCase().includes(frag.toLowerCase()));
      if (!items.length) { this.loraAC.open = false; return; }
      Object.assign(this.loraAC, {
        open: true, items, index: 0, start: lt, key: o.key, el,
        set: o.set, wrap: (n) => `<lora:${n}:1.0>`,
      });
    },
    loraKeydown(e) {
      const ac = this.loraAC;
      if (!ac.open) return;
      const n = ac.items.length;
      if (e.key === 'ArrowDown') { e.preventDefault(); ac.index = (ac.index + 1) % n; }
      else if (e.key === 'ArrowUp') { e.preventDefault(); ac.index = (ac.index - 1 + n) % n; }
      else if (e.key === 'Enter' || e.key === 'Tab') { e.preventDefault(); this.applyLora(ac.items[ac.index]); }
      else if (e.key === 'Escape') { e.preventDefault(); ac.open = false; }
    },

    // Keydown for the prompt textarea: Ctrl/Cmd+Enter triggers Generate (the
    // usual shortcut in every comparable UI), unless the LoRA autocomplete is
    // open — then Enter inserts the selected LoRA and Ctrl/Cmd+Enter is left
    // alone so the user can dismiss it first. Other keys fall through to
    // loraKeydown for autocomplete navigation.
    promptKeydown(e) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (this.loraAC.open) return;   // don't fire through an open dropdown
        e.preventDefault();
        if (!this.busy) this.runGenerate();
        return;
      }
      this.loraKeydown(e);
    },
    applyLora(name) {
      const ac = this.loraAC;
      const el = ac.el;
      const before = el.value.slice(0, ac.start);
      const after = el.value.slice(el.selectionStart);
      const insert = ac.wrap(name);
      ac.set(before + insert + after);
      ac.open = false;
      this.$nextTick(() => {
        const caret = before.length + insert.length;
        el.focus();
        el.setSelectionRange(caret, caret);
        if (el.tagName === 'TEXTAREA') this.autogrow(el);
      });
    },

    // ── generate ────────────────────────────────────────────────
    runGenerate() {
      return this.sweeping ? this.generateXyz() : this.generate();
    },

    async generate() {
      // A load in flight is fine: the job queues behind it and the model is
      // loaded by the time it runs. Only refuse when nothing's loaded or loading.
      if (!this.modelLoaded && !this.loadingModel) { this.flash('Load a model first'); return; }
      if (this.mode === 'i2i' && !this.inputImage) { this.flash('Provide an input image'); return; }
      if (this.mode === 'inpaint') {
        if (!this.inputImage) { this.flash('Provide an input image'); return; }
        if (!this.maskPainted) { this.flash('Paint a mask over the image'); return; }
        this.maskImage = this.exportMask();
      }
      this.busy = true;
      this.cancelling = false;
      this.progress = { step: 0, total: 0 };
      this.previewUrl = null;
      this.info = '';
      try {
        // Seamless OSS: calibrate this steps/size/shift on first use, then
        // generate — all under one click. Re-check status fresh so a just-changed
        // steps/size/shift isn't missed.
        if (this.needsOss()) {
          await this.checkOssStatus();
          if (this.ossCalibrated === false) {
            if (!await this._streamCalibrate()) return;
            this.progress = { step: 0, total: 0 };
          }
        }
        const payload = {
          mode: this.mode,
          prompt: this.form.prompt, neg: this.form.neg,
          sampler: this.form.sampler, scheduler: this.form.scheduler,
          steps: this.form.steps, cfg: this.form.cfg, seed: this.form.seed,
          width: this.form.width, height: this.form.height,
          strength: this.form.strength, shift: this.form.shift,
          teacache: this.form.teacacheOn ? this.form.teacache : 0,
          teacache_calibrated: this.form.teacacheCalibrated,
          deepcache: this.form.deepcacheOn ? this.form.deepcache : 1,
          input_image: this.mode !== 't2i' ? this.inputImage : null,
          mask_image: this.mode === 'inpaint' ? this.maskImage : null,
          preview: this.preview,
          detail_enabled: this.detail.enabled,
          detail_models: this.detail.models,
          detail_neg: this.detail.neg,
          detail_confidence: this.detail.confidence,
          detail_strength: this.detail.strength,
          detail_dilation: this.detail.dilation,
          detail_padding: this.detail.padding,
          detail_blur: this.detail.blur,
          detail_max: this.detail.maxDet,
          detail_teacache: this.detail.teacache,
          upscale_enabled: this.upscale.enabled,
          upscale_scale: this.upscale.scale,
          upscale_denoise: this.upscale.denoise,
          upscale_tile: this.upscale.tile,
          upscale_overlap: this.upscale.overlap,
          upscale_prompt: this.upscale.prompt,
          upscale_teacache: this.upscale.teacache,
          upscale_base: this.upscale.base,
        };
        // Batch: submit N copies, await all. Single (count=1) keeps the original
        // submitJob path so load/upscale/xyz/calibrate semantics are unchanged.
        const batchCount = Math.max(1, Math.min(16, Math.floor(this.batchCount) || 1));
        if (batchCount > 1) {
          const promises = await this.submitBatch('/api/generate', payload, batchCount);
          const results = await Promise.all(promises);
          this.previewUrl = null;
          let lastDone = null, nDone = 0, nErr = 0, nCancel = 0;
          for (const ev of results) {
            if (ev.type === 'done') { nDone++; lastDone = ev; }
            else if (ev.type === 'error') nErr++;
            else if (ev.type === 'cancelled') nCancel++;
          }
          if (lastDone) {
            this.resultUrl = lastDone.image_url + '?t=' + Date.now();
            this.info = `Batch: ${nDone} done`
              + (nErr ? `, ${nErr} errored` : '')
              + (nCancel ? `, ${nCancel} cancelled` : '')
              + `  |  ${lastDone.info}`;
            this.lastSeed = lastDone.seed;
          } else {
            this.info = `Batch: ${nCancel ? 'cancelled' : 'no images'}`
              + (nErr ? ` · ${nErr} errored` : '');
          }
          if (nErr) this.flash(`${nErr} batch job(s) failed`);
        } else {
          // Progress + preview arrive on the shared stream; we await the result.
          const ev = await this.submitJob('/api/generate', payload);
          if (ev.type === 'done') {
            this.previewUrl = null;
            this.resultUrl = ev.image_url + '?t=' + Date.now();
            this.info = ev.info;
            this.lastSeed = ev.seed;
          } else if (ev.type === 'cancelled') {
            this.previewUrl = null;
            this.info = 'Cancelled';
          } else if (ev.type === 'error') {
            this.info = 'Error: ' + ev.message;
            this.flash(ev.message);
          }
        }
      } catch (e) {
        this.info = 'Error: ' + e;
      } finally {
        this.busy = false;
        this.cancelling = false;
        this.previewUrl = null;
        // Belt-and-suspenders: terminal events should have emptied the set, but
        // a missed event (e.g. a future code path that doesn't register a waiter)
        // would otherwise leave stale ids routing unrelated progress here.
        this._myBatchIds = [];
      }
    },

    // Cancel a job by id (defaults to this device's running job). A running job
    // stops at its next sampling step; a still-queued job is dropped. With no
    // arg and an in-flight batch, every batch member is cancelled.
    async cancel(jobId) {
      if (jobId == null && this._myBatchIds.length > 0) {
        this.cancelling = true;
        for (const id of [...this._myBatchIds]) {
          try {
            await fetch('/api/cancel', {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ job: id }),
            });
          } catch (e) { /* the stream still resolves each */ }
        }
        return;
      }
      const id = jobId ?? this.myJobId ?? this.runningJob;
      if (id == null) return;
      if (id === this.myJobId) this.cancelling = true;
      try {
        await fetch('/api/cancel', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ job: id }),
        });
      } catch (e) {
        /* the stream still resolves; just drop the cancelling state */
      }
    },

    // ── x/y/z sweep ─────────────────────────────────────────────
    async generateXyz() {
      // Queue behind an in-flight load (see generate()); only refuse when idle.
      if (!this.modelLoaded && !this.loadingModel) { this.flash('Load a model first'); return; }
      // A Prompt S/R axis's search term (its first value) must appear in the
      // prompt or negative prompt, or every cell is identical. Refuse early —
      // same case-sensitive match the backend's replace uses.
      const haystack = this.form.prompt + '\n' + this.form.neg;
      for (const axis of [this.axes.x, this.axes.y, this.axes.z]) {
        if (axis.type !== 'Prompt S/R') continue;
        const search = (axis.text.split(',')[0] || '').trim();
        if (search && !haystack.includes(search)) {
          this.flash(`Prompt S/R: "${search}" is not in the prompt or negative prompt`);
          return;
        }
      }
      this.busy = true;
      this.progress = { step: 0, total: 0 };
      this.previewUrl = null;
      this.xyzInfo = '';
      const payload = {
        prompt: this.form.prompt, neg: this.form.neg,
        width: this.form.width, height: this.form.height,
        steps: this.form.steps, cfg: this.form.cfg,
        sampler: this.form.sampler, scheduler: this.form.scheduler,
        seed: this.form.seed, shift: this.form.shift,
        teacache: this.form.teacacheOn ? this.form.teacache : 0,
        teacache_calibrated: this.form.teacacheCalibrated,
        x_type: this.axes.x.type, x_vals: this.axisValues(this.axes.x),
        y_type: this.axes.y.type, y_vals: this.axisValues(this.axes.y),
        z_type: this.axes.z.type, z_vals: this.axisValues(this.axes.z),
        preview: this.preview,
      };
      try {
        const ev = await this.submitJob('/api/xyz', payload);
        if (ev.type === 'done') {
          this.xyzGrids = ev.grids;
          this.xyzInfo = ev.info;
        } else if (ev.type === 'cancelled') {
          this.xyzInfo = 'Cancelled';
        } else if (ev.type === 'error') {
          this.xyzInfo = 'Error: ' + ev.message;
          this.flash(ev.message);
        }
      } catch (e) {
        this.xyzInfo = 'Error: ' + e;
      } finally {
        this.busy = false;
        this.cancelling = false;
        this.previewUrl = null;
      }
    },

    // ── OSS calibration ─────────────────────────────────────────
    // Whether the current (steps, resolution, shift) already has a calibrated
    // schedule. Reads form fields synchronously so Alpine's x-effect re-checks
    // when any of them change.
    async checkOssStatus() {
      const { scheduler, steps, width, height, shift } = this.form;
      if (this.modelType !== 'Anima' || scheduler !== 'oss' || !this.modelLoaded) {
        this.ossCalibrated = null;
        return;
      }
      const q = new URLSearchParams({ steps, width, height, shift });
      try {
        this.ossCalibrated = (await fetchJSON('/api/oss_status?' + q)).calibrated;
      } catch (e) {
        this.ossCalibrated = null;
      }
    },

    needsOss() {
      return this.modelType === 'Anima' && this.mode === 't2i' && this.form.scheduler === 'oss';
    },

    // Queue a calibration job. Updates progress/status but does NOT own `busy`
    // — the caller does, so it can chain calibrate→generate under one spinner.
    // Returns true on success.
    async _streamCalibrate() {
      this.calibrating = true;
      this.ossInfo = '';
      this.progress = { step: 0, total: 0 };
      let ok = false;
      try {
        const ev = await this.submitJob('/api/calibrate_oss', {
          prompt: this.form.prompt, neg: this.form.neg,
          steps: this.form.steps, cfg: this.form.cfg, seed: this.form.seed,
          width: this.form.width, height: this.form.height, shift: this.form.shift,
        });
        if (ev.type === 'done') {
          this.ossInfo = ev.info;
          this.ossCalibrated = true;
          ok = true;
        } else if (ev.type === 'cancelled') {
          this.ossInfo = 'Cancelled';
        } else if (ev.type === 'error') {
          this.ossInfo = 'Error: ' + ev.message;
          this.flash(ev.message);
        }
      } catch (e) {
        this.ossInfo = 'Error: ' + e;
      } finally {
        this.calibrating = false;
      }
      return ok;
    },

    // Manual "Calibrate" button.
    async calibrateOss() {
      if (!this.modelLoaded) { this.flash('Load a model first'); return; }
      this.busy = true;
      try {
        if (await this._streamCalibrate()) this.flash('OSS calibrated');
      } finally {
        this.busy = false;
        this.cancelling = false;
      }
    },

    // ── settings panel ──────────────────────────────────────────
    async loadSettings() {
      try { this.settings = await fetchJSON('/api/settings'); }
      catch (e) { /* keep defaults */ }
    },

    async saveSettings() {
      try {
        this.settings = await fetchJSON('/api/settings', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.settings),
        });
        this.flash('Settings saved');
      } catch (e) { this.flash('Could not save settings'); }
    },

    async refreshTeacacheStatus() {
      try { this.teacacheStatus = await fetchJSON('/api/teacache_status'); }
      catch (e) { this.teacacheStatus = { loaded: false, calibratable: false, family: null, coefficients: null }; }
    },

    // Seed the Generate form from saved defaults, then re-validate the sampler/
    // scheduler against the current model type (so a default that doesn't apply
    // to the loaded family falls back instead of sticking an invalid value).
    applyGenDefaults() {
      const d = this.settings.gen_defaults;
      if (!d) return;
      for (const k of ['sampler', 'scheduler', 'steps', 'cfg', 'width', 'height', 'shift', 'prompt', 'neg']) {
        if (d[k] !== undefined && d[k] !== null) this.form[k] = d[k];
      }
      this.syncSampler();
      this.syncScheduler();
    },

    async saveGenDefaults() {
      const f = this.form;
      const d = {
        sampler: f.sampler, scheduler: f.scheduler, steps: f.steps,
        cfg: f.cfg, width: f.width, height: f.height, shift: f.shift,
      };
      // Prompt/negative are only worth pinning when the user actually filled them.
      if (f.prompt && f.prompt.trim()) d.prompt = f.prompt;
      if (f.neg && f.neg.trim()) d.neg = f.neg;
      this.settings.gen_defaults = d;
      await this.saveSettings();
    },

    swapDimensions() {
      const w = this.form.width;
      this.form.width = this.form.height;
      this.form.height = w;
    },

    clearGenDefaults() {
      this.settings.gen_defaults = null;
      this.saveSettings();
    },

    openSettings() {
      this.settingsOpen = true;
      this.refreshTeacacheStatus();
    },

    // Fit the TeaCache rescaling polynomial for the loaded Anima family. Long,
    // GPU-heavy job routed through the shared queue (progress shows on the bar),
    // exactly like OSS calibrate.
    async calibrateTeacache() {
      if (!this.teacacheStatus.calibratable) { this.flash('Load an Anima model first'); return; }
      this.busy = true;
      this.calibratingTea = true;
      this.progress = { step: 0, total: 0 };
      try {
        const ev = await this.submitJob('/api/calibrate_teacache', {
          prompt: 'a detailed photograph of a fox in a forest',
          neg: 'blurry, low quality',
          steps: 50, cfg: 4.0, seed: 0, width: 1024, height: 1024, shift: 3.0,
        });
        if (ev.type === 'done') { await this.refreshTeacacheStatus(); this.flash('TeaCache calibrated'); }
        else if (ev.type === 'error') { this.flash('Error: ' + ev.message); }
        else if (ev.type === 'cancelled') { this.flash('Calibration cancelled'); }
      } finally {
        this.busy = false;
        this.calibratingTea = false;
        this.cancelling = false;
      }
    },

    // ── detailer model stack ────────────────────────────────────
    addDetailModel() {
      this.detail.models.push({ model: this.detailerChoices[0], prompt: '' });
    },
    removeDetailModel(i) {
      this.detail.models.splice(i, 1);
      if (!this.detail.models.length) this.detail.models.push({ model: this.detailerChoices[0], prompt: '' });
    },

    // ── standalone upscale ───────────────────────────────────────
    async openUpscale() {
      // Capture the source the moment the popover opens, so the run upscales the
      // image you launched from. A gallery selection carries its own metadata
      // (so the refine pass matches how it was made); a fresh result is
      // described by the live form.
      let fromGallery = false;
      if (this.lightbox.open && this.selected) {
        await this._metaLoad;                    // metadata may still be in flight
        this._upscaleSrc = { url: this.selected.url, meta: this.selectedFields };
        fromGallery = true;
      } else if (this.resultUrl) {
        this._upscaleSrc = { url: this.resultUrl, meta: null };
      } else {
        this._upscaleSrc = null;
      }
      const meta = this._upscaleSrc && this._upscaleSrc.meta;
      this.upscaleForm.prompt = (meta && meta.prompt) || this.upscale.prompt;
      this.upscaleForm.scale = this.upscale.scale;
      this.upscaleForm.denoise = this.upscale.denoise;
      this.upscaleForm.tile = this.upscale.tile;
      this.upscaleForm.overlap = this.upscale.overlap;
      this.upscaleForm.teacache = this.upscale.teacache;
      this.upscaleForm.base = this.upscale.base;
      // The popover and the progress bar both live on the Generate tab, so a
      // gallery launch moves there (closing the lightbox) — otherwise the
      // popover stays hidden behind the gallery.
      if (fromGallery) { this.closeLightbox(); this.tab = 'generate'; }
      this.upscalePopover.open = true;
      this.upscalePopover.busy = false;
    },
    closeUpscale() { this.upscalePopover.open = false; },
    async runUpscale() {
      const src = this._upscaleSrc;
      if (!src) { this.flash('No image to upscale'); return; }
      if (!this.modelLoaded) { this.flash('Load a model first'); return; }
      this.closeUpscale();   // reveal the Generate-tab progress bar + live preview
      this.busy = true;
      this.progress = { step: 0, total: 0 };
      this.previewUrl = null;
      try {
        const blob = await (await fetch(src.url)).blob();
        const inputImage = await new Promise((res, rej) => {
          const r = new FileReader();
          r.onload = () => res(r.result);
          r.onerror = rej;
          r.readAsDataURL(blob);
        });
        // Refine params describe the source: a gallery image's own metadata
        // (over the form as a fallback), or the live form for a fresh result.
        const refine = src.meta ? { ...this.form, ...src.meta } : this.form;
        const payload = {
          input_image: inputImage,
          scale: this.upscaleForm.scale, denoise: this.upscaleForm.denoise,
          tile: this.upscaleForm.tile, overlap: this.upscaleForm.overlap,
          base: this.upscaleForm.base,
          prompt: this.upscaleForm.prompt,
          neg: refine.neg,
          steps: refine.steps, cfg: refine.cfg,
          sampler: refine.sampler, scheduler: refine.scheduler,
          seed: refine.seed,
          teacache: this.upscaleForm.teacache,
          teacache_calibrated: this.form.teacacheCalibrated,
          preview: this.preview,
        };
        const ev = await this.submitJob('/api/upscale', payload);
        if (ev.type === 'done') {
          this.previewUrl = null;
          this.resultUrl = ev.image_url + '?t=' + Date.now();
          this.info = ev.info;
          this.closeUpscale();
          // Surface the result on the Generate tab (no-op when already there),
          // closing the gallery lightbox if the upscale was launched from it.
          this.closeLightbox();
          this.tab = 'generate';
          this.flash('Upscale done');
        } else if (ev.type === 'cancelled') {
          this.previewUrl = null;
          this.info = 'Upscale cancelled';
        } else if (ev.type === 'error') {
          this.info = 'Error: ' + ev.message;
          this.flash(ev.message);
        }
      } catch (e) {
        this.info = 'Error: ' + e;
      } finally {
        this.upscalePopover.busy = false;
        this.busy = false;
        this.cancelling = false;
        this.previewUrl = null;
      }
    },

    // ── gallery ─────────────────────────────────────────────────
    async openGallery() {
      this.tab = 'gallery';
      this.selected = null;
      this.selectedMeta = '';
      this.galleryLimit = 60;
      this.galleryQuery = '';
      this.gallerySearching = false;
      this.gallery = (await fetchJSON('/api/gallery')).images;
      this.buildGalleryGroups();
    },

    // Debounced search field: fetch the filtered list from the backend's cached
    // metadata index. An empty query restores the full list (no index needed).
    async searchGallery() {
      const q = (this.galleryQuery || '').trim();
      this.gallerySearching = true;
      try {
        const url = q ? `/api/gallery?q=${encodeURIComponent(q)}` : '/api/gallery';
        this.gallery = (await fetchJSON(url)).images;
        this.selected = null;
        this.selectedMeta = '';
        this.galleryLimit = 60;
        this.buildGalleryGroups();
      } catch (e) {
        /* keep the previous list on a transient fetch error */
      } finally {
        this.gallerySearching = false;
      }
    },

    // Group the (date-desc) gallery list into day sections, carrying each
    // image's flat index so the lightbox carousel still pages across all of them.
    // Only the first `galleryLimit` images are grouped (and rendered); the rest
    // stay in `gallery` for the lightbox and load in as the sentinel scrolls in.
    buildGalleryGroups() {
      const groups = [];
      let cur = null;
      this.gallery.slice(0, this.galleryLimit).forEach((img, i) => {
        if (!cur || cur.date !== img.date) {
          cur = { date: img.date, label: this.dateLabel(img.date), images: [] };
          groups.push(cur);
        }
        cur.images.push({ img, index: i });
      });
      this.galleryGroups = groups;
    },

    // Watch the bottom sentinel; render the next chunk as it nears the viewport.
    // rootMargin pre-loads 400px early so the "Loading more…" row rarely shows.
    observeSentinel(el) {
      new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting) this.loadMoreGallery();
      }, { rootMargin: '400px' }).observe(el);
    },
    loadMoreGallery() {
      if (this.galleryLimit >= this.gallery.length) return;
      this.galleryLimit += 60;
      this.buildGalleryGroups();
    },

    dateLabel(d) {
      // ISO YYYY-MM-DD folder names (legacy dirs are migrated server-side).
      const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(d || '');
      if (!m) return d || 'Unknown date';
      const dt = new Date(+m[1], +m[2] - 1, +m[3]);
      const today = new Date(); today.setHours(0, 0, 0, 0);
      const diff = Math.round((today - dt) / 86400000);
      if (diff === 0) return 'Today';
      if (diff === 1) return 'Yesterday';
      return dt.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' });
    },

    async selectImage(img) {
      this.selected = img;
      // Drop the previous image's metadata up front so the panel and the
      // send-to-generator buttons never act on stale data while this loads.
      this.selectedMeta = '';
      this.selectedFields = null;
      // Token guards against out-of-order responses: paging fast on a slow
      // link, an earlier image's fetch can resolve last — ignore it so the
      // image on screen always wins. Consumers await _metaLoad (below).
      const token = img.url;
      this._metaToken = token;
      this._metaLoad = (async () => {
        try {
          const r = await fetchJSON('/api/metadata?path=' + encodeURIComponent(img.path));
          if (this._metaToken !== token) return;   // a newer selection won
          this.selectedMeta = r.raw;
          this.selectedFields = r.fields;
        } catch (e) {
          if (this._metaToken === token) this.selectedMeta = '';
        }
      })();
      return this._metaLoad;
    },

    // ── lightbox carousel ───────────────────────────────────────
    openLightbox(i) {
      this.lightbox.index = i;
      this.lightbox.open = true;
      this.selectImage(this.gallery[i]);
    },
    closeLightbox() { this.lightbox.open = false; this.deleteConfirm = false; },
    lbPrev() { this.lbGo(-1); },
    lbNext() { this.lbGo(1); },
    lbGo(d) {
      const n = this.gallery.length;
      if (!n) return;
      this.deleteConfirm = false;   // reset the two-click confirm on navigation
      this.lightbox.index = (this.lightbox.index + d + n) % n;
      this.selectImage(this.gallery[this.lightbox.index]);
    },
    lightboxKey(e) {
      if (!this.lightbox.open) return;
      if (e.key === 'Escape') this.closeLightbox();
      else if (e.key === 'ArrowLeft') this.lbPrev();
      else if (e.key === 'ArrowRight') this.lbNext();
    },
    lbTouchStart(e) { this._touchX = e.changedTouches[0].clientX; },
    lbTouchEnd(e) {
      const dx = e.changedTouches[0].clientX - this._touchX;
      if (Math.abs(dx) > 40) (dx < 0 ? this.lbNext() : this.lbPrev());
    },

    async loadToWorkspace() {
      await this._metaLoad;   // wait for this image's metadata on slow links
      this.applyFields(this.selectedFields);
      this.closeLightbox();
      this.tab = 'generate';
      this.resizeTextareas();
      this.flash('Loaded settings into Generate');
    },

    // Two-click delete: first click arms the confirm; the second fires the
    // DELETE. Clicking anywhere else (click.outside on the button) disarms it.
    // After the file is gone, drop it from the in-memory list, rebuild the
    // grid groups, and step the lightbox to a neighbor (or close if it was the
    // last image entirely).
    async deleteSelected() {
      if (!this.selected) return;
      if (!this.deleteConfirm) {
        this.deleteConfirm = true;
        return;
      }
      this.deleteConfirm = false;
      const path = this.selected.path;
      const idx = this.lightbox.index;
      try {
        const r = await fetch('/api/gallery?path=' + encodeURIComponent(path), {
          method: 'DELETE',
        });
        if (!r.ok) {
          const detail = await r.json().catch(() => ({}));
          this.flash('Could not delete: ' + (detail.detail || r.statusText));
          return;
        }
      } catch (e) {
        this.flash('Could not delete: ' + e);
        return;
      }
      // Remove from the in-memory gallery; if the gallery list is currently a
      // search result, the next openGallery()/searchGallery() will refetch —
      // the backend already invalidated its search index.
      this.gallery = this.gallery.filter((g) => g.path !== path);
      this.buildGalleryGroups();
      if (this.gallery.length === 0) {
        this.closeLightbox();
        this.selected = null;
        this.selectedMeta = '';
        this.selectedFields = null;
        this.flash('Deleted. Gallery is now empty');
        return;
      }
      // Step to the neighbor (clamp, since idx may now point past the end).
      const nextIdx = Math.min(idx, this.gallery.length - 1);
      this.lightbox.index = nextIdx;
      this.selectImage(this.gallery[nextIdx]);
      this.flash('Deleted');
    },

    // Send the selected gallery image into img2img / inpaint as the input image.
    async sendToMode(mode) {
      if (!this.selected) return;
      try {
        const blob = await (await fetch(this.selected.url)).blob();
        this.inputImage = await new Promise((res, rej) => {
          const r = new FileReader();
          r.onload = () => res(r.result);
          r.onerror = rej;
          r.readAsDataURL(blob);
        });
      } catch (e) {
        this.flash('Could not load image: ' + e);
        return;
      }
      this.maskImage = null;
      this.maskPainted = false;
      await this._metaLoad;   // wait for this image's metadata on slow links
      this.applyFields(this.selectedFields);
      this.syncOutputSize(this.inputImage);
      this.mode = mode;
      this.closeLightbox();
      this.tab = 'generate';
      this.resizeTextareas();
      this.flash('Sent to ' + (mode === 'i2i' ? 'img2img' : 'inpaint'));
    },

    // ── metadata reader ─────────────────────────────────────────
    uploadMeta(evt) { this.readMeta(evt.target.files[0]); },
    onMetaDrop(evt) { this.dragKey = null; this.readMeta(evt.dataTransfer.files[0]); },
    async readMeta(f) {
      if (!f) return;
      // Drops bypass the input's accept="image/png"; only PNGs carry the
      // parameters text chunk, so refuse other types with feedback.
      if (f.type && f.type !== 'image/png') { this.flash('Metadata lives in PNGs. Drop a PNG file'); return; }
      const pre = new FileReader();
      pre.onload = () => { this.metaPreview = pre.result; };
      pre.readAsDataURL(f);
      const fd = new FormData();
      fd.append('file', f);
      try {
        const r = await fetchJSON('/api/metadata/parse', { method: 'POST', body: fd });
        this.metaText = r.text;
        this.metaFields = Object.keys(r.fields).length ? r.fields : null;
      } catch (e) {
        this.metaText = '';
        this.metaFields = null;
        this.flash('Could not read metadata: ' + e.message);
      }
    },

    sendMetaToGenerate() {
      if (this.metaFields) this.applyFields(this.metaFields);
      this.mode = 't2i';
      this.tab = 'generate';
      this.resizeTextareas();
      this.flash('Sent to txt2img');
    },

    // map a normalised workspace-fields dict onto the form
    applyFields(f) {
      if (!f) return;
      const keys = ['prompt', 'neg', 'steps', 'cfg', 'sampler', 'scheduler',
                    'seed', 'shift', 'strength', 'width', 'height',
                    'teacacheOn', 'teacache', 'teacacheCalibrated',
                    'deepcacheOn', 'deepcache'];
      for (const k of keys) if (f[k] !== undefined) this.form[k] = f[k];
      if (f.detailer) this.applyDetailer(f.detailer);
      if (f.upscale) this.applyUpscale(f.upscale);
    },

    // Restore the upscaler panel from a saved `upscale` metadata chunk. Additive,
    // like applyDetailer — an image without the chunk leaves the panel untouched.
    applyUpscale(u) {
      this.upscale.enabled = u.enabled !== false;
      for (const k of ['scale', 'denoise', 'tile', 'overlap', 'teacache', 'base', 'prompt']) {
        if (u[k] !== undefined) this.upscale[k] = u[k];
      }
    },

    // Restore the detailer panel from a saved `detailer` metadata chunk. Additive,
    // like applyFields — an image without the chunk leaves the panel untouched.
    applyDetailer(d) {
      this.detail.enabled = d.enabled !== false;
      if (Array.isArray(d.models) && d.models.length) {
        this.detail.models = d.models.map(m => ({ model: m.model || '', prompt: m.prompt || '' }));
      }
      if (d.neg !== undefined) this.detail.neg = d.neg;
      for (const k of ['confidence', 'strength', 'dilation', 'padding', 'blur', 'maxDet']) {
        if (d[k] !== undefined) this.detail[k] = d[k];
      }
    },

    // Parse AUTO1111-style parameters pasted into the prompt box and apply them
    // to the form (prompt/negative/settings), like SD WebUI's read-params arrow.
    async importFromPrompt() {
      const text = this.form.prompt;
      if (!text || !text.trim()) { this.flash('Paste generation parameters into the prompt first'); return; }
      try {
        const r = await fetchJSON('/api/metadata/parse_text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        });
        // workspace_fields always echoes a default seed, so "has params" means
        // an actual settings key was parsed — not just prompt/neg/seed.
        const f = r.fields || {};
        const hasSettings = ['steps', 'cfg', 'sampler', 'scheduler', 'width', 'height', 'strength', 'shift']
          .some((k) => f[k] !== undefined);
        if (!hasSettings) { this.flash('No generation parameters found'); return; }
        this.applyFields(f);
        this.resizeTextareas();
        this.flash('Imported generation settings');
      } catch (e) {
        this.flash('Could not parse: ' + e);
      }
    },

    // Arrow-key navigation for a segmented control / tablist. Moves focus
    // between the sibling <button>s of the group the event fired on. Radio
    // groups activate on move (selection follows focus, per the radio
    // pattern); tablists pass activate=false so arrows only move focus and
    // Enter/Space (native button) activates — avoids firing expensive tab
    // handlers on every keypress.
    navGroup(e, activate = true) {
      const keys = ['ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp', 'Home', 'End'];
      if (!keys.includes(e.key)) return;
      const btns = [...e.currentTarget.querySelectorAll('button')]
        .filter(b => !b.disabled && b.offsetParent !== null);
      if (!btns.length) return;
      let i = btns.indexOf(document.activeElement);
      if (i < 0) i = 0;
      let n;
      if (e.key === 'Home') n = 0;
      else if (e.key === 'End') n = btns.length - 1;
      else {
        const fwd = e.key === 'ArrowRight' || e.key === 'ArrowDown';
        n = (i + (fwd ? 1 : -1) + btns.length) % btns.length;
      }
      e.preventDefault();
      btns[n].focus();
      if (activate) btns[n].click();
    },

    // ── toast ───────────────────────────────────────────────────
    // A small FIFO queue so a burst of messages (e.g. several batch errors)
    // each get a turn instead of the last overwriting the rest instantly.
    // Capped at 4 pending so a runaway loop can't pile up dozens.
    flash(msg, kind) {
      if (!msg) return;
      this._toastQueue.push({ msg, kind: kind || this._toastKind(msg) });
      if (this._toastQueue.length > 4) this._toastQueue.length = 4;
      if (!this._toastShowing) this._toastNext();
    },
    // Infer severity from the message when the caller didn't pass one, so
    // failures read red and completions read green without tagging every
    // callsite. An explicit `kind` always wins.
    _toastKind(msg) {
      if (/fail|error|could ?n.t|could not|cannot|unable/i.test(msg)) return 'error';
      if (/saved|done|copied|loaded|imported|installed|updated|deleted|sent|calibrated/i.test(msg)) return 'success';
      return 'info';
    },
    _toastQueue: [],
    _toastShowing: false,
    _toastNext() {
      const item = this._toastQueue.shift();
      if (item == null) { this._toastShowing = false; this.toast = ''; return; }
      this._toastShowing = true;
      this.toast = item.msg;
      this.toastKind = item.kind;
      clearTimeout(this._toastTimer);
      this._toastTimer = setTimeout(() => {
        this.toast = '';
        // Brief gap before the next so the fade-out completes (x-transition)
        // and the new toast fades in, rather than two messages snapping together.
        setTimeout(() => this._toastNext(), 250);
      }, 2500);
    },

    // Copy the lightbox's raw AUTO1111 parameters string to the clipboard.
    // `navigator.clipboard` is the modern path but needs a secure context
    // (HTTPS or localhost); on a plain-HTTP LAN (--listen) it's unavailable, so
    // fall back to a hidden-textarea + execCommand for those clients.
    async copyMeta() {
      const text = this.selectedMeta;
      if (!text) return;
      try {
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(text);
        } else {
          const ta = document.createElement('textarea');
          ta.value = text;
          ta.setAttribute('readonly', '');
          ta.style.position = 'fixed';
          ta.style.opacity = '0';
          document.body.appendChild(ta);
          ta.select();
          document.execCommand('copy');
          document.body.removeChild(ta);
        }
        this.flash('Parameters copied');
      } catch (e) {
        this.flash('Could not copy: ' + e);
      }
    },

    // ── extensions ──────────────────────────────────────────────
    // Backend list + install/toggle/reload/uninstall. The DiffucoreExt global
    // (set up at the top of this file, before Alpine inits) is how extension
    // scripts register tabs and settings panels into extTabs/extSettingsSpecs.

    async refreshExtensions() {
      try {
        const r = await fetchJSON('/api/extensions');
        this.extensions = r.extensions || [];
      } catch (e) { /* server may be mid-startup; silently retry on next open */ }
    },

    // x-effect on the Extensions settings section: only fetch when it's the
    // active tab, so an idle settings modal doesn't poll.
    refreshExtensionsIfOpen(tab) {
      if (tab === 'extensions') this.refreshExtensions();
    },

    isExtTab(tab) {
      return this.extTabs.some(t => t.id === tab);
    },

    switchExtTab(t) {
      this.tab = t.id;
    },

    // x-effect: when `tab` changes, mount/unmount the extension tab panel.
    // Extensions own their DOM; we just hand them the container element.
    mountExtTab(tab, el) {
      if (!el) return;
      const spec = this.extTabs.find(t => t.id === tab);
      if (spec && this._mountedExtTab !== spec) {
        this._unmountExtTab();
        try { spec.mount(el); this._mountedExtTab = spec; }
        catch (e) { el.textContent = 'Extension UI error: ' + e; }
      } else if (!spec && this._mountedExtTab) {
        this._unmountExtTab();
      }
    },

    _unmountExtTab() {
      if (this._mountedExtTab && this._mountedExtTab.unmount) {
        const el = this.$refs.extTabMount;
        try { this._mountedExtTab.unmount(el); } catch (e) { /* best-effort */ }
      }
      this._mountedExtTab = null;
      const el = this.$refs.extTabMount;
      if (el) el.innerHTML = '';
    },

    mountExtSettings(tab, el) {
      if (!el) return;
      if (tab !== 'extensions') return;
      // Extensions registered via DiffucoreExt.registerSettingsPanel each get a
      // child container; we mount all of them once.
      if (this._mountedExtSettings) return;
      const specs = (window.DiffucoreExt && window.DiffucoreExt.settingsPanels) || [];
      for (const spec of specs) {
        const child = document.createElement('div');
        child.className = 'ext-settings-panel';
        el.appendChild(child);
        try { spec.mount(child); } catch (e) {
          child.textContent = 'Extension settings UI error: ' + e;
        }
      }
      this._mountedExtSettings = true;
    },

    async installExtension() {
      const url = this.extInstallUrl.trim();
      if (!url) return;
      this.extBusy = true;
      try {
        // Installs run on the shared job worker (not the request threadpool), so
        // this returns a job id; the terminal SSE event resolves the promise.
        const r = await fetchJSON('/api/extensions/install', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url, install_pip_deps: this.extInstallPip }),
        });
        if (!r.job) throw new Error('install did not return a job id');
        const ev = await new Promise((resolve) => { this._jobWaiters[r.job] = resolve; });
        if (ev.type === 'error') throw new Error(ev.message || 'install failed');
        if (ev.type === 'cancelled') { this.flash('Install cancelled'); return; }
        const name = (ev.extension && (ev.extension.name || ev.extension.title)) || url;
        this.flash('Installed ' + name);
        this.extInstallUrl = '';
        await this.refreshExtensions();
        // Extension JS is injected server-side into the page on render; a
        // reload picks up the newly-installed extension's UI script.
        if (ev.extension && ev.extension.has_ui) this.flash('Reload the page to load the extension UI');
        // Surface a skipped-deps / load-error note (opt-in pip leaves a note on
        // the extension record when requirements.txt was skipped).
        if (ev.extension && ev.extension.load_error) {
          this.flash('Note: ' + ev.extension.load_error);
        }
      } catch (e) {
        this.flash('Install failed: ' + e.message);
      } finally {
        this.extBusy = false;
      }
    },

    async updateExtension(name) {
      this.extBusy = true;
      try {
        // Like install, update runs on the shared job worker and returns a job
        // id; the terminal SSE event resolves with the updated record.
        const r = await fetchJSON('/api/extensions/update', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, install_pip_deps: this.extInstallPip }),
        });
        if (!r.job) throw new Error('update did not return a job id');
        const ev = await new Promise((resolve) => { this._jobWaiters[r.job] = resolve; });
        if (ev.type === 'error') throw new Error(ev.message || 'update failed');
        if (ev.type === 'cancelled') { this.flash('Update cancelled'); return; }
        const ext = ev.extension || {};
        this.flash('Updated ' + name + (ext.version ? ' to v' + ext.version : ''));
        await this.refreshExtensions();
        if (ext.has_ui) this.flash('Reload the page to load the updated extension UI');
        if (ext.load_error) this.flash('Note: ' + ext.load_error);
      } catch (e) {
        this.flash('Update failed: ' + e.message);
      } finally {
        this.extBusy = false;
      }
    },

    async toggleExtension(name, enabled) {
      this.extBusy = true;
      try {
        const r = await fetch('/api/extensions/toggle', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, enabled }),
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || 'toggle failed');
        await this.refreshExtensions();
        if (enabled && data.extension.has_ui)
          this.flash('Reload the page to load the extension UI');
      } catch (e) {
        this.flash('Toggle failed: ' + e.message);
        await this.refreshExtensions();
      } finally {
        this.extBusy = false;
      }
    },

    async reloadExtension(name) {
      this.extBusy = true;
      try {
        const r = await fetch('/api/extensions/reload?name=' + encodeURIComponent(name), {
          method: 'POST',
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || 'reload failed');
        await this.refreshExtensions();
        this.flash('Reloaded ' + name + (data.extension.load_error
          ? ' (error: ' + data.extension.load_error + ')' : ''));
      } catch (e) {
        this.flash('Reload failed: ' + e.message);
      } finally {
        this.extBusy = false;
      }
    },

    async uninstallExtension(name) {
      if (!confirm('Uninstall extension "' + name + '"? This deletes its folder.')) return;
      this.extBusy = true;
      try {
        const r = await fetch('/api/extensions/uninstall', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name }),
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || 'uninstall failed');
        this.flash('Uninstalled ' + name);
        await this.refreshExtensions();
      } catch (e) {
        this.flash('Uninstall failed: ' + e.message);
      } finally {
        this.extBusy = false;
      }
    },
  }));
});

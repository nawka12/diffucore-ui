// Diffucore UI: Alpine state + streaming. No build step.

// ── extension bridge ──────────────────────────────────────────────
// Set up before Alpine inits, so extension scripts (injected between this file
// and alpine.min.js) can register tabs and settings panels.
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

// Kept outside the Alpine state so DOM nodes never pass through its reactive proxy.
const _modalState = { active: null, prevFocus: null };

// ── fetch helper ──────────────────────────────────────────────────
// fetch + JSON parse that fails loudly: a non-2xx throws an Error carrying the
// status and FastAPI's `detail` (or a body snippet) instead of a cryptic
// .json() parse error.
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
    perf: { compile: false, cudaGraphs: false, channelsLast: true, tf32: false, fp16Acc: false, vaeFp16: false, fa2Attn: false, offload: 'full' },
    fa2Available: false,
    recommendedOffload: 'full',   // VRAM-based default from the backend
    status: 'No model loaded',
    modelLoaded: false,
    loadingModel: false,
    // The user edited the model rack but hasn't loaded yet; SSE status from
    // another device must not clobber that selection.
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
      teacacheOn: false, teacache: 0.15, teacacheCalibrated: true, teacacheForecast: 'hermite',
      teacacheRule: 'drift',
      deepcacheOn: false, deepcache: 2,
    },
    // >1 submits N jobs: seed+i with a pinned seed, a fresh random with -1.
    batchCount: 1,

    // ── <lora:…> autocomplete in the prompt ─────────────────────
    loraAC: { open: false, items: [], index: 0, start: -1, key: 'prompt', el: null, set: null, wrap: null },

    inputImage: null,
    maskImage: null,
    dragKey: null,
    maskBrush: 40,
    maskTool: 'brush',   // brush | eraser | rect
    maskMax: false,      // fullscreen the input & mask editor
    maskZoom: 1,         // display zoom while maximized (1 = fit)

    // ── detailer (ADetailer-style passes after generate) ────────
    // `models` is a stack of {model, prompt} run in sequence; the rest is shared.
    detail: {
      enabled: false, neg: '',
      models: [{ model: '', prompt: '' }],
      confidence: 0.3, strength: 0.4,
      dilation: 4, padding: 32, blur: 4, maxDet: 0,
      teacache: false,
    },

    // ── upscaler (tiled, after generate) ─────────────────────────
    // Refine-pass TeaCache, independent of the main slider (0 = off).
    upscale: {
      enabled: false, scale: 2.0, denoise: 0.35,
      tile: 1024, overlap: 128, prompt: '', teacache: 0.0, base: '',
    },

    // ── standalone upscale popover (from result / lightbox) ──────
    upscalePopover: { open: false, busy: false },
    upscaleForm: { scale: 2.0, denoise: 0.35, tile: 1024, overlap: 128, prompt: '', teacache: 0.0, base: '' },

    // ── standalone detailer popover (from result / lightbox) ─────
    detailPopover: { open: false, busy: false },
    detailForm: {
      models: [{ model: '', prompt: '' }],
      prompt: '', neg: '',
      confidence: 0.3, strength: 0.4,
      dilation: 4, padding: 32, blur: 4, maxDet: 0,
      teacache: 0.0,
    },

    // ── generation output ───────────────────────────────────────
    busy: false,
    cancelling: false,
    progress: { step: 0, total: 0 },
    resultUrl: null,
    // The fresh result is blurred when its rating reaches the Settings tier
    // (prompt verdict first, AI tagger's when it lands) until clicked.
    genBlur: true,
    genReveal: false,          // user clicked the blurred result to reveal it
    resultPath: null,          // outputs/… path of the currently shown result
    _promptNsfw: {},           // path -> {nsfw, rating} from the done event
    _visionNsfw: {},           // path -> {nsfw, rating} from the AI tagger (wins)
    batchResults: [],       // thumbnails for a >1 batch: [{url, seed}]; the last is shown
    previewUrl: null,
    preview: true,
    info: '',
    lastSeed: -1,
    _titleBase: '',       // original tab title, captured on init
    _titleDone: false,    // a job finished while this tab was hidden

    // ── shared queue (broadcast over /api/events to every device) ──
    queue: [],            // [{id, kind, label, status}], running first
    runningJob: null,     // id of the job currently on the GPU (any device)
    runProg: { step: 0, total: 0 },  // progress of the running job (for the queue panel)
    myJobId: null,        // the single job THIS device submitted and is watching
    _myBatchIds: [],      // ids of in-flight batch jobs (empty for single-job flows)
    _jobWaiters: {},      // id -> resolve fn, fulfilled by the terminal SSE event

    // ── SSE connection state ─────────────────────────────────────
    // 'connected' | 'reconnecting' | 'down'; drives the "not live" banner.
    connState: 'connected',
    connDownSince: null,        // epoch ms when the connection went 'down'
    CONN_DOWN_TIMEOUT: 8000,    // reconnect grace before the outage banner
    _connTimer: null,

    // ── OSS calibration ─────────────────────────────────────────
    calibrating: false,
    ossCalibrated: null,          // null = unknown, true/false = checked
    ossInfo: '',
    _ossToken: 0,                 // bumped per status check; stale replies dropped

    // ── settings panel (global, non-per-image knobs) ────────────
    settingsOpen: false,
    settingsTab: 'teacache',
    settings: { curvature: 0.25, eta_max: 1.0, gate_reduce: 'all', beta_alpha: 0.6, beta_beta: 0.6, lq_threshold: 0.025, cfg_interval_start: 0.0, cfg_interval_end: 1.0, teacache_uncond_scale: 1.0, vae_tiling: 'auto', metadata_format: 'a1111', gen_defaults: null, nsfw_blur: true, blur_min_rating: 'R' },
    teacacheStatus: { loaded: false, calibratable: false, family: null, coefficients: null },
    calibratingTea: false,
    taggerStatus: { available: false, loaded: false, rated: 0, total: 0 },
    // THIS device's gallery-scan job; its progress keeps taggerStatus live.
    scanJob: null,

    // ── extensions ──────────────────────────────────────────────
    // `extTabs` come from window.DiffucoreExt.registerTab calls.
    extensions: [],
    extTabs: [],
    extInstallUrl: '',
    extInstallPip: false,  // opt-in: pip install -r is RCE on untrusted sources
    extBusy: false,
    _mountedExtTab: null,
    _mountedExtSettings: null,

    // ── gallery ─────────────────────────────────────────────────
    gallery: [],
    galleryGroups: [],
    galleryLimit: 60,   // chunked rendering: thumbs live in the DOM
    galleryQuery: '',   // substring filter applied via /api/gallery?q=
    gallerySearching: false,
    _galleryToken: 0,   // bumped per listing fetch; stale responses dropped
    selected: null,
    selectedMeta: '',
    selectedFields: {},
    lightbox: { open: false, index: 0, info: false },
    // Reset on every navigation so a revealed image never carries over.
    lbReveal: false,
    deleteConfirm: false,   // two-click confirm in the lightbox Delete button

    // ── metadata reader ─────────────────────────────────────────
    metaPreview: null,
    metaText: '',
    metaFields: null,

    // ── x/y/z sweep (txt2img only; reuses the generate form) ──
    xyzSweep: false,
    // Axes start empty: preset values would turn bogus after a type switch.
    axes: {
      x: { type: 'Sampler', text: '', list: [] },
      y: { type: 'Steps', text: '', list: [] },
      z: { type: 'None', text: '', list: [] },
    },
    xyzGrids: [],
    xyzInfo: '',
    // One prompt, so one verdict covers every grid; `xyzPath` (the last grid)
    // lets a later "rated" event refine it.
    xyzNsfw: null,             // {nsfw, rating} from the done event
    xyzPath: null,
    xyzReveal: false,

    toast: '',
    toastKind: 'info',     // info | success | error

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
    // Shift is flow-only, and honoured by Anima everywhere but flow_dyn and by
    // FLUX only on plain flow.
    get isFlowModel() {
      return this.modelType === 'Anima' || this.modelType === 'FLUX';
    },
    get shiftHonored() {
      if (this.modelType === 'Anima') return this.form.scheduler !== 'flow_dyn';
      if (this.modelType === 'FLUX') return this.form.scheduler === 'flow';
      return false;
    },
    get offloadOptions() {
      return ['stream', 'full', 'encoders', 'none'];
    },
    get sweeping() {
      return this.mode === 't2i' && this.xyzSweep;
    },

    // Sampler / Scheduler / Checkpoint axes are multi-selects; numeric axes
    // keep a free-text comma list.
    axisIsList(axis) { return axis.type === 'Sampler' || axis.type === 'Scheduler' || axis.type === 'Checkpoint'; },
    axisOptions(axis) {
      if (axis.type === 'Scheduler') return this.schedulers;
      // Anima's "checkpoint" is the DiT (VAE + TE stay fixed).
      if (axis.type === 'Checkpoint') return this.modelType === 'Anima' ? this.dits : this.checkpoints;
      return this.samplers;
    },
    axisValues(axis) {
      if (axis.type === 'None') return '';
      return this.axisIsList(axis) ? axis.list.join(', ') : axis.text;
    },
    // A type switch makes old values meaningless (stale sampler names would
    // load as bogus checkpoints), so clear both stores.
    clearAxisValues(axis) { axis.list = []; axis.text = ''; },
    get progressPct() {
      const t = this.progress.total;
      return t > 0 ? Math.round((this.progress.step / t) * 100) : 0;
    },
    // The AI verdict once it lands, else the done event's prompt verdict.
    get resultMeta() {
      const p = this.resultPath;
      if (!p) return null;
      return this._visionNsfw[p] || this._promptNsfw[p] || null;
    },
    // Blurred at or above the Settings tier. Decided from the rating, not the
    // server's fixed R-and-up `nsfw` flag, so changing the tier needs no re-rate.
    blurs(rating) {
      const order = ['PG', 'PG13', 'R', 'X', 'XXX'];
      const i = order.indexOf(rating);
      return i >= 0 && i >= order.indexOf(this.settings.blur_min_rating || 'R');
    },
    get resultNsfw() { return !!(this.resultMeta && this.blurs(this.resultMeta.rating)); },
    get resultRating() { return this.resultMeta ? this.resultMeta.rating : ''; },
    get resultBlurred() {
      return !!(this.blurOn && this.resultNsfw && !this.genReveal);
    },
    // The Generate page's own toggle, independent of the gallery setting. It
    // rides along as `blur_check` so the server rates what this page blurs.
    get blurOn() { return !!this.genBlur; },
    // Blurred like a gallery thumbnail regardless of the canvas reveal.
    batchBlurred(b) {
      if (!this.blurOn) return false;
      const m = (b.path && this._visionNsfw[b.path]) || b.nsfw;
      return !!(m && this.blurs(m.rating));
    },
    get xyzMeta() {
      return (this.xyzPath && this._visionNsfw[this.xyzPath]) || this.xyzNsfw || null;
    },
    get xyzNsfwFlag() { return !!(this.xyzMeta && this.blurs(this.xyzMeta.rating)); },
    get xyzBlurred() { return !!(this.blurOn && this.xyzNsfwFlag && !this.xyzReveal); },
    get progressLabel() {
      const t = this.progress.total;
      if (t <= 0) return 'Starting…';
      const steps = `${this.progress.step} / ${t}  (${this.progressPct}%)`;
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
      this.extTabs = window.DiffucoreExt ? [...window.DiffucoreExt.tabs] : [];
      await this.refreshModels();
      await this.loadSettings();
      this.applyGenDefaults();
      this._initGenToggles();
      this._initTitle();
      this.connectEvents();
      this.refreshExtensions();
    },

    // Per-device, so they live in localStorage rather than the shared
    // settings.json. Storage can throw (private mode, blocked site data).
    _initGenToggles() {
      for (const k of ['preview', 'genBlur']) {
        const key = 'diffucore.' + k;
        try {
          const v = localStorage.getItem(key);
          if (v !== null) this[k] = v === '1';
        } catch (e) { /* keep default */ }
        this.$watch(k, (v) => {
          try { localStorage.setItem(key, v ? '1' : '0'); } catch (e) { /* not persisted */ }
        });
      }
    },

    // ── browser-tab title reflects this device's job state ──────
    // Spinner + % while running; a "done" badge if it finished while hidden.
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
      // Leaving the Generate tab re-blurs a revealed result, like closing the
      // lightbox does.
      this.$watch('tab', (t) => {
        if (t !== 'generate') { this.genReveal = false; this.xyzReveal = false; }
      });
    },

    // ── shared events (one SSE stream per device) ───────────────
    // A stalled stream would freeze the queue panel silently, so onerror goes
    // 'reconnecting' and, without a reconnect within CONN_DOWN_TIMEOUT, 'down'
    // (banner). The reconnect snapshot re-syncs everything.
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
          // A terminal event is lost if the SSE drops while the job ends. The
          // reconnect snapshot lists every live job, so resolve waiters whose
          // job is gone instead of leaving busy stuck.
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
          // Feeds the queue panel; the main bar only tracks THIS device's jobs.
          this.runProg = { step: ev.step, total: ev.total };
          if (ev.job === this.scanJob) {
            this.taggerStatus.rated = ev.step;
            this.taggerStatus.total = ev.total;
          }
          if (ev.job === this.myJobId || this._myBatchIds.includes(ev.job))
            this.progress = { step: ev.step, total: ev.total, cell: ev.cell, cells: ev.cells };
          break;
        case 'preview':
          if (ev.job === this.myJobId || this._myBatchIds.includes(ev.job))
            this.previewUrl = ev.image;
          break;
        case 'rated':
          // The AI tagger's verdict for a freshly saved output.
          if (ev.path) this._visionNsfw[ev.path] = { nsfw: !!ev.nsfw, rating: ev.rating };
          break;
        case 'done':
        case 'error':
        case 'cancelled': {
          const w = this._jobWaiters[ev.job];
          if (w) { delete this._jobWaiters[ev.job]; w(ev); }
          const bi = this._myBatchIds.indexOf(ev.job);
          if (bi !== -1) this._myBatchIds.splice(bi, 1);
          // Only when THIS job ends: a job queued behind it already moved
          // myJobId, and nulling it would stop its progress routing.
          if (ev.job === this.myJobId) this.myJobId = null;
          break;
        }
      }
    },

    // Submit a job; resolves on its terminal SSE event.
    async submitJob(url, payload) {
      const r = await fetchJSON(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      this.myJobId = r.job;
      return new Promise((resolve) => { this._jobWaiters[r.job] = resolve; });
    },

    // Submit N copies of a job. A pinned seed is offset by i per copy; -1 is
    // forwarded so each gets a fresh random. Progress routes through
    // _myBatchIds, not myJobId.
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
        // Cancel the jobs already enqueued, then surface the error.
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

    applyState(s) {
      this.status = s.status;
      this.modelLoaded = !!s.loaded;
      if (s.last_seed !== undefined) this.lastSeed = s.last_seed;
      // Don't overwrite a selection the user is editing locally.
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
      this.perf.vaeFp16 = !!f.vae_fp16;
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
      // The first fetch seeds every selector; a later Refresh only replaces
      // selections whose file vanished.
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
        // '' is a valid CLIP pick ("(none)", FLUX.2).
        if (this.clip !== '' && !this.teChoices.includes(this.clip)) this.clip = '';
        for (const dm of this.detail.models) dm.model = keep(dm.model, this.detailerChoices);
      }
      this.syncSampler();
      this.syncScheduler();
      // Hydrate from the server's load state last.
      this.applyState(m);
    },

    setModelType(type) {
      this.modelType = type;
      this.loadFormDirty = true;
      this.syncSampler();
      this.syncScheduler();
      // FLUX always streams its DiT; the others take the backend's VRAM-based
      // default.
      this.perf.offload = (type === 'FLUX') ? 'stream' : this.recommendedOffload;
      // channels_last only helps conv backbones (SD/SDXL UNet + VAE).
      this.perf.channelsLast = (type === 'SD/SDXL');
      if (type === 'Anima' && !this.animaApplied) {
        this.animaApplied = true;
        this.form.sampler = 'er_sde';
        this.form.steps = 30;
        this.form.cfg = 4.0;
        // shift=3 makes a given strength far noisier than SDXL's EDM; 0.4
        // over-regenerates faces.
        this.detail.strength = 0.25;
      }
      if (type === 'FLUX' && !this.fluxApplied) {
        // FLUX-dev defaults, applied once (cfg = distilled guidance)
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
          vae_fp16: this.perf.vaeFp16,
          // fa2 only applies to the DiT families.
          attention: (this.perf.fa2Attn && this.modelType !== 'SD/SDXL') ? 'fa2_turing' : 'sdpa',
        };
        // Queued like any job; the server broadcasts the new state everywhere.
        const ev = await this.submitJob('/api/load', body);
        if (ev.type === 'done') {
          this.status = ev.status;
          this.modelLoaded = !!ev.loaded;
          // Our selection is now the loaded form, so broadcasts may sync again.
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
      // Drops bypass the input's accept filter.
      if (f.type && !f.type.startsWith('image/')) { this.flash('Not an image file'); return; }
      const r = new FileReader();
      r.onload = () => {
        this[key] = r.result;
        if (key === 'inputImage') this.syncOutputSize(r.result);
      };
      r.readAsDataURL(f);
    },

    // Match the output size to the source, rounded to ×8, so it isn't stretched.
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
    // The offscreen `_mask` buffer is the source of truth (opaque white where
    // masked); the visible canvas is always base + orange-tinted mask, so every
    // tool just mutates `_mask` and redraws. Export flattens it onto black.
    // Buffers hang off the canvas element so Alpine never proxies them.
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
    // Maximized: fit-to-viewport base × zoom; the stage scrolls to pan.
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
      // Capture the pointer so a stroke can run off the canvas edge.
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
        // Hover preview of the brush footprint.
        this.redrawMask(c);
        this.strokeBrushRing(c, p);
      }
    },
    // Clear the hover ring on leave (strokes and drags repaint themselves).
    maskLeave(e) {
      const c = e.currentTarget;
      if (c._mask && !c._painting && !c._dragging) this.redrawMask(c);
    },
    // Brush ring in canvas coordinates; a dark halo under a light ring keeps it
    // visible over the image and the mask tint.
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
      this.redrawMask(c);
    },
    fillRectMask(c, a, b) {
      const w = Math.abs(b.x - a.x), h = Math.abs(b.y - a.y);
      if (!w || !h) { this.redrawMask(c); return; }
      this.pushUndo(c);
      const mctx = c._mask.getContext('2d');
      mctx.fillStyle = '#fff';
      mctx.fillRect(Math.min(a.x, b.x), Math.min(a.y, b.y), w, h);
      this.redrawMask(c);
    },
    // Snapshot the mask before a mutating op. The first snapshot of a chain is
    // the empty mask, so painted-ness is read from the buffer, never the stack.
    pushUndo(c) {
      const mctx = c._mask.getContext('2d');
      c._undo.push(mctx.getImageData(0, 0, c._mask.width, c._mask.height));
      if (c._undo.length > 30) c._undo.shift();
    },
    undoMask() {
      const c = this.$refs.maskCanvas;
      if (!c || !c._undo || !c._undo.length) return;
      c._mask.getContext('2d').putImageData(c._undo.pop(), 0, 0);
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
      this.redrawMask(c);
    },
    clearMask() {
      const c = this.$refs.maskCanvas;
      if (!c || !c._base) return;
      this.pushUndo(c);
      c._mask.getContext('2d').clearRect(0, 0, c._mask.width, c._mask.height);
      this.redrawMask(c);
    },
    // Flatten onto black: the white-on-black PNG the engine expects.
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

    // One alpha scan of `_mask` is the source of truth for coverage (erase to
    // empty, undo, invert all change it); a few ms on a 2K canvas.
    maskHasCoverage(c) {
      const m = c && c._mask;
      if (!m) return false;
      const d = m.getContext('2d').getImageData(0, 0, m.width, m.height).data;
      for (let i = 3; i < d.length; i += 4) if (d[i] > 128) return true;
      return false;
    },

    // ── auto-growing textareas ──────────────────────────────────
    autogrow(el) {
      if (!el || el.offsetParent === null) return;   // hidden
      el.style.height = 'auto';
      el.style.height = el.scrollHeight + 'px';
    },
    resizeTextareas() {
      this.$nextTick(() => {
        this.$root.querySelectorAll('textarea.autosize').forEach((el) => this.autogrow(el));
      });
    },

    // ── <lora:…> autocomplete ───────────────────────────────────
    // Typing `<` lists LoRAs, filtered by the fragment after the last unclosed
    // `<` (past an optional `lora:`); picking one inserts `<lora:name:1.0>`. The
    // dropdown hides the .safetensors suffix.
    loraLabel(name) {
      return name.replace(/\.safetensors$/i, '');
    },
    // Used by the prompt (default) and the X/Y/Z Prompt S/R fields, which pass
    // {key, set} to write back to the axis text.
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

    // Ctrl/Cmd+Enter generates, unless the LoRA autocomplete is open; other
    // keys go to loraKeydown.
    promptKeydown(e) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (this.loraAC.open) return;
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
      // A load in flight is fine: the job queues behind it.
      if (!this.modelLoaded && !this.loadingModel) { this.flash('Load a model first'); return; }
      if (this.mode === 'i2i' && !this.inputImage) { this.flash('Provide an input image'); return; }
      if (this.mode === 'inpaint') {
        if (!this.inputImage) { this.flash('Provide an input image'); return; }
        if (!this.maskHasCoverage(this.$refs.maskCanvas)) {
          this.flash('Paint a mask over the image');
          return;
        }
        this.maskImage = this.exportMask();
      }
      this.busy = true;
      this.cancelling = false;
      this.progress = { step: 0, total: 0 };
      this.previewUrl = null;
      this.info = '';
      this.batchResults = [];
      this.resultPath = null;
      this.genReveal = false;
      this._promptNsfw = {};
      this._visionNsfw = {};
      try {
        // Calibrate OSS for this steps/size/shift on first use, then generate,
        // under one click. The status is re-checked fresh.
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
          teacache_forecast: this.form.teacacheForecast,
          teacache_rule: this.form.teacacheRule,
          deepcache: this.form.deepcacheOn ? this.form.deepcache : 1,
          input_image: this.mode !== 't2i' ? this.inputImage : null,
          mask_image: this.mode === 'inpaint' ? this.maskImage : null,
          preview: this.preview,
          blur_check: this.blurOn,
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
        // Batch: submit N copies and await all; a single job keeps submitJob.
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
            // Keep every finished image for the strip; the canvas shows the
            // last (FIFO on the single worker).
            this.batchResults = results
              .filter(ev => ev.type === 'done')
              .map(ev => ({
                url: ev.image_url + '?t=' + Date.now(),
                seed: ev.seed,
                path: ev.path,
                nsfw: { nsfw: !!ev.nsfw_prompt, rating: ev.prompt_rating },
              }));
            for (const ev of results) {
              if (ev.type === 'done' && ev.path) {
                this._promptNsfw[ev.path] = { nsfw: !!ev.nsfw_prompt, rating: ev.prompt_rating };
              }
            }
            const shown = this.batchResults[this.batchResults.length - 1];
            this.resultPath = shown.path;
            this.genReveal = false;
            this.resultUrl = shown.url;
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
          const ev = await this.submitJob('/api/generate', payload);
          if (ev.type === 'done') {
            this.previewUrl = null;
            this.resultUrl = ev.image_url + '?t=' + Date.now();
            this.resultPath = ev.path || null;
            if (ev.path) this._promptNsfw[ev.path] = { nsfw: !!ev.nsfw_prompt, rating: ev.prompt_rating };
            this.genReveal = false;
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
        // In case a terminal event was missed.
        this._myBatchIds = [];
      }
    },

    // Cancel a job by id (default: this device's). A running job stops at its
    // next step, a queued one is dropped; with no arg every batch member goes.
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
      // Queue behind an in-flight load (see generate()).
      if (!this.modelLoaded && !this.loadingModel) { this.flash('Load a model first'); return; }
      // A Prompt S/R search term missing from both prompts makes every cell
      // identical. Same case-sensitive match as the backend's replace.
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
      this.xyzNsfw = null;
      this.xyzPath = null;
      this.xyzReveal = false;
      this.batchResults = [];
      const payload = {
        prompt: this.form.prompt, neg: this.form.neg,
        width: this.form.width, height: this.form.height,
        steps: this.form.steps, cfg: this.form.cfg,
        sampler: this.form.sampler, scheduler: this.form.scheduler,
        seed: this.form.seed, shift: this.form.shift,
        teacache: this.form.teacacheOn ? this.form.teacache : 0,
        teacache_calibrated: this.form.teacacheCalibrated,
        teacache_forecast: this.form.teacacheForecast,
        teacache_rule: this.form.teacacheRule,
        x_type: this.axes.x.type, x_vals: this.axisValues(this.axes.x),
        y_type: this.axes.y.type, y_vals: this.axisValues(this.axes.y),
        z_type: this.axes.z.type, z_vals: this.axisValues(this.axes.z),
        preview: this.preview,
        blur_check: this.blurOn,
      };
      try {
        const ev = await this.submitJob('/api/xyz', payload);
        if (ev.type === 'done') {
          this.xyzGrids = ev.grids;
          this.xyzPath = ev.path || null;
          this.xyzNsfw = { nsfw: !!ev.nsfw_prompt, rating: ev.prompt_rating };
          this.xyzReveal = false;
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
    // Reads form fields synchronously so Alpine's x-effect re-checks on change.
    async checkOssStatus() {
      const { scheduler, steps, width, height, shift } = this.form;
      // A token drops stale replies while a slider fires one check per tick. It
      // is bumped before the early return too, so an in-flight check can't
      // revive a verdict after switching away from oss.
      const token = ++this._ossToken;
      if (this.modelType !== 'Anima' || scheduler !== 'oss' || !this.modelLoaded) {
        this.ossCalibrated = null;
        return;
      }
      const q = new URLSearchParams({ steps, width, height, shift });
      try {
        const calibrated = (await fetchJSON('/api/oss_status?' + q)).calibrated;
        if (token !== this._ossToken) return;
        this.ossCalibrated = calibrated;
      } catch (e) {
        if (token === this._ossToken) this.ossCalibrated = null;
      }
    },

    needsOss() {
      return this.modelType === 'Anima' && this.mode === 't2i' && this.form.scheduler === 'oss';
    },

    // Queue a calibration job; returns true on success. The caller owns `busy`,
    // so calibrate→generate runs under one spinner.
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
    // Sampler-section auto-save. Skip while a field is mid-edit (''/null under
    // .number) or the CFG interval pair is out of order, which the backend
    // rejects, so a two-field edit commits once instead of toasting per blur.
    saveSamplerSettings() {
      const s = this.settings;
      const nums = [s.curvature, s.eta_max, s.beta_alpha, s.beta_beta,
                    s.lq_threshold, s.cfg_interval_start, s.cfg_interval_end];
      if (nums.some((n) => n == null || n === '' || Number.isNaN(n))) return;
      if (s.cfg_interval_start >= s.cfg_interval_end) return;
      this.saveSettings();
    },
    // Same guard for the TeaCache tab's lone number field.
    saveTeacacheSettings() {
      const v = this.settings.teacache_uncond_scale;
      if (v == null || v === '' || Number.isNaN(v)) return;
      this.saveSettings();
    },

    async refreshTeacacheStatus() {
      try { this.teacacheStatus = await fetchJSON('/api/teacache_status'); }
      catch (e) { this.teacacheStatus = { loaded: false, calibratable: false, family: null, coefficients: null }; }
    },

    async refreshTaggerStatus() {
      try { this.taggerStatus = await fetchJSON('/api/tagger_status'); }
      catch (e) { this.taggerStatus = { available: false, loaded: false, rated: 0, total: 0 }; }
    },

    // Rate every gallery image lacking an AI verdict, as one queued job.
    async scanGallery() {
      if (this.scanJob) { this.flash('Already rating the gallery'); return; }
      if (this.busy) return;
      try {
        const r = await fetchJSON('/api/gallery_scan', { method: 'POST' });
        if (!r.job) { this.flash('All images already rated'); return; }
        this.scanJob = r.job;
        this.flash(`Rating ${r.total} images…`);
        const ev = await new Promise((resolve) => { this._jobWaiters[r.job] = resolve; });
        this.scanJob = null;
        if (ev.type === 'done') {
          this.flash(`Rated ${r.total} images`);
          this.refreshTaggerStatus();
          if (this.tab === 'gallery') this.searchGallery();
        } else if (ev.type === 'error') {
          this.flash('Rating failed: ' + ev.message);
        }
      } catch (e) {
        this.scanJob = null;
        this.flash('' + e);
      }
    },
    get scanPct() {
      const t = this.taggerStatus.total;
      return t > 0 ? Math.min(100, Math.round((this.taggerStatus.rated / t) * 100)) : 0;
    },

    // Seed the Generate form from saved defaults, then re-validate the sampler
    // and scheduler against the current model type.
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
      // Pin prompt/negative only when filled.
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
      this.refreshTaggerStatus();
    },

    // ── modal a11y ─────────────────────────────────────────────
    // Overlays get dialog semantics, a Tab trap, and an `inert` app shell.
    // Wired via x-effect on each overlay.
    modalA11y(el, open, label) {
      if (!el) return;
      if (open) {
        if (_modalState.active === label) return;
        _modalState.active = label;
        el.setAttribute('role', 'dialog');
        el.setAttribute('aria-modal', 'true');
        if (label) el.setAttribute('aria-label', label);
        _modalState.prevFocus = document.activeElement;
        const shell = this.$root.querySelector('.shell');
        if (shell) shell.setAttribute('inert', '');
        this.$nextTick(() => {
          const f = el.querySelector('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
          if (f) f.focus();
        });
      } else if (_modalState.active === label) {
        _modalState.active = null;
        const shell = this.$root.querySelector('.shell');
        if (shell) shell.removeAttribute('inert');
        const prev = _modalState.prevFocus;
        _modalState.prevFocus = null;
        // The previous element may be hidden by now (the gallery→upscale
        // handoff closes the lightbox), and focusing it would drop focus to
        // <body>; fall back to the header gear.
        if (prev && prev.isConnected && prev.offsetParent !== null) {
          prev.focus();
        } else {
          const gear = this.$root.querySelector('.topbar .gear');
          if (gear) gear.focus();
        }
      }
    },
    // Keep Tab/Shift+Tab cycling inside the open dialog.
    modalTab(e, el) {
      if (e.key !== 'Tab') return;
      const list = [...el.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')]
        .filter(f => !f.disabled && f.offsetParent !== null);
      if (!list.length) return;
      const first = list[0], last = list[list.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    },

    // Fit the TeaCache polynomial for the loaded Anima family, as a queued job.
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
      // Capture the source now. A gallery image carries its own metadata for the
      // refine pass; a fresh result is described by the live form.
      let fromGallery = false;
      if (this.lightbox.open && this.selected) {
        await this._metaLoad;                    // metadata may still be loading
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
      // The popover and progress bar live on the Generate tab.
      if (fromGallery) { this.closeLightbox(); this.tab = 'generate'; }
      this.upscalePopover.open = true;
      this.upscalePopover.busy = false;
    },
    closeUpscale() { this.upscalePopover.open = false; },
    async runUpscale() {
      const src = this._upscaleSrc;
      if (!src) { this.flash('No image to upscale'); return; }
      if (!this.modelLoaded) { this.flash('Load a model first'); return; }
      this.closeUpscale();   // reveal the progress bar + live preview
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
        // A gallery image's metadata over the form, or the form for a fresh result.
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
          teacache_forecast: this.form.teacacheForecast,
          teacache_rule: this.form.teacacheRule,
          preview: this.preview,
          blur_check: this.blurOn,
        };
        const ev = await this.submitJob('/api/upscale', payload);
        if (ev.type === 'done') {
          this.previewUrl = null;
          this.resultUrl = ev.image_url + '?t=' + Date.now();
          // Carry the upscale's own verdict, or the canvas would keep the
          // previous result's blur state.
          this.resultPath = ev.path || null;
          if (ev.path) this._promptNsfw[ev.path] = { nsfw: !!ev.nsfw_prompt, rating: ev.prompt_rating };
          this.genReveal = false;
          this.batchResults = [];
          this.info = ev.info;
          this.closeUpscale();
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

    // ── standalone detailer ──────────────────────────────────────
    // Like the upscale popover: refine an existing image without re-sampling.
    addDetailFormModel() {
      this.detailForm.models.push({ model: this.detailerChoices[0], prompt: '' });
    },
    removeDetailFormModel(i) {
      this.detailForm.models.splice(i, 1);
      if (!this.detailForm.models.length) this.addDetailFormModel();
    },
    async openDetail() {
      let fromGallery = false;
      if (this.lightbox.open && this.selected) {
        await this._metaLoad;                    // metadata may still be loading
        this._detailSrc = { url: this.selected.url, meta: this.selectedFields };
        fromGallery = true;
      } else if (this.resultUrl) {
        this._detailSrc = { url: this.resultUrl, meta: null };
      } else {
        this._detailSrc = null;
      }
      // Seed the popover from the Generate-tab detailer panel.
      const meta = this._detailSrc && this._detailSrc.meta;
      this.detailForm.models = this.detail.models.map(dm => ({ ...dm }));
      if (!this.detailForm.models.length) this.addDetailFormModel();
      for (const dm of this.detailForm.models) {
        if (!dm.model) dm.model = this.detailerChoices[0];
      }
      this.detailForm.prompt = (meta && meta.prompt) || this.form.prompt;
      this.detailForm.neg = this.detail.neg || (meta && meta.neg) || this.form.neg;
      this.detailForm.confidence = this.detail.confidence;
      this.detailForm.strength = this.detail.strength;
      this.detailForm.dilation = this.detail.dilation;
      this.detailForm.padding = this.detail.padding;
      this.detailForm.blur = this.detail.blur;
      this.detailForm.maxDet = this.detail.maxDet;
      if (fromGallery) { this.closeLightbox(); this.tab = 'generate'; }
      this.detailPopover.open = true;
      this.detailPopover.busy = false;
    },
    closeDetail() { this.detailPopover.open = false; },
    async runDetail() {
      const src = this._detailSrc;
      if (!src) { this.flash('No image to detail'); return; }
      if (!this.modelLoaded) { this.flash('Load a model first'); return; }
      const models = this.detailForm.models.filter(dm => dm.model && !dm.model.startsWith('('));
      if (!models.length) { this.flash('Pick at least one detection model'); return; }
      this.closeDetail();   // reveal the progress bar + live preview
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
        // A gallery image's metadata over the form, or the form for a fresh result.
        const refine = src.meta ? { ...this.form, ...src.meta } : this.form;
        const payload = {
          input_image: inputImage,
          models,
          prompt: this.detailForm.prompt,
          neg: this.detailForm.neg,
          confidence: this.detailForm.confidence,
          strength: this.detailForm.strength,
          dilation: this.detailForm.dilation,
          padding: this.detailForm.padding,
          blur: this.detailForm.blur,
          max_det: this.detailForm.maxDet,
          steps: refine.steps, cfg: refine.cfg,
          sampler: refine.sampler, scheduler: refine.scheduler,
          seed: refine.seed,
          teacache: this.detailForm.teacache,
          teacache_calibrated: this.form.teacacheCalibrated,
          teacache_forecast: this.form.teacacheForecast,
          teacache_rule: this.form.teacacheRule,
          preview: this.preview,
          blur_check: this.blurOn,
        };
        const ev = await this.submitJob('/api/detail', payload);
        if (ev.type === 'done') {
          this.previewUrl = null;
          this.resultUrl = ev.image_url + '?t=' + Date.now();
          // Carry this run's own verdict, not the previous result's blur state.
          this.resultPath = ev.path || null;
          if (ev.path) this._promptNsfw[ev.path] = { nsfw: !!ev.nsfw_prompt, rating: ev.prompt_rating };
          this.genReveal = false;
          this.batchResults = [];
          this.info = ev.info;
          this.closeLightbox();
          this.tab = 'generate';
          this.flash('Detailer done');
        } else if (ev.type === 'cancelled') {
          this.previewUrl = null;
          this.info = 'Detailer cancelled';
        } else if (ev.type === 'error') {
          this.info = 'Error: ' + ev.message;
          this.flash(ev.message);
        }
      } catch (e) {
        this.info = 'Error: ' + e;
      } finally {
        this.detailPopover.busy = false;
        this.busy = false;
        this.cancelling = false;
        this.previewUrl = null;
      }
    },

    // ── gallery ─────────────────────────────────────────────────
    // Both listing fetches share a monotonic token, so an out-of-order response
    // can't overwrite a newer one.
    async openGallery() {
      this.tab = 'gallery';
      this.selected = null;
      this.selectedMeta = '';
      this.galleryLimit = 60;
      this.galleryQuery = '';
      this.gallerySearching = false;
      const token = ++this._galleryToken;
      try {
        const images = (await fetchJSON('/api/gallery')).images;
        if (token !== this._galleryToken) return;
        this.gallery = images;
        this.buildGalleryGroups();
      } catch (e) {
        /* keep the previous list on a transient error */
      }
    },

    // Debounced search against the backend's cached metadata index.
    async searchGallery() {
      const q = (this.galleryQuery || '').trim();
      this.gallerySearching = true;
      const token = ++this._galleryToken;
      try {
        const url = q ? `/api/gallery?q=${encodeURIComponent(q)}` : '/api/gallery';
        const images = (await fetchJSON(url)).images;
        if (token !== this._galleryToken) return;
        this.gallery = images;
        this.selected = null;
        this.selectedMeta = '';
        this.galleryLimit = 60;
        this.buildGalleryGroups();
      } catch (e) {
        /* keep the previous list on a transient error */
      } finally {
        if (token === this._galleryToken) this.gallerySearching = false;
      }
    },

    // Group the first `galleryLimit` images into day sections, keeping each
    // image's flat index so the lightbox pages across the whole list.
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

    // Render the next chunk as the bottom sentinel nears the viewport.
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
      // Drop the previous metadata so nothing acts on stale data meanwhile.
      this.selectedMeta = '';
      this.selectedFields = null;
      // Token: when paging fast, an earlier fetch may resolve last. Consumers
      // await _metaLoad.
      const token = img.url;
      this._metaToken = token;
      this._metaLoad = (async () => {
        try {
          const r = await fetchJSON('/api/metadata?path=' + encodeURIComponent(img.path));
          if (this._metaToken !== token) return;
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
      this.lbReveal = false;
      this.selectImage(this.gallery[i]);
    },
    closeLightbox() {
      this.lightbox.open = false;
      this.deleteConfirm = false;
      this.lbReveal = false;
    },
    lbPrev() { this.lbGo(-1); },
    lbNext() { this.lbGo(1); },
    lbGo(d) {
      const n = this.gallery.length;
      if (!n) return;
      this.deleteConfirm = false;
      this.lightbox.index = (this.lightbox.index + d + n) % n;
      this.lbReveal = false;
      this.selectImage(this.gallery[this.lightbox.index]);
    },
    // Toggle the blur of a blurred image; a no-op otherwise, so a plain click
    // still selects.
    reveal() {
      if (!this.lbReveal && !this.lbBlurred) return;
      this.lbReveal = !this.lbReveal;
    },
    get lbBlurred() {
      return !!(this.selected && this.blurs(this.selected.rating) && this.settings.nsfw_blur);
    },
    lightboxKey(e) {
      if (!this.lightbox.open) return;
      if (e.key === 'Escape') this.closeLightbox();
      else if (e.key === 'ArrowLeft') this.lbPrev();
      else if (e.key === 'ArrowRight') this.lbNext();
      else if (e.key === 'r' || e.key === 'R') this.reveal();
    },
    lbTouchStart(e) { this._touchX = e.changedTouches[0].clientX; },
    lbTouchEnd(e) {
      const dx = e.changedTouches[0].clientX - this._touchX;
      if (Math.abs(dx) > 40) (dx < 0 ? this.lbNext() : this.lbPrev());
    },

    async loadToWorkspace() {
      await this._metaLoad;
      this.applyFields(this.selectedFields);
      this.closeLightbox();
      this.tab = 'generate';
      this.resizeTextareas();
      this.flash('Loaded settings into Generate');
    },

    // Two-click delete: the first click arms, the second fires (click.outside
    // disarms). Then step to a neighbour, or close if the gallery is empty.
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
      // A search-result list refetches on the next open/search.
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
      const nextIdx = Math.min(idx, this.gallery.length - 1);
      this.lightbox.index = nextIdx;
      this.lbReveal = false;
      this.selectImage(this.gallery[nextIdx]);
      this.flash('Deleted');
    },

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
      await this._metaLoad;
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
      // Drops bypass accept="image/png", and only PNGs carry the parameters chunk.
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

    // Map a normalised workspace-fields dict onto the form.
    applyFields(f) {
      if (!f) return;
      const keys = ['prompt', 'neg', 'steps', 'cfg', 'sampler', 'scheduler',
                    'seed', 'shift', 'strength', 'width', 'height',
                    'teacacheOn', 'teacache', 'teacacheCalibrated', 'teacacheForecast',
                    'teacacheRule',
                    'deepcacheOn', 'deepcache'];
      for (const k of keys) if (f[k] !== undefined) this.form[k] = f[k];
      // Metadata from another family's image may name a sampler the active
      // family can't run.
      this.syncSampler();
      this.syncScheduler();
      if (f.detailer) this.applyDetailer(f.detailer);
      if (f.upscale) this.applyUpscale(f.upscale);
    },

    // Restore the upscaler panel from `upscale` metadata; absent leaves it alone.
    applyUpscale(u) {
      this.upscale.enabled = u.enabled !== false;
      for (const k of ['scale', 'denoise', 'tile', 'overlap', 'teacache', 'base', 'prompt']) {
        if (u[k] !== undefined) this.upscale[k] = u[k];
      }
    },

    // Restore the detailer panel from `detailer` metadata; absent leaves it alone.
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

    // Parse A1111-style parameters pasted into the prompt box onto the form,
    // like SD WebUI's read-params arrow.
    async importFromPrompt() {
      const text = this.form.prompt;
      if (!text || !text.trim()) { this.flash('Paste generation parameters into the prompt first'); return; }
      try {
        const r = await fetchJSON('/api/metadata/parse_text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        });
        // Only an actual settings key counts, not just prompt/neg/seed.
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

    // Arrow-key navigation among a group's sibling <button>s. Radio groups
    // activate on move; tablists pass activate=false so arrows only move focus
    // and Enter/Space activates (tab handlers can be expensive).
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
    // A small FIFO (capped at 4) so a burst of messages each get a turn.
    flash(msg, kind) {
      if (!msg) return;
      this._toastQueue.push({ msg, kind: kind || this._toastKind(msg) });
      if (this._toastQueue.length > 4) this._toastQueue.length = 4;
      if (!this._toastShowing) this._toastNext();
    },
    // Infer severity from the message unless `kind` is given.
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
        // A gap so the fade-out completes before the next toast fades in.
        setTimeout(() => this._toastNext(), 250);
      }, 2500);
    },

    // Copy the lightbox's raw parameters string. navigator.clipboard needs a
    // secure context, so plain-HTTP LAN clients fall back to execCommand.
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

    async refreshExtensions() {
      try {
        const r = await fetchJSON('/api/extensions');
        this.extensions = r.extensions || [];
      } catch (e) { /* server may be mid-startup; retried on next open */ }
    },

    // x-effect: fetch only while the Extensions settings tab is active.
    refreshExtensionsIfOpen(tab) {
      if (tab === 'extensions') this.refreshExtensions();
    },

    isExtTab(tab) {
      return this.extTabs.some(t => t.id === tab);
    },

    switchExtTab(t) {
      this.tab = t.id;
    },

    // x-effect: mount/unmount the extension tab panel when `tab` changes.
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
      // Every registered settings panel gets a child container, mounted once.
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
        // Installs run as a job; the terminal SSE event resolves the promise.
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
        // Extension JS is injected server-side, so its UI needs a page reload.
        if (ev.extension && ev.extension.has_ui) this.flash('Reload the page to load the extension UI');
        // Surface a skipped-deps / load-error note.
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
        // Like install, update runs as a job.
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

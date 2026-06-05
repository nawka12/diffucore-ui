// Diffucore UI — Alpine state + streaming. No build step.

document.addEventListener('alpine:init', () => {
  Alpine.data('app', () => ({
    // ── model rack ──────────────────────────────────────────────
    modelType: 'SD/SDXL',
    checkpoints: [], dits: [], vaes: [], tes: [], loras: [], detailers: [],
    checkpoint: '', dit: '', vae: '', te: '', clip: '', fluxCheckpoint: '',
    perf: { compile: false, cudaGraphs: false, channelsLast: false, offload: 'full' },
    recommendedOffload: 'full',   // GPU-VRAM-based default from the backend (set on init)
    status: 'No model loaded',
    modelLoaded: false,
    loadingModel: false,
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
    },

    // ── <lora:…> autocomplete in the prompt ─────────────────────
    loraAC: { open: false, items: [], index: 0, start: -1 },

    inputImage: null,
    maskImage: null,
    dragKey: null,
    maskBrush: 40,
    maskPainted: false,

    // ── detailer (ADetailer-style passes after generate) ────────
    // `models` is a stack of {model, prompt} run in sequence; rest is shared.
    detail: {
      enabled: false, neg: '',
      models: [{ model: '', prompt: '' }],
      confidence: 0.3, strength: 0.4,
      dilation: 4, padding: 32, blur: 4, maxDet: 0,
    },

    // ── generation output ───────────────────────────────────────
    busy: false,
    cancelling: false,
    progress: { step: 0, total: 0 },
    resultUrl: null,
    previewUrl: null,
    preview: true,
    info: '',
    lastSeed: -1,

    // ── shared queue (broadcast over /api/events to every device) ──
    queue: [],            // [{id, kind, label, status}] — running first, then pending
    runningJob: null,     // id of the job currently on the GPU (any device)
    runProg: { step: 0, total: 0 },  // progress of the running job (for the queue panel)
    myJobId: null,        // id of the job THIS device submitted and is watching
    _jobWaiters: {},      // id -> resolve fn, fulfilled by the terminal SSE event

    // ── OSS calibration ─────────────────────────────────────────
    calibrating: false,
    ossCalibrated: null,          // null = unknown, true/false = checked
    ossInfo: '',

    // ── gallery ─────────────────────────────────────────────────
    gallery: [],
    galleryGroups: [],
    selected: null,
    selectedMeta: '',
    selectedFields: {},
    lightbox: { open: false, index: 0, info: false },

    // ── metadata reader ─────────────────────────────────────────
    metaPreview: null,
    metaText: '',
    metaFields: null,

    // ── x/y/z sweep (txt2img only — reuses the shared form for base params) ──
    xyzSweep: false,
    axes: {
      x: { type: 'Sampler', text: '', list: ['euler', 'dpmpp_2m', 'dpmpp_2m_sde'] },
      y: { type: 'Steps', text: '15, 25, 35', list: [] },
      z: { type: 'None', text: '', list: [] },
    },
    xyzGrids: [],
    xyzInfo: '',

    toast: '',

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
    get offloadOptions() {
      // "stream" streams the FLUX DiT blocks — FLUX-only; the rest stage the
      // whole backbone (or keep it resident) and work for every family.
      return this.modelType === 'FLUX'
        ? ['stream', 'full', 'encoders', 'none']
        : ['full', 'encoders', 'none'];
    },
    get sweeping() {
      return this.mode === 't2i' && this.xyzSweep;
    },

    // X/Y/Z axes whose values come from a known set get a multi-select; numeric
    // axes (Steps / CFG / Seed) keep a free-text comma list.
    axisIsList(axis) { return axis.type === 'Sampler' || axis.type === 'Scheduler'; },
    axisOptions(axis) { return axis.type === 'Scheduler' ? this.schedulers : this.samplers; },
    axisValues(axis) { return this.axisIsList(axis) ? axis.list.join(', ') : axis.text; },
    get progressPct() {
      const t = this.progress.total;
      return t > 0 ? Math.round((this.progress.step / t) * 100) : 0;
    },
    get progressLabel() {
      const t = this.progress.total;
      return t > 0 ? `${this.progress.step} / ${t}  (${this.progressPct}%)` : 'Starting…';
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
      await this.refreshModels();
      this.connectEvents();
    },

    // ── shared events (one SSE stream per device) ───────────────
    connectEvents() {
      const es = new EventSource('/api/events');
      es.onmessage = (e) => this.onServerEvent(JSON.parse(e.data));
      // On error the browser auto-reconnects; the fresh snapshot re-syncs us.
    },

    onServerEvent(ev) {
      switch (ev.type) {
        case 'snapshot':
          this.applyState(ev);
          this.queue = ev.jobs; this.runningJob = ev.running;
          if (ev.progress) this.runProg = { step: ev.progress.step, total: ev.progress.total };
          break;
        case 'queue':
          this.queue = ev.jobs; this.runningJob = ev.running;
          break;
        case 'status':
          this.applyState(ev);
          break;
        case 'progress':
          // Running-job progress feeds the queue panel; the main bar only
          // tracks THIS device's own job, so a queued device isn't shown
          // another device's progress.
          this.runProg = { step: ev.step, total: ev.total };
          if (ev.job === this.myJobId) this.progress = { step: ev.step, total: ev.total };
          break;
        case 'preview':
          if (ev.job === this.myJobId) this.previewUrl = ev.image;
          break;
        case 'done':
        case 'error':
        case 'cancelled': {
          const w = this._jobWaiters[ev.job];
          if (w) { delete this._jobWaiters[ev.job]; w(ev); }
          break;
        }
      }
    },

    // Submit a job, then resolve once its terminal event arrives on the stream.
    // Live progress/preview are handled globally by onServerEvent.
    async submitJob(url, payload) {
      const r = await (await fetch(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })).json();
      this.myJobId = r.job;
      return new Promise((resolve) => { this._jobWaiters[r.job] = resolve; });
    },

    // Reflect server-side model-load state (shared across devices/refresh).
    applyState(s) {
      this.status = s.status;
      this.modelLoaded = !!s.loaded;
      if (s.last_seed !== undefined) this.lastSeed = s.last_seed;
      if (s.load_form) this.restoreLoadForm(s.load_form);
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
      this.syncSampler();
      this.syncScheduler();
    },

    async refreshModels() {
      const m = await (await fetch('/api/models')).json();
      this.checkpoints = m.checkpoints; this.dits = m.dits;
      this.vaes = m.vaes; this.tes = m.tes; this.loras = m.loras;
      this.detailers = m.detailers || [];
      this.samplersSd = m.samplers_sd;
      this.samplersAnima = m.samplers_anima;
      this.samplersFlux = m.samplers_flux;
      this.schedulersSd = m.schedulers_sd;
      this.schedulersAnima = m.schedulers_anima;
      this.schedulersFlux = m.schedulers_flux;
      this.paramTypes = m.xyz_param_types;
      this.recommendedOffload = m.recommended_offload || 'full';
      this.perf.offload = this.recommendedOffload;
      this.uiId = m.ui_id; this.diffId = m.diff_id;
      this.checkpoint = this.checkpointChoices[0];
      this.dit = this.ditChoices[0];
      this.vae = this.vaeChoices[0];
      this.te = this.teChoices[0];
      this.clip = this.teChoices[0];
      this.detail.models[0].model = this.detailerChoices[0];
      this.syncSampler();
      this.syncScheduler();
      // Hydrate from server-side load state last, so a model already loaded by
      // another device (or before a refresh) restores its exact selections.
      this.applyState(m);
    },

    setModelType(type) {
      this.modelType = type;
      this.syncSampler();
      this.syncScheduler();
      // FLUX must stream its DiT to fit a 24 GB card; the rest use the
      // GPU-VRAM-based default the backend recommended at startup.
      this.perf.offload = (type === 'FLUX') ? 'stream' : this.recommendedOffload;
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
        };
        // Load is queued like any job; it waits for a running generation to
        // finish. The server also broadcasts the new state to every device.
        const ev = await this.submitJob('/api/load', body);
        if (ev.type === 'done') {
          this.status = ev.status;
          this.modelLoaded = !!ev.loaded;
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
        this.myJobId = null;
      }
    },

    // ── files ───────────────────────────────────────────────────
    onFile(evt, key) { this.readImage(evt.target.files[0], key); },
    onDrop(evt, key) { this.dragKey = null; this.readImage(evt.dataTransfer.files[0], key); },
    readImage(f, key) {
      if (!f) return;
      const r = new FileReader();
      r.onload = () => { this[key] = r.result; };
      r.readAsDataURL(f);
    },

    // ── inpaint mask painting ───────────────────────────────────
    // Draw the input image onto the visible canvas and keep a same-size
    // offscreen buffer that holds the mask as white-on-black (the convention
    // the engine expects). Both buffers are stashed on the canvas element so
    // Alpine never proxies the DOM nodes.
    initMask(c) {
      if (!c || !this.inputImage) return;
      const img = new Image();
      img.onload = () => {
        c.width = img.naturalWidth;
        c.height = img.naturalHeight;
        c.getContext('2d').drawImage(img, 0, 0);
        const m = document.createElement('canvas');
        m.width = img.naturalWidth;
        m.height = img.naturalHeight;
        const mctx = m.getContext('2d');
        mctx.fillStyle = '#000';
        mctx.fillRect(0, 0, m.width, m.height);
        c._mask = m;
        c._base = img;
        c._painting = false;
        this.maskPainted = false;
      };
      img.src = this.inputImage;
    },

    maskDown(e) {
      const c = e.currentTarget;
      if (!c._mask) return;
      c._painting = true;
      c._last = this.maskPos(c, e);
      this.maskStroke(c, c._last, c._last);
    },
    maskMove(e) {
      const c = e.currentTarget;
      if (!c._painting) return;
      const p = this.maskPos(c, e);
      this.maskStroke(c, c._last, p);
      c._last = p;
    },
    maskUp(e) { e.currentTarget._painting = false; },

    maskPos(c, e) {
      const r = c.getBoundingClientRect();
      return {
        x: (e.clientX - r.left) * (c.width / r.width),
        y: (e.clientY - r.top) * (c.height / r.height),
      };
    },
    maskStroke(c, a, b) {
      this.maskSeg(c.getContext('2d'), a, b, 'rgba(232,160,101,0.5)');
      this.maskSeg(c._mask.getContext('2d'), a, b, '#fff');
      this.maskPainted = true;
    },
    maskSeg(ctx, a, b, color) {
      ctx.strokeStyle = color;
      ctx.lineWidth = this.maskBrush;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    },
    clearMask() {
      const c = this.$refs.maskCanvas;
      if (!c || !c._base) return;
      c.getContext('2d').drawImage(c._base, 0, 0);
      const mctx = c._mask.getContext('2d');
      mctx.fillStyle = '#000';
      mctx.fillRect(0, 0, c._mask.width, c._mask.height);
      this.maskPainted = false;
    },
    exportMask() {
      const c = this.$refs.maskCanvas;
      return c && c._mask ? c._mask.toDataURL('image/png') : null;
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
    loraAutocomplete(el) {
      const before = el.value.slice(0, el.selectionStart);
      const lt = before.lastIndexOf('<');
      const m = lt === -1 ? null : before.slice(lt + 1).match(/^(?:lora:)?([^:>]*)$/i);
      if (!m) { this.loraAC.open = false; return; }
      const q = m[1].toLowerCase();
      const items = this.loras.filter((n) => n.toLowerCase().includes(q));
      if (!items.length) { this.loraAC.open = false; return; }
      Object.assign(this.loraAC, { open: true, items, index: 0, start: lt });
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
    applyLora(name) {
      const el = this.$refs.promptEl;
      const before = el.value.slice(0, this.loraAC.start);
      const after = el.value.slice(el.selectionStart);
      const insert = `<lora:${name}:1.0>`;
      this.form.prompt = before + insert + after;
      this.loraAC.open = false;
      this.$nextTick(() => {
        const caret = before.length + insert.length;
        el.focus();
        el.setSelectionRange(caret, caret);
        this.autogrow(el);
      });
    },

    // ── generate ────────────────────────────────────────────────
    runGenerate() {
      return this.sweeping ? this.generateXyz() : this.generate();
    },

    async generate() {
      if (!this.modelLoaded) { this.flash('Load a model first'); return; }
      if (this.mode === 'i2i' && !this.inputImage) { this.flash('Provide an input image'); return; }
      if (this.mode === 'inpaint') {
        if (!this.inputImage) { this.flash('Provide an input image'); return; }
        if (!this.maskPainted) { this.flash('Paint a mask over the image'); return; }
        this.maskImage = this.exportMask();
      }
      this.busy = true;
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
        };
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
      } catch (e) {
        this.info = 'Error: ' + e;
      } finally {
        this.busy = false;
        this.cancelling = false;
        this.previewUrl = null;
        this.myJobId = null;
      }
    },

    // Cancel a job by id (defaults to this device's running job). A running job
    // stops at its next sampling step; a still-queued job is dropped.
    async cancel(jobId) {
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
      if (!this.modelLoaded) { this.flash('Load a model first'); return; }
      this.busy = true;
      this.progress = { step: 0, total: 0 };
      this.xyzInfo = '';
      const payload = {
        prompt: this.form.prompt, neg: this.form.neg,
        width: this.form.width, height: this.form.height,
        steps: this.form.steps, cfg: this.form.cfg,
        sampler: this.form.sampler, scheduler: this.form.scheduler,
        seed: this.form.seed, shift: this.form.shift,
        x_type: this.axes.x.type, x_vals: this.axisValues(this.axes.x),
        y_type: this.axes.y.type, y_vals: this.axisValues(this.axes.y),
        z_type: this.axes.z.type, z_vals: this.axisValues(this.axes.z),
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
        this.myJobId = null;
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
        this.ossCalibrated = (await (await fetch('/api/oss_status?' + q)).json()).calibrated;
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

    // ── detailer model stack ────────────────────────────────────
    addDetailModel() {
      this.detail.models.push({ model: this.detailerChoices[0], prompt: '' });
    },
    removeDetailModel(i) {
      this.detail.models.splice(i, 1);
      if (!this.detail.models.length) this.detail.models.push({ model: this.detailerChoices[0], prompt: '' });
    },

    // ── gallery ─────────────────────────────────────────────────
    async openGallery() {
      this.tab = 'gallery';
      this.selected = null;
      this.selectedMeta = '';
      this.gallery = (await (await fetch('/api/gallery')).json()).images;
      this.buildGalleryGroups();
    },

    // Group the (date-desc) gallery list into day sections, carrying each
    // image's flat index so the lightbox carousel still pages across all of them.
    buildGalleryGroups() {
      const groups = [];
      let cur = null;
      this.gallery.forEach((img, i) => {
        if (!cur || cur.date !== img.date) {
          cur = { date: img.date, label: this.dateLabel(img.date), images: [] };
          groups.push(cur);
        }
        cur.images.push({ img, index: i });
      });
      this.galleryGroups = groups;
    },

    dateLabel(d) {
      const m = /^(\d{2})-(\d{2})-(\d{4})$/.exec(d || '');
      if (!m) return d || 'Unknown date';
      const dt = new Date(+m[3], +m[2] - 1, +m[1]);
      const today = new Date(); today.setHours(0, 0, 0, 0);
      const diff = Math.round((today - dt) / 86400000);
      if (diff === 0) return 'Today';
      if (diff === 1) return 'Yesterday';
      return dt.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' });
    },

    async selectImage(img) {
      this.selected = img;
      const r = await (await fetch('/api/metadata?path=' + encodeURIComponent(img.path))).json();
      this.selectedMeta = r.raw;
      this.selectedFields = r.fields;
    },

    // ── lightbox carousel ───────────────────────────────────────
    openLightbox(i) {
      this.lightbox.index = i;
      this.lightbox.open = true;
      this.selectImage(this.gallery[i]);
    },
    closeLightbox() { this.lightbox.open = false; },
    lbPrev() { this.lbGo(-1); },
    lbNext() { this.lbGo(1); },
    lbGo(d) {
      const n = this.gallery.length;
      if (!n) return;
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

    loadToWorkspace() {
      this.applyFields(this.selectedFields);
      this.closeLightbox();
      this.tab = 'generate';
      this.resizeTextareas();
      this.flash('Loaded settings into Generate');
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
      const pre = new FileReader();
      pre.onload = () => { this.metaPreview = pre.result; };
      pre.readAsDataURL(f);
      const fd = new FormData();
      fd.append('file', f);
      const r = await (await fetch('/api/metadata/parse', { method: 'POST', body: fd })).json();
      this.metaText = r.text;
      this.metaFields = Object.keys(r.fields).length ? r.fields : null;
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
                    'seed', 'shift', 'strength', 'width', 'height'];
      for (const k of keys) if (f[k] !== undefined) this.form[k] = f[k];
    },

    // ── toast ───────────────────────────────────────────────────
    flash(msg) {
      this.toast = msg;
      clearTimeout(this._toastTimer);
      this._toastTimer = setTimeout(() => { this.toast = ''; }, 3500);
    },
  }));
});

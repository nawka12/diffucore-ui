// Diffucore UI — Alpine state + streaming. No build step.

document.addEventListener('alpine:init', () => {
  Alpine.data('app', () => ({
    // ── model rack ──────────────────────────────────────────────
    modelType: 'SD/SDXL',
    checkpoints: [], dits: [], vaes: [], tes: [], loras: [],
    checkpoint: '', dit: '', vae: '', te: '',
    perf: { compile: false, cudaGraphs: false, channelsLast: false },
    status: 'No model loaded',
    modelLoaded: false,
    loadingModel: false,
    animaApplied: false,
    uiId: 'diffucore-ui', diffId: 'diffucore',

    // ── shared option sets ──────────────────────────────────────
    samplers: [],
    schedulersSd: ['karras'],
    schedulersAnima: ['flow'],
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
    inputImage: null,
    maskImage: null,
    dragKey: null,
    maskBrush: 40,
    maskPainted: false,

    // ── generation output ───────────────────────────────────────
    busy: false,
    progress: { step: 0, total: 0 },
    resultUrl: null,
    info: '',
    lastSeed: -1,

    // ── gallery ─────────────────────────────────────────────────
    gallery: [],
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
    get schedulers() {
      return this.modelType === 'Anima' ? this.schedulersAnima : this.schedulersSd;
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

    choices(list, where) {
      return list.length ? list : [`(none in ${where})`];
    },

    // ── init ────────────────────────────────────────────────────
    async init() {
      await this.refreshModels();
    },

    async refreshModels() {
      const m = await (await fetch('/api/models')).json();
      this.checkpoints = m.checkpoints; this.dits = m.dits;
      this.vaes = m.vaes; this.tes = m.tes; this.loras = m.loras;
      this.samplers = m.samplers;
      this.schedulersSd = m.schedulers_sd;
      this.schedulersAnima = m.schedulers_anima;
      this.paramTypes = m.xyz_param_types;
      this.status = m.status;
      this.lastSeed = m.last_seed;
      this.uiId = m.ui_id; this.diffId = m.diff_id;
      this.checkpoint = this.checkpointChoices[0];
      this.dit = this.ditChoices[0];
      this.vae = this.vaeChoices[0];
      this.te = this.teChoices[0];
      this.syncScheduler();
    },

    setModelType(type) {
      this.modelType = type;
      this.syncScheduler();
      if (type === 'Anima' && !this.animaApplied) {
        // sensible Anima defaults, applied once
        this.animaApplied = true;
        this.form.sampler = 'er_sde';
        this.form.steps = 30;
        this.form.cfg = 4.0;
      }
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
          checkpoint: this.checkpoint,
          dit: this.dit, vae: this.vae, te: this.te,
          compile: this.perf.compile,
          cuda_graphs: this.perf.cudaGraphs,
          channels_last: this.perf.channelsLast,
        };
        const r = await (await fetch('/api/load', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })).json();
        this.status = r.status;
        this.modelLoaded = /^(Loaded|Model already loaded)/.test(r.status);
        if (!this.modelLoaded) this.flash(r.status);
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

    // ── ndjson streaming ────────────────────────────────────────
    async stream(url, payload, onEvent) {
      const res = await fetch(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (line) onEvent(JSON.parse(line));
        }
      }
      buf = buf.trim();
      if (buf) onEvent(JSON.parse(buf));
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
      this.info = '';
      const payload = {
        mode: this.mode,
        prompt: this.form.prompt, neg: this.form.neg,
        sampler: this.form.sampler, scheduler: this.form.scheduler,
        steps: this.form.steps, cfg: this.form.cfg, seed: this.form.seed,
        width: this.form.width, height: this.form.height,
        strength: this.form.strength, shift: this.form.shift,
        input_image: this.mode !== 't2i' ? this.inputImage : null,
        mask_image: this.mode === 'inpaint' ? this.maskImage : null,
      };
      try {
        await this.stream('/api/generate', payload, (ev) => {
          if (ev.type === 'progress') {
            this.progress = { step: ev.step, total: ev.total };
          } else if (ev.type === 'done') {
            this.resultUrl = ev.image_url + '?t=' + Date.now();
            this.info = ev.info;
            this.lastSeed = ev.seed;
          } else if (ev.type === 'error') {
            this.info = 'Error: ' + ev.message;
            this.flash(ev.message);
          }
        });
      } catch (e) {
        this.info = 'Error: ' + e;
      } finally {
        this.busy = false;
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
        await this.stream('/api/xyz', payload, (ev) => {
          if (ev.type === 'progress') {
            this.progress = { step: ev.step, total: ev.total };
          } else if (ev.type === 'done') {
            this.xyzGrids = ev.grids;
            this.xyzInfo = ev.info;
          } else if (ev.type === 'error') {
            this.xyzInfo = 'Error: ' + ev.message;
            this.flash(ev.message);
          }
        });
      } catch (e) {
        this.xyzInfo = 'Error: ' + e;
      } finally {
        this.busy = false;
      }
    },

    // ── gallery ─────────────────────────────────────────────────
    async openGallery() {
      this.tab = 'gallery';
      this.selected = null;
      this.selectedMeta = '';
      this.gallery = (await (await fetch('/api/gallery')).json()).images;
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

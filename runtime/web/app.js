// Character Forge studio, served by the local Diffusers CUDA runtime.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const store = {
  get(k, d) { try { const v = localStorage.getItem('forge.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('forge.' + k, JSON.stringify(v)); } catch {} },
};

const S = {
  cfg: null, refs: store.get('refs', []), mode: store.get('mode', 'sheet'), sheet: store.get('sheet', 'turnaround5'),
  bg: store.get('bg', 'white'), models: store.get('models', null), jobs: [], filter: 'all', modelFilter: '',
  project: null, lastBatch: null, lb: -1, sig: '', selected: new Set(),
};

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || j.detail || r.statusText);
  return j;
}
const viewUrl = (img) => `/view?filename=${encodeURIComponent(img.filename)}&subfolder=${encodeURIComponent(img.subfolder || '')}&type=${img.type || 'output'}`;
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const gb = (b) => (b / 1e9).toFixed(1);

// ------------------------------------------------------------------ config / models
async function loadConfig() {
  S.cfg = await api('/forge/api/config');
  const hw = $('#hw');
  if (hw) { hw.textContent = S.cfg.cuda ? 'CUDA enabled' : 'CUDA unavailable'; hw.title = S.cfg.gpu || ''; hw.classList.toggle('good', !!S.cfg.cuda); }
  if (!S.project) S.project = S.cfg.last_project;
  renderRuntime(S.cfg.runtime);
  renderProjects(); renderSheets(); renderSizes(); renderModelList(); renderModelCards();
}

function renderRuntime(runtime) {
  runtime = runtime || { state: 'unloaded', model_name: null, queue: 0, busy: false };
  const status = $('#modelStatus');
  const detail = $('#runtimeDetail');
  const unload = $('#unloadModel');
  if (!status) return;
  const labels = {
    unloaded: 'Model: unloaded',
    loading: `Loading: ${runtime.model_name || 'model'}…`,
    loaded: `Loaded: ${runtime.model_name || 'model'}`,
  };
  status.textContent = labels[runtime.state] || 'Model: unknown';
  status.className = `model-status ${runtime.state}`;
  status.title = runtime.state === 'loaded' ? 'A model is cached and ready; unload it to release memory.' : 'No completed model is cached.';
  if (detail) detail.textContent = runtime.state === 'loaded'
    ? `${runtime.model_name} is cached${runtime.queue ? ` · ${runtime.queue} queued` : ''}.`
    : runtime.state === 'loading' ? `Loading ${runtime.model_name || 'the selected model'}; progress is shown in the Studio header.`
    : 'No model loaded. It will load when you queue an image.';
  if (unload) { unload.disabled = runtime.state !== 'loaded' || runtime.busy || runtime.queue > 0; unload.title = unload.disabled && runtime.state === 'loaded' ? 'Stop renders and clear the queue before unloading.' : ''; }
}

function renderProjects() {
  const sel = $('#project');
  const list = [...new Set([...(S.cfg.projects || []), S.project])];
  sel.innerHTML = list.map(p => `<option ${p === S.project ? 'selected' : ''}>${esc(p)}</option>`).join('');
}

function renderSheets() {
  const sheets = S.cfg.presets.sheets;
  $('#sheetCards').innerHTML = sheets.map(s => `<div class="card ${s.id === S.sheet ? 'on' : ''}" data-id="${s.id}">${esc(s.name)}</div>`).join('');
  $$('#sheetCards .card').forEach(c => c.onclick = () => {
    S.sheet = c.dataset.id; store.set('sheet', S.sheet); renderSheets();
    const sh = sheets.find(x => x.id === S.sheet); if (sh) $('#size').value = sh.size; updateQueueBtn();
  });
}

function renderSizes() {
  const sizes = S.cfg.presets.sizes;
  const cur = $('#size').value || store.get('size', 'sheet-wide');
  $('#size').innerHTML = Object.entries(sizes).map(([k, v]) => `<option value="${k}">${esc(v.label)}${v.w ? ` · ${v.w}×${v.h}` : ''}</option>`).join('');
  $('#size').value = sizes[cur] ? cur : 'sheet-wide';
}

function licTag(m) { return m.commercial_ok ? '<span class="tag ok">commercial OK</span>' : '<span class="tag nc">check license</span>'; }
function selectedMaxRefs() {
  const selected = (S.models || []).map(id => S.cfg.models.find(m => m.id === id)).filter(Boolean);
  return selected.length ? Math.max(...selected.map(m => m.max_refs)) : 1;
}

function renderModelList() {
  const ms = S.cfg.models;
  if (S.models === null) S.models = ms.filter(m => m.installed && m.recommended).map(m => m.id);
  S.models = S.models.filter(id => ms.some(m => m.id === id));
  $('#modelList').innerHTML = ms.map(m => `
    <label class="mrow ${m.installed ? '' : 'off'}">
      <input type="checkbox" value="${m.id}" ${S.models.includes(m.id) && m.installed ? 'checked' : ''} ${m.installed ? '' : 'disabled'}>
      <div><div class="nm">${esc(m.name)} ${m.recommended ? '<span class="tag rec">top pick</span>' : ''}${licTag(m)}</div>
      <div class="meta">${m.installed ? `${esc(m.params)} · ${esc(m.speed)} · up to ${m.max_refs} refs` : `Not downloaded (${m.total_gb} GB): get it in the Models tab`}</div></div>
    </label>`).join('');
  $$('#modelList input').forEach(i => i.onchange = () => {
    S.models = $$('#modelList input:checked').map(x => x.value); store.set('models', S.models); updateQueueBtn();
    renderRefs(); updateSamplingHint();
  });
  const mf = $('#modelFilter'), cur = mf.value;
  mf.innerHTML = '<option value="">All models</option>' + ms.map(m => `<option value="${m.id}">${esc(m.name)}</option>`).join('');
  mf.value = cur;
  const missing = ms.filter(m => !m.installed).length;
  const anyInstalled = ms.some(m => m.installed);
  $('#modelsBadge').textContent = anyInstalled ? '' : '!';
  $('#modelsBadge').classList.toggle('hidden', anyInstalled);
  updateQueueBtn();
}

function updateSamplingHint() {
  const selected = (S.models || []).map(id => S.cfg.models.find(m => m.id === id)).filter(Boolean);
  const qwen = selected.find(m => m.family === 'qwen-edit');
  const hint = $('#samplingHint');
  if (qwen) {
    hint.textContent = 'Qwen Rapid AIO: fixed Qwen Euler ancestral Beta-flow sampler · 4 steps default · guidance (true CFG) 1.0 default. Higher steps can improve detail but take longer; guidance above 1 enables negative prompting.';
    $('#steps').placeholder = '4 (default)'; $('#cfg').placeholder = '1.0 (default)';
  } else {
    hint.textContent = 'SDXL: steps and CFG are passed to the DPM++ Karras sampler. Higher values take longer and can overconstrain the prompt.';
    $('#steps').placeholder = 'model default'; $('#cfg').placeholder = 'model default';
  }
}

function renderModelCards() {
  $('#modelsDir').textContent = S.cfg.models_dir;
  $('#freeGb').textContent = S.cfg.free_gb;
  $('#tokenState').textContent = S.cfg.has_token ? '✓ token saved' : 'no token';
  $('#modelCards').innerHTML = S.cfg.models.map(m => {
    const d = m.download || {};
    let act;
    if (m.installed) act = `<span class="inst">✓ Installed</span>`;
    else if (d.state === 'downloading') {
      const pct = d.total ? Math.min(100, 100 * d.done / d.total) : 0;
      act = `<div style="flex:1"><div class="bar"><i style="width:${pct.toFixed(1)}%"></i></div>
             <div class="facts">${gb(d.done)} / ${gb(d.total)} GB · ${esc(d.file || '')}</div></div><button data-cancel>Cancel</button>`;
    } else if (d.state === 'queued') act = `<span class="dim">Queued for download…</span>`;
    else {
      const need = m.bytes_needed / 1e9;
      const tooBig = need > S.cfg.free_gb - 1;
      act = `<button data-dl="${m.id}" ${tooBig ? 'disabled' : ''}>⬇ Download ${need.toFixed(1)} GB</button>
             ${tooBig ? '<span class="dim" style="color:var(--bad)">not enough free space</span>' : ''}
             ${m.bytes_done > 0 ? `<span class="dim">${gb(m.bytes_done)} GB already here</span>` : ''}`;
    }
    const err = d.state === 'error' ? `<p style="color:var(--bad)">${esc(d.error)}</p>` : '';
    const gate = m.gated ? `<p class="dim">🔒 Gated: needs a Hugging Face token and one click on <a href="${m.gate_url}" target="_blank" style="color:var(--acc)">the model page</a>.</p>` : '';
    return `<div class="mcard">
      <h4>${esc(m.name)} ${m.recommended ? '<span class="tag rec">top pick</span>' : ''}</h4>
      <div class="facts">${esc(m.maker)} · released ${esc(m.released)} · ${esc(m.params)} · ${m.total_gb} GB on disk · up to ${m.max_refs} reference images</div>
      <div class="facts">${esc(m.sampler || 'default sampler')} · ${esc(m.guidance_name || 'CFG')} ${m.defaults?.cfg} · ${m.defaults?.steps} steps default</div>
      <div class="lic ${m.commercial_ok ? 'ok' : 'nc'}">${esc(m.license)}: ${esc(m.commercial)}</div>
      <p>${esc(m.blurb)}</p>${gate}${err}
      <div class="act">${act}</div></div>`;
  }).join('');
  $$('[data-dl]').forEach(b => b.onclick = async () => { b.disabled = true; await api('/forge/api/download', { id: b.dataset.dl }); pollConfig(); });
  $$('[data-cancel]').forEach(b => b.onclick = async () => { await api('/forge/api/download/cancel', {}); pollConfig(); });
}

let cfgTimer = null;
async function pollConfig() {
  clearTimeout(cfgTimer);
  try { await loadConfig(); } catch {}
  const busy = S.cfg && S.cfg.models.some(m => m.download && ['downloading', 'queued'].includes(m.download.state));
  if (busy || !$('#models').classList.contains('hidden')) cfgTimer = setTimeout(pollConfig, busy ? 1500 : 5000);
}

// ------------------------------------------------------------------ references
function renderRefs() {
  store.set('refs', S.refs);
  $('#refs').innerHTML = S.refs.map((r, i) => `
    <div class="ref"><img src="${r.url}" alt=""><span class="n">${i === 0 ? 'CHAR' : 'REF' + (i + 1)}</span>
      <button class="x" data-i="${i}" title="Remove">✕</button>
      ${i > 0 ? `<button class="mv" data-i="${i}" title="Make this the main character">◀ 1st</button>` : ''}</div>`).join('');
  $$('#refs .x').forEach(b => b.onclick = (e) => { e.stopPropagation(); S.refs.splice(+b.dataset.i, 1); renderRefs(); updateQueueBtn(); });
  $$('#refs .mv').forEach(b => b.onclick = (e) => { e.stopPropagation(); const [r] = S.refs.splice(+b.dataset.i, 1); S.refs.unshift(r); renderRefs(); });
  const max = selectedMaxRefs();
  $('.drop-hint').innerHTML = S.refs.length ? (max > 1 ? `Drop more reference images (${S.refs.length}/${max})` : 'Drop an image to replace the character reference') :
    'Drop the character image here<br><small>or click to choose · Ctrl+V to paste · first image = the character</small>';
  $('#refHint').innerHTML = max > 1 ? `Qwen Rapid AIO can use up to ${max} references for identity, outfit or location. The SDXL alternatives use the first image only.` : 'The selected SDXL models use one character reference. Describe clothing, props and locations in the prompt.';
}

async function uploadFiles(files) {
  const max = selectedMaxRefs();
  for (const f of files) {
    if (!f.type.startsWith('image/')) continue;
    if (max > 1 && S.refs.length >= max) break;
    const fd = new FormData();
    fd.append('image', f, f.name || 'pasted.png');
    fd.append('subfolder', 'forge_refs');
    fd.append('type', 'input');
    const r = await fetch('/upload/image', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok) { $('#queueMsg').textContent = j.error || j.detail || 'Upload failed'; $('#queueMsg').className = 'msg err'; return; }
    const name = j.subfolder ? `${j.subfolder}/${j.name}` : j.name;
    if (max === 1) S.refs = [];
    S.refs.push({ name, url: viewUrl({ filename: j.name, subfolder: j.subfolder, type: 'input' }) });
  }
  renderRefs(); updateQueueBtn();
}

function setupDrop() {
  const drop = $('#drop'), fi = $('#fileIn');
  drop.onclick = () => fi.click();
  fi.onchange = () => { uploadFiles(fi.files); fi.value = ''; };
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => uploadFiles(e.dataTransfer.files));
  document.addEventListener('paste', e => {
    if (['TEXTAREA', 'INPUT'].includes(document.activeElement.tagName)) return;
    const files = [...e.clipboardData.items].filter(i => i.kind === 'file').map(i => i.getAsFile());
    if (files.length) uploadFiles(files);
  });
}

// ------------------------------------------------------------------ queueing
function formReq() {
  const n = (id) => { const v = $(id).value; return v === '' ? null : Number(v); };
  return {
    project: S.project, mode: S.mode, sheet: S.sheet, bg: S.bg,
    notes: $('#notes').value, scene: $('#scene').value, raw: $('#raw').value, negative: $('#negative').value,
    refs: S.refs.map(r => r.name), models: S.models || [], count: n('#count') || 1,
    size: $('#size').value, fast: $('#fast').checked, seed_mode: $('#seedMode').value, seed: n('#seed'),
    steps: n('#steps'), cfg: n('#cfg'), width: n('#w'), height: n('#h'),
    reference_strength: Number($('#referenceStrength').value),
  };
}

function updateQueueBtn() {
  const n = Math.max(1, Number($('#count').value) || 1);
  const ms = (S.models || []).filter(id => S.cfg && S.cfg.models.find(m => m.id === id && m.installed));
  const total = n * ms.length;
  const b = $('#queueBtn');
  b.textContent = total ? `Queue ${total} image${total > 1 ? 's' : ''}${ms.length > 1 ? ` (${n} × ${ms.length} models)` : ''}` : 'Pick a downloaded model';
  b.disabled = !total || !S.refs.length;
  if (!S.refs.length && total) b.textContent = 'Drop a character image first';
  const sdxlOnly = ms.length > 0 && ms.every(id => S.cfg.models.find(m => m.id === id).family === 'sdxl');
  $('#referenceStrength').disabled = !sdxlOnly;
  $('#referenceStrength').title = sdxlOnly ? '' : 'Only used by the SDXL face adapter. Qwen uses native image-edit conditioning.';
}

async function queue(reqOverride) {
  const req = reqOverride || formReq();
  const msg = $('#queueMsg');
  msg.className = 'msg'; msg.textContent = 'Queueing…';
  try {
    const r = await api('/forge/api/queue', req);
    S.lastBatch = r.batch;
    msg.className = r.errors.length ? 'msg err' : 'msg ok';
    msg.textContent = `Queued ${r.queued}.` + (r.errors.length ? '\n' + r.errors.join('\n') : ' You can leave it running and come back.');
    pollJobs();
  } catch (e) { msg.className = 'msg err'; msg.textContent = e.message; }
}

// ------------------------------------------------------------------ gallery
let jobsTimer = null;
async function pollJobs() {
  clearTimeout(jobsTimer);
  try {
    const r = await api('/forge/api/jobs?project=' + encodeURIComponent(S.project));
    S.jobs = r.jobs;
    const busy = r.queue_running + r.queue_pending;
    const q = $('#qstat');
    const cur = r.current;
    renderRuntime(r.runtime);
    if (cur) {
      const pct = Math.round((cur.progress || 0) * 100);
      q.textContent = `${cur.stage || 'Rendering'}${cur.stage === 'sampling' ? ' · ' + pct + '%' : ''} · ${duration(cur.elapsed)} · ${r.queue_pending} waiting`;
      q.title = cur.steps ? `${cur.step || 0} / ${cur.steps} sampling steps` : '';
    } else q.textContent = busy ? `Queued · ${r.queue_pending} waiting` : 'Idle';
    q.classList.toggle('busy', !!busy);
    document.title = busy ? `(${busy}) Character Forge` : 'Character Forge';
    renderGrid();
    jobsTimer = setTimeout(pollJobs, busy ? 2000 : 6000);
  } catch (e) { $('#qstat').textContent = 'Server disconnected · retrying'; $('#qstat').classList.remove('busy'); jobsTimer = setTimeout(pollJobs, 5000); }
}

function duration(seconds) { const s = Math.max(0, Math.floor(seconds || 0)); return s < 60 ? s + 's' : Math.floor(s / 60) + 'm ' + s % 60 + 's'; }

function visibleDone() {
  return S.jobs.filter(j => j.status === 'done' && j.images.length)
    .filter(j => S.filter !== 'fav' || j.fav)
    .filter(j => S.filter !== 'batch' || j.batch === (S.lastBatch || (S.jobs[0] && S.jobs[0].batch)))
    .filter(j => !S.modelFilter || j.model === S.modelFilter);
}

function renderGrid() {
  const done = visibleDone();
  const running = S.jobs.filter(j => j.status === 'running');
  const queued = S.jobs.filter(j => j.status === 'queued');
  const latest = S.jobs[0] && S.jobs[0].batch;
  const errors = S.jobs.filter(j => j.status === 'error' && j.batch === latest).slice(0, 4);
  S.selected = new Set([...S.selected].filter(id => done.some(j => j.id === id)));
  const sig = JSON.stringify([done.map(j => j.id + j.fav + S.selected.has(j.id)), running.map(j => [j.id,j.stage,j.step,j.progress,j.elapsed]), queued.length, errors.map(j => j.id), S.filter, S.modelFilter]);
  if (sig === S.sig) return; S.sig = sig;
  const tiles = [];
  running.forEach(j => {
    const pct = Math.round((j.progress || 0) * 100);
    const detail = j.stage === 'sampling' && j.steps ? `${j.step || 0} / ${j.steps} steps` : (j.stage || 'starting');
    const elapsed = j.elapsed || 0;
    tiles.push(`<div class="tile pending running" style="--ar:${(j.width || 1) / (j.height || 1)}"><div class="spin"></div><b>${esc(j.model_name)}</b><small>${esc(detail)}${j.stage === 'sampling' ? ' · ' + pct + '%' : ''}</small><div class="jobbar" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100"><i style="width:${pct}%"></i></div><small>Elapsed ${duration(elapsed)} · seed ${j.seed}</small></div>`);
  });
  if (queued.length) tiles.push(`<div class="tile pending"><b style="font-size:22px;color:var(--txt)">${queued.length}</b>waiting in queue</div>`);
  errors.forEach(j => tiles.push(`<div class="tile pending error" title="${esc(j.error)}">⚠ ${esc(j.model_name)} failed<br><small>${esc((j.error || '').slice(0, 140))}</small></div>`));
  done.forEach((j, i) => tiles.push(`<div class="tile" data-i="${i}" style="--ar:${(j.width || 1) / (j.height || 1)}">
      <img loading="lazy" src="${viewUrl(j.images[0])}" alt="">
      <span class="mb">${esc(j.model_name)}${j.fast ? ' ⚡' : ''}</span>
      <input class="pick" type="checkbox" data-id="${esc(j.id)}" ${S.selected.has(j.id) ? 'checked' : ''} aria-label="Select image">
      <button class="star ${j.fav ? 'on' : ''}" data-id="${j.id}">★</button>
      <button class="remove" data-id="${esc(j.id)}" title="Delete image" aria-label="Delete image">×</button>
      <div class="cap"><span>${esc(j.mode)}${j.sheet && j.mode === 'sheet' ? ' · ' + esc(j.sheet) : ''}</span><span>seed ${j.seed}</span></div></div>`));
  $('#grid').innerHTML = tiles.join('');
  $('#empty').classList.toggle('hidden', tiles.length > 0);
  $('#gcount').textContent = `${done.length} image${done.length === 1 ? '' : 's'}`;
  $$('#grid .tile[data-i]').forEach(t => t.onclick = (e) => { if (!e.target.classList.contains('star') && !e.target.classList.contains('pick') && !e.target.classList.contains('remove')) openLb(+t.dataset.i); });
  $$('#grid .pick').forEach(b => b.onchange = () => {
    if (b.checked) S.selected.add(b.dataset.id); else S.selected.delete(b.dataset.id);
    S.sig = ''; renderGrid();
  });
  $$('#grid .star').forEach(b => b.onclick = (e) => { e.stopPropagation(); toggleFav(b.dataset.id); });
  $$('#grid .remove').forEach(b => b.onclick = (e) => { e.stopPropagation(); deleteOne(b.dataset.id); });
  $('#deleteSelected').disabled = !S.selected.size;
  $('#deleteSelected').textContent = S.selected.size ? `Delete selected (${S.selected.size})` : 'Delete selected';
}

async function toggleFav(id) {
  const j = S.jobs.find(x => x.id === id); if (!j) return;
  j.fav = !j.fav; S.sig = ''; renderGrid(); if (S.lb >= 0) renderLb();
  await api('/forge/api/fav', { project: S.project, id, fav: j.fav });
}

async function deleteOne(id) {
  try {
    await api('/forge/api/delete', { project: S.project, ids: [id] });
    S.selected.delete(id); S.sig = '';
    pollJobs();
  } catch (e) { $('#queueMsg').className = 'msg err'; $('#queueMsg').textContent = e.message; }
}

function selectVisible() {
  const ids = visibleDone().map(j => j.id);
  const allSelected = ids.length && ids.every(id => S.selected.has(id));
  ids.forEach(id => allSelected ? S.selected.delete(id) : S.selected.add(id));
  S.sig = ''; renderGrid();
  $('#selectVisible').textContent = allSelected ? 'Select visible' : 'Clear visible';
}

async function deleteImages(all = false) {
  const ids = [...S.selected];
  if (!all && !ids.length) return;
  const count = all ? S.jobs.filter(j => j.status === 'done' && j.images.length).length : ids.length;
  const phrase = all ? 'all generated images in this project' : `${count} selected image${count === 1 ? '' : 's'}`;
  if (!window.confirm(`Delete ${phrase}? This removes the image files and gallery entries.`)) return;
  try {
    const result = await api('/forge/api/delete', all ? { project: S.project, all: true } : { project: S.project, ids });
    S.selected.clear(); S.sig = '';
    $('#queueMsg').className = 'msg ok';
    $('#queueMsg').textContent = `Deleted ${result.deleted} image${result.deleted === 1 ? '' : 's'}.` + (result.skipped ? ` ${result.skipped} active job${result.skipped === 1 ? '' : 's'} kept.` : '');
    pollJobs();
  } catch (e) { $('#queueMsg').className = 'msg err'; $('#queueMsg').textContent = e.message; }
}

// ------------------------------------------------------------------ lightbox
let lbList = [];
function openLb(i) { lbList = visibleDone(); S.lb = i; $('#lightbox').classList.remove('hidden'); renderLb(); }
function closeLb() { S.lb = -1; $('#lightbox').classList.add('hidden'); }
function renderLb() {
  const j = lbList[S.lb]; if (!j) return closeLb();
  const img = j.images[0];
  $('#lbImg').src = viewUrl(img);
  $('#lbDl').href = viewUrl(img); $('#lbDl').setAttribute('download', img.filename);
  $('#lbTitle').textContent = `${j.model_name}${j.fast ? ' ⚡ fast' : ''}`;
  $('#lbMeta').innerHTML = `${esc(j.mode)}${j.sheet && j.mode === 'sheet' ? ' · ' + esc(j.sheet) : ''}<br>seed ${j.seed} · ${j.steps} steps · ${esc(j.sampler || 'default sampler')} · ${esc(j.guidance_name || 'CFG')} ${j.cfg}<br>${j.width ? j.width + '×' + j.height : 'ref size'}<br>
     ${new Date(j.created * 1000).toLocaleString()}<br><span title="${esc(img.subfolder + '/' + img.filename)}">${esc(img.filename)}</span><br>${S.lb + 1} / ${lbList.length}`;
  $('#lbPrompt').textContent = j.prompt;
  $('#lbFav').classList.toggle('on', j.fav);
}
function lbUseRef() {
  const j = lbList[S.lb]; const img = j.images[0];
  if (S.refs.length >= selectedMaxRefs()) S.refs.shift();
  S.refs.push({ name: `${img.subfolder}/${img.filename} [output]`, url: viewUrl(img) });
  renderRefs(); updateQueueBtn();
  $('#queueMsg').className = 'msg ok'; $('#queueMsg').textContent = `Using ${img.filename} as the character reference.`;
}
function lbLoad() {
  const j = lbList[S.lb], r = j.req || {};
  if (r.mode) setMode(r.mode);
  if (r.sheet) { S.sheet = r.sheet; renderSheets(); }
  if (r.bg) setBg(r.bg);
  $('#notes').value = r.notes || ''; $('#scene').value = r.scene || ''; $('#raw').value = r.raw || '';
  $('#negative').value = r.negative || j.negative_template || '';
  if (r.size) $('#size').value = r.size;
  $('#fast').checked = !!r.fast;
  $('#steps').value = j.steps || ''; $('#cfg').value = j.cfg || '';
  $('#w').value = j.width || ''; $('#h').value = j.height || '';
  $('#referenceStrength').value = j.reference_strength ?? 0.65;
  S.models = [j.model]; store.set('models', S.models); renderModelList(); renderRefs();
  closeLb();
}
function lbMore() {
  const j = lbList[S.lb];
  const r = Object.assign({}, j.req, { project: S.project, models: [j.model], count: 4, seed_mode: 'random' });
  queue(r); closeLb();
}

// ------------------------------------------------------------------ wiring
function setMode(m) {
  S.mode = m; store.set('mode', m);
  $$('#modeSeg button').forEach(b => b.classList.toggle('on', b.dataset.mode === m));
  $$('.mode').forEach(d => d.classList.toggle('hidden', d.dataset.mode !== m));
  if (m === 'scene' && $('#size').value === 'sheet-wide') $('#size').value = 'landscape';
  if (m === 'sheet') { const sh = S.cfg && S.cfg.presets.sheets.find(x => x.id === S.sheet); if (sh) $('#size').value = sh.size; }
}
function setBg(bg) { S.bg = bg; store.set('bg', bg); $$('#bgSeg button').forEach(b => b.classList.toggle('on', b.dataset.bg === bg)); }

function setupCollapsibleSteps() {
  const collapsed = store.get('collapsedSteps', {});
  $$('.panel section').forEach((section, index) => {
    const heading = $('h3', section);
    if (!heading) return;
    const key = String(index + 1);
    if (collapsed[key]) section.classList.add('collapsed');
    heading.setAttribute('role', 'button');
    heading.setAttribute('tabindex', '0');
    heading.setAttribute('aria-expanded', String(!section.classList.contains('collapsed')));
    const toggle = () => {
      const isCollapsed = section.classList.toggle('collapsed');
      heading.setAttribute('aria-expanded', String(!isCollapsed));
      const next = store.get('collapsedSteps', {}); next[key] = isCollapsed; store.set('collapsedSteps', next);
    };
    heading.onclick = toggle;
    heading.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } };
  });
}

function wire() {
  $$('#pageTabs button').forEach(b => b.onclick = () => {
    $$('#pageTabs button').forEach(x => x.classList.toggle('on', x === b));
    $$('.page').forEach(p => p.classList.toggle('hidden', p.id !== b.dataset.page));
    if (b.dataset.page === 'models') pollConfig(); else loadConfig();
  });
  $$('#modeSeg button').forEach(b => b.onclick = () => setMode(b.dataset.mode));
  $$('#bgSeg button').forEach(b => b.onclick = () => setBg(b.dataset.bg));
  $$('#countChips button').forEach(b => b.onclick = () => { $('#count').value = b.textContent; updateQueueBtn(); });
  $('#count').oninput = updateQueueBtn;
  $('#size').onchange = () => store.set('size', $('#size').value);
  $('#queueBtn').onclick = () => queue();
  $('#project').onchange = () => { S.project = $('#project').value; S.sig = ''; api('/forge/api/settings', { last_project: S.project }); pollJobs(); };
  $('#newProject').onclick = async () => {
    const name = prompt('New project name'); if (!name) return;
    const r = await api('/forge/api/project', { name }); S.project = r.name; await loadConfig(); S.sig = ''; pollJobs();
  };
  $('#openFolder').onclick = () => api('/forge/api/open_folder', { project: S.project });
  $('#stopAll').onclick = async () => { if (confirmStop()) { await api('/forge/api/stop', {}); pollJobs(); } };
  $$('#filterSeg button').forEach(b => b.onclick = () => { S.filter = b.dataset.f; $$('#filterSeg button').forEach(x => x.classList.toggle('on', x === b)); renderGrid(); });
  $('#modelFilter').onchange = () => { S.modelFilter = $('#modelFilter').value; renderGrid(); };
  const th = store.get('thumb', 260); $('#thumb').value = th; $('#grid').style.setProperty('--thumb', th + 'px');
  $('#thumb').oninput = () => { $('#grid').style.setProperty('--thumb', $('#thumb').value + 'px'); store.set('thumb', +$('#thumb').value); };
  $('#selectVisible').onclick = selectVisible;
  $('#deleteSelected').onclick = () => deleteImages(false);
  $('#deleteAll').onclick = () => deleteImages(true);
  $('#saveToken').onclick = async () => { await api('/forge/api/settings', { hf_token: $('#token').value }); $('#token').value = ''; pollConfig(); };
  $('#unloadModel').onclick = async () => {
    try { const r = await api('/forge/api/unload', {}); $('#queueMsg').className = 'msg ok'; $('#queueMsg').textContent = `Model unloaded${r.freed_cuda_mb ? ` · freed ${r.freed_cuda_mb} MB CUDA memory` : ''}.`; await loadConfig(); }
    catch (e) { $('#queueMsg').className = 'msg err'; $('#queueMsg').textContent = e.message; }
  };
  $('#lbClose').onclick = closeLb;
  $('#lbImg').onclick = closeLb;
  $('#lightbox').onclick = e => { if (e.target === e.currentTarget) closeLb(); };
  $('#lbPrev').onclick = () => { S.lb = (S.lb - 1 + lbList.length) % lbList.length; renderLb(); };
  $('#lbNext').onclick = () => { S.lb = (S.lb + 1) % lbList.length; renderLb(); };
  $('#lbFav').onclick = () => toggleFav(lbList[S.lb].id);
  $('#lbRef').onclick = lbUseRef; $('#lbLoad').onclick = lbLoad; $('#lbMore').onclick = lbMore;
  document.addEventListener('keydown', e => {
    if (S.lb < 0 || ['TEXTAREA', 'INPUT'].includes(document.activeElement.tagName)) return;
    if (e.key === 'Escape') closeLb();
    else if (e.key === 'ArrowLeft') $('#lbPrev').click();
    else if (e.key === 'ArrowRight') $('#lbNext').click();
    else if (e.key.toLowerCase() === 'f') $('#lbFav').click();
    else if (e.key.toLowerCase() === 'r') lbUseRef();
  });
}
// A tiny in-page confirm (browser confirm() dialogs are avoided on purpose)
let stopArmed = 0;
function confirmStop() {
  const b = $('#stopAll');
  if (Date.now() - stopArmed < 3000) { b.textContent = 'Stop all'; stopArmed = 0; return true; }
  stopArmed = Date.now(); b.textContent = 'Click again to stop';
  setTimeout(() => { if (stopArmed) { b.textContent = 'Stop all'; stopArmed = 0; } }, 3000);
  return false;
}

(async function init() {
  wire(); setupDrop(); setupCollapsibleSteps();
  S.refs = S.refs.slice(0, 3);
  await loadConfig(); updateSamplingHint();
  setMode(S.mode); setBg(S.bg); renderRefs(); updateQueueBtn();
  if (!S.cfg.models.some(m => m.installed)) $$('#pageTabs button')[1].click();
  pollJobs();
})();

'use strict';

let grid = [];
let dragSrc = null;
let dragEnterCount = 0;
let _rafPending = false;
let _lastDragX = 0, _lastDragY = 0;
let _cellDragoverRafPending = false;
let _pendingCellDragover = null;

const appBody       = document.getElementById('app-body');
const dropZoneEl    = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const gridOuter     = document.getElementById('grid-outer');
const gridEl        = document.getElementById('grid');
const addLeftBtn    = document.getElementById('add-left-btn');
const addRightBtn   = document.getElementById('add-right-btn');
const addTopBtn     = document.getElementById('add-top-btn');
const addBottomBtn  = document.getElementById('add-bottom-btn');
const addImagesBtn  = document.getElementById('add-images-btn');
const optionsEl     = document.getElementById('options');
const statusEl      = document.getElementById('status');
const processingEl  = document.getElementById('processing');
const previewImg    = document.getElementById('preview-img');
const previewLive   = document.getElementById('preview-live');
const downloadBtn   = document.getElementById('download-btn');
const overlapCtrl   = document.getElementById('overlap-control');
const overlapSlider = document.getElementById('overlap-slider');
const overlapValEl  = document.getElementById('overlap-val');
const overlapAutoBtn    = document.getElementById('overlap-auto-btn');
const overlapDecBtn     = document.getElementById('overlap-dec-btn');
const overlapIncBtn     = document.getElementById('overlap-inc-btn');
const overlapFadeToggle = document.getElementById('overlap-fade-toggle');
const fitToggle         = document.getElementById('fit-toggle');
const zoomToggleBtn     = document.getElementById('zoom-toggle-btn');
const zoomPanel         = document.getElementById('zoom-panel');
const zoomCanvas        = document.getElementById('zoom-canvas');

let _zoomActive     = false;
let _zoomLevel      = 3;
let _zoomOffsetX    = null;   // null = re-center on next render
let _zoomOffsetY    = null;
let _lastZoomParams = null;

function _setOverlapPx(px) {
  overlapSlider.value = px;
  overlapValEl.textContent = px + ' px';
}

overlapSlider.addEventListener('input', () => {
  overlapValEl.textContent = overlapSlider.value + ' px';
  onLayoutChanged();
});

// Auto is a one-shot detection, not a mode: the server suggests a value,
// which then lands on the slider like any manual setting. (A sticky "auto"
// would re-detect on Download and could save something the preview never
// showed.)
overlapAutoBtn.addEventListener('click', async () => {
  if (imgCount() < 2) return;
  _statusError = null;
  overlapAutoBtn.disabled = true;
  overlapAutoBtn.textContent = 'Detecting…';
  try {
    const fd = _buildFormData();
    fd.set('overlap', 'auto');
    const res = await fetch('/preview', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text().catch(() => res.statusText));
    const px = parseInt(res.headers.get('X-Overlap-Px'), 10);
    if (!Number.isNaN(px)) _setOverlapPx(px);
    onLayoutChanged();
  } catch (err) {
    showError('Overlap detection failed: ' + err.message);
  } finally {
    overlapAutoBtn.disabled = false;
    overlapAutoBtn.textContent = 'Auto';
  }
});

fitToggle.addEventListener('change', () => onLayoutChanged());

function _stepOverlap(delta) {
  const v = Math.min(parseInt(overlapSlider.max), Math.max(0, parseInt(overlapSlider.value) + delta));
  _setOverlapPx(v);
  onLayoutChanged();
}
overlapDecBtn.addEventListener('click', () => _stepOverlap(-1));
overlapIncBtn.addEventListener('click', () => _stepOverlap( 1));

overlapFadeToggle.addEventListener('change', () => onLayoutChanged());

zoomToggleBtn.addEventListener('click', () => {
  _zoomActive  = !_zoomActive;
  _zoomOffsetX = null;
  _zoomOffsetY = null;
  zoomToggleBtn.classList.toggle('active', _zoomActive);
  zoomToggleBtn.setAttribute('aria-pressed', _zoomActive);
  if (!_zoomActive) zoomPanel.style.display = 'none';
  scheduleLivePreview();
});

document.querySelectorAll('.zoom-level-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const newLevel = parseInt(btn.dataset.zoom);
    if (_zoomOffsetX !== null) {
      // Keep the same viewport center when changing magnification
      const zc  = zoomCanvas;
      const cxSrc = _zoomOffsetX + Math.ceil(zc.width  / _zoomLevel) / 2;
      const cySrc = _zoomOffsetY + Math.ceil(zc.height / _zoomLevel) / 2;
      _zoomLevel   = newLevel;
      _zoomOffsetX = Math.round(cxSrc - Math.ceil(zc.width  / _zoomLevel) / 2);
      _zoomOffsetY = Math.round(cySrc - Math.ceil(zc.height / _zoomLevel) / 2);
    } else {
      _zoomLevel = newLevel;
    }
    document.querySelectorAll('.zoom-level-btn').forEach(b => b.classList.toggle('active', b === btn));
    if (_lastZoomParams) _renderZoomStrip(_lastZoomParams);
  });
});

let _zoomDragX = null, _zoomDragY = null;
let _zoomMoveRafPending = false;
let _lastZoomMoveX = 0, _lastZoomMoveY = 0;
zoomCanvas.addEventListener('mousedown', e => {
  _zoomDragX = e.clientX;
  _zoomDragY = e.clientY;
  zoomCanvas.style.cursor = 'grabbing';
  e.preventDefault();
});
window.addEventListener('mousemove', e => {
  if (_zoomDragX === null) return;
  _lastZoomMoveX = e.clientX;
  _lastZoomMoveY = e.clientY;
  if (_zoomMoveRafPending) return;
  _zoomMoveRafPending = true;
  requestAnimationFrame(() => {
    _zoomMoveRafPending = false;
    if (_zoomDragX === null) return;
    const dx = _lastZoomMoveX - _zoomDragX;
    const dy = _lastZoomMoveY - _zoomDragY;
    _zoomDragX = _lastZoomMoveX;
    _zoomDragY = _lastZoomMoveY;
    if (_zoomOffsetX !== null) {
      _zoomOffsetX = Math.round(_zoomOffsetX - dx / _zoomLevel);
      _zoomOffsetY = Math.round(_zoomOffsetY - dy / _zoomLevel);
    }
    if (_lastZoomParams) _renderZoomStrip(_lastZoomParams);
  });
});
window.addEventListener('mouseup', () => {
  if (_zoomDragX !== null) { _zoomDragX = null; _zoomDragY = null; zoomCanvas.style.cursor = 'grab'; }
});

const uid = () => Math.random().toString(36).slice(2, 9);

// Thumbnails are blob: URLs rather than base64 data: URLs — toBlob encodes
// off the main thread and the DOM doesn't carry ~100 KB attribute strings.
// Each cell owns its URL; _disposeCell revokes it.
async function _thumbFromSource(src, maxPx) {
  const scale = Math.min(1, maxPx / Math.max(src.width, src.height));
  const w = Math.max(1, Math.round(src.width * scale));
  const h = Math.max(1, Math.round(src.height * scale));
  const cv = Object.assign(document.createElement('canvas'), { width: w, height: h });
  cv.getContext('2d').drawImage(src, 0, 0, w, h);
  const blob = await new Promise(res => cv.toBlob(res, 'image/jpeg', 0.75));
  return URL.createObjectURL(blob);
}


// Releases everything a cell owns (thumbnail URL, cached preview bitmap).
// Call whenever a cell object is dropped from the grid or replaced.
function _disposeCell(cell) {
  if (!cell) return;
  cell.disposed = true;
  _evictBitmap(cell.id);
  if (cell.dataURL && cell.dataURL.startsWith('blob:')) URL.revokeObjectURL(cell.dataURL);
}

// A cell's rotate/flip, applied on top of the original file: mirror
// left-right if `flip`, then `rot` quarter turns clockwise. The file itself
// is never re-encoded; the server applies the same steps to the upload
// (_apply_orientation), so the output keeps the source's encoding.
const NO_TURN = Object.freeze({ rot: 0, flip: false });

function _turn({ rot, flip }, type) {
  if (type === 'rotate-cw')  return { rot: (rot + 1) % 4, flip };
  if (type === 'rotate-ccw') return { rot: (rot + 3) % 4, flip };
  // Mirroring after a turn equals mirroring first and turning the other way.
  if (type === 'flip-h') return { rot: (4 - rot) % 4, flip: !flip };
  if (type === 'flip-v') return { rot: (6 - rot) % 4, flip: !flip };  // flip-h, then 180°
  return { rot, flip };
}

// Decodes a cell's file, upright and with its rotate/flip applied.
async function _decodeCell(cell) {
  const bmp = await createImageBitmap(cell.file, { imageOrientation: 'from-image' });
  const { rot, flip } = cell.orient;
  if (!rot && !flip) return bmp;
  try {
    const turned = rot % 2 === 1;
    const cv = Object.assign(document.createElement('canvas'), {
      width: turned ? bmp.height : bmp.width, height: turned ? bmp.width : bmp.height,
    });
    const ctx = cv.getContext('2d');
    ctx.translate(cv.width / 2, cv.height / 2);
    ctx.rotate(rot * Math.PI / 2);
    if (flip) ctx.scale(-1, 1);
    ctx.drawImage(bmp, -bmp.width / 2, -bmp.height / 2);
    return await createImageBitmap(cv);
  } finally { bmp.close(); }
}

// One full-resolution decode yields both the grid thumbnail and the
// (pre-warmed) live-preview bitmap, so the first slider drag never stalls
// on a decode. The preview entry is cached at once, so a live preview
// that runs meanwhile shares this decode. Resolves to the thumbnail URL.
function _decodeInto(cell) {
  let thumbP;
  const entry = _decodeCell(cell).then(full => {
    // Both start their synchronous draws from `full` before it is closed.
    thumbP = _thumbFromSource(full, 400);
    return _previewEntryFrom(full);
  });
  _bitmapCache[cell.id] = entry;
  entry.catch(() => { if (_bitmapCache[cell.id] === entry) delete _bitmapCache[cell.id]; });
  return entry.then(() => thumbP);
}

const nRows     = () => grid.length;
const nCols     = () => (grid.length ? grid[0].length : 0);
const hasImages = () => grid.some(row => row.some(Boolean));
const imgCount  = () => grid.reduce((n, r) => n + r.filter(Boolean).length, 0);

function isFileDrag(e) {
  return e.dataTransfer && [...e.dataTransfer.types].includes('Files');
}

function padGrid() {
  const cols = Math.max(...grid.map(r => r.length), 0);
  for (const row of grid) while (row.length < cols) row.push(null);
}

function insertCol(after) {
  for (const row of grid) row.splice(after + 1, 0, null);
}

function insertRow(after) {
  grid.splice(after + 1, 0, Array(nCols()).fill(null));
}

// One level of undo for Remove. The removed cell isn't disposed until
// the undo window closes, so restoring it needs no re-decode.
let _undo = null;   // { grid, cell, timer }
const UNDO_MS = 8000;
const toastEl  = document.getElementById('toast');
const toastMsg = document.getElementById('toast-msg');

function _finalizeUndo() {
  if (!_undo) return;
  clearTimeout(_undo.timer);
  _disposeCell(_undo.cell);
  _undo = null;
  toastEl.classList.remove('visible');
}

function removeCell(r, c) {
  _finalizeUndo();
  const cell = grid[r][c];
  if (!cell) return;
  _undo = { grid: grid.map(row => row.slice()), cell, timer: setTimeout(_finalizeUndo, UNDO_MS) };
  toastMsg.textContent = `Removed ${cell.name}`;
  toastEl.classList.add('visible');
  grid[r][c] = null;
  compact();
}

function undoRemove() {
  if (!_undo) return;
  clearTimeout(_undo.timer);
  grid = _undo.grid;
  _undo = null;
  toastEl.classList.remove('visible');
  render();
}
document.getElementById('toast-undo').addEventListener('click', undoRemove);

function compact() {
  grid = grid.filter(row => row.some(Boolean));
  if (!grid.length) return;
  padGrid();
  const cols = nCols();
  const dead = Array.from({length: cols}, (_, i) => i)
    .filter(c => grid.every(row => !row[c]));
  if (dead.length) {
    const keep = new Set(Array.from({length: cols}, (_, i) => i).filter(i => !dead.includes(i)));
    for (let r = 0; r < grid.length; r++) {
      grid[r] = grid[r].filter((_, i) => keep.has(i));
    }
  }
}

function swapCells(r1, c1, r2, c2) {
  [grid[r1][c1], grid[r2][c2]] = [grid[r2][c2], grid[r1][c1]];
}

// Returns true when the grid's row/col shape is unchanged (a plain swap),
// so the caller can patch just the two affected cells instead of a full
// render(); returns false when rows/cols were inserted or removed and a
// full render() is required.
function moveCellToZone(srcR, srcC, dstR, dstC, zone) {
  if (zone === 'center') { swapCells(srcR, srcC, dstR, dstC); return true; }
  const srcCell = grid[srcR][srcC];
  const dstCell = grid[dstR][dstC];
  grid[srcR][srcC] = null;
  compact();
  // Re-locate dstCell after compact may have shifted indices
  let nr = -1, nc = -1;
  outer: for (let r = 0; r < grid.length; r++)
    for (let c = 0; c < grid[r].length; c++)
      if (grid[r][c] === dstCell) { nr = r; nc = c; break outer; }
  if (nr === -1) { grid.push([srcCell]); return false; }
  if (zone === 'left')   { insertCol(nc - 1); grid[nr][nc]     = srcCell; }
  else if (zone === 'right')  { insertCol(nc);     grid[nr][nc + 1] = srcCell; }
  else if (zone === 'top')    { insertRow(nr - 1); grid[nr][nc]     = srcCell; }
  else if (zone === 'bottom') { insertRow(nr);     grid[nr + 1][nc] = srcCell; }
  return false;
}

// Updates the grid at once, so quick repeated presses all count; the new
// thumbnail follows when its decode finishes.
function applyTransform(r, c, type) {
  _finalizeUndo();  // undo snapshots hold cell objects this would replace
  const cell = grid[r][c];
  if (!cell) return;
  // A new id makes the request signature and the preview cache see the
  // change. `next` shows the old thumbnail until its own is ready.
  const next = { ...cell, id: uid(), orient: _turn(cell.orient, type) };
  cell.retired = true;
  _evictBitmap(cell.id);
  grid[r][c] = next;
  onLayoutChanged();
  _decodeInto(next).then(url => {
    // Superseded by a later turn, or removed: this thumbnail isn't needed.
    if (next.retired || next.disposed) { URL.revokeObjectURL(url); return; }
    URL.revokeObjectURL(next.dataURL);
    next.dataURL = url;
    const pos = _posOf(next);
    if (pos) updateCellInPlace(pos[0], pos[1]);
  }, err => {
    console.error(err);
    if (!next.retired && !next.disposed) showError(`Couldn’t turn ${cell.name}`);
  });
}

function _posOf(cell) {
  for (let r = 0; r < grid.length; r++) {
    const c = grid[r].indexOf(cell);
    if (c !== -1) return [r, c];
  }
  return null;
}

// Patches a single cell's thumbnail <img> in place instead of tearing down
// and rebuilding the whole grid (which forces the browser to re-layout and
// re-decode every other cell's thumbnail too). Only valid when the cell's
// position and the grid's shape haven't changed — falls back to a full
// render() if the expected element isn't found.
function updateCellInPlace(r, c) {
  const cell = grid[r][c];
  if (!cell) { render(); return; }
  const img = gridEl.querySelector(`.cell[data-r="${r}"][data-c="${c}"] img`);
  if (!img) { render(); return; }
  img.src = cell.dataURL;
  img.alt = cell.name;
}

// Decodes the image files into cells, with a loading message while it
// runs. Unreadable files are skipped and reported rather than failing the
// whole batch.
async function makeCells(files) {
  _finalizeUndo();  // undo would silently drop the images being added
  _statusError = null;
  const all = [...files];
  const imgs = all.filter(f => f.type.startsWith('image/'));
  if (!imgs.length) {
    if (all.length) showError('Those files aren’t images. Use JPEG, PNG or WebP.');
    return [];
  }
  statusEl.classList.remove('error');
  statusEl.textContent = `Loading ${imgs.length} image${imgs.length !== 1 ? 's' : ''}…`;
  document.body.classList.add('loading');
  try {
    const results = await Promise.allSettled(imgs.map(_makeCell));
    const bad = imgs.filter((_, i) => results[i].status === 'rejected').map(f => f.name);
    const skipped = all.length - imgs.length;
    if (bad.length) showError(`Couldn’t read ${bad.join(', ')}`);
    else if (skipped) showError(`Skipped ${skipped} file${skipped !== 1 ? 's' : ''} that aren’t images`);
    else updateStatus();
    return results.filter(r => r.status === 'fulfilled').map(r => r.value);
  } finally {
    document.body.classList.remove('loading');
  }
}

async function _makeCell(f) {
  const cell = { id: uid(), name: f.name, file: f, orient: NO_TURN };
  try {
    cell.dataURL = await _decodeInto(cell);
  } catch (err) {
    _evictBitmap(cell.id);
    throw err;
  }
  return cell;
}

// Adds images after the existing ones: down a single column, otherwise
// along the first row.
async function appendFiles(files) {
  if (!hasImages()) return initFromFiles(files);
  const cells = await makeCells(files);
  if (!cells.length) return;
  if (nCols() === 1 && nRows() > 1) for (const c of cells) grid.push([c]);
  else for (const c of cells) { insertCol(nCols() - 1); grid[0][nCols() - 1] = c; }
  render();
}

async function initFromFiles(files) {
  const cells = await makeCells(files);
  if (!cells.length) return;
  _finalizeUndo();  // the old grid is going away; nothing to undo into
  for (const row of grid) for (const old of row) _disposeCell(old);
  grid = [cells];
  render();
}

async function dropFilesOnCell(r, c, zone, files) {
  const cells = await makeCells(files);
  if (!cells.length) return;
  const first = cells[0];
  if (zone === 'center') {
    _disposeCell(grid[r][c]);
    grid[r][c] = first;
    for (let i = 1; i < cells.length; i++) { insertCol(c + i - 1); grid[r][c + i] = cells[i]; }
  } else if (zone === 'right')  { insertCol(c);     grid[r][c + 1] = first; }
  else if   (zone === 'left')   { insertCol(c - 1); grid[r][c]     = first; }
  else if   (zone === 'bottom') { insertRow(r);     grid[r + 1][c] = first; }
  else if   (zone === 'top')    { insertRow(r - 1); grid[r][c]     = first; }
  render();
}

function render() {
  if (!hasImages()) {
    gridOuter.style.display = 'none';
    dropZoneEl.style.display = 'flex';
    overlapCtrl.style.display = 'none';
    addImagesBtn.style.display = 'none';
    optionsEl.style.display = 'none';
    onLayoutChanged();
    return;
  }
  dropZoneEl.style.display = 'none';
  gridOuter.style.display = 'flex';
  const rows = nRows(), cols = nCols();
  gridEl.style.gridTemplateColumns = `repeat(${cols}, auto)`;
  gridEl.style.gridTemplateRows    = `repeat(${rows}, auto)`;
  gridEl.innerHTML = '';
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = grid[r][c];
      gridEl.appendChild(cell ? buildCell(cell, r, c) : buildEmptyCell(r, c));
    }
  }
  const n = imgCount();
  addImagesBtn.style.display = '';
  optionsEl.style.display = n >= 2 ? 'flex' : 'none';
  // Overlap only applies to single-row or single-column grids — _stitch_grid
  // silently zeroes it for any other shape, so don't show a control that
  // would have no effect.
  const overlapApplies = rows === 1 || cols === 1;
  overlapCtrl.style.display = (n >= 2 && overlapApplies) ? 'flex' : 'none';
  // The seam zoom only draws for side-by-side layouts.
  const zoomApplies = rows === 1;
  zoomToggleBtn.style.display = zoomApplies ? '' : 'none';
  if (!zoomApplies && _zoomActive) {
    _zoomActive = false;
    zoomToggleBtn.classList.remove('active');
    zoomToggleBtn.setAttribute('aria-pressed', 'false');
    zoomPanel.style.display = 'none';
  }
  onLayoutChanged();
}

// One status line, rebuilt from current state rather than overwritten by
// whichever code path ran last.
let _statusError = null;

// Errors stay in the status line (in bold) until the next action starts.
function showError(msg) {
  _statusError = msg;
  updateStatus();
}

function updateStatus() {
  statusEl.classList.toggle('error', !!_statusError);
  if (_statusError) { statusEl.textContent = _statusError; return; }
  if (!hasImages()) { statusEl.textContent = 'Drop images to start'; return; }
  const n = imgCount();
  statusEl.textContent = n === 1
    ? '1 image · add another to stitch'
    : `${n} images · ${nRows()}×${nCols()} grid`;
}

const previewPane    = document.getElementById('preview-pane');
const previewCaption = document.getElementById('preview-caption');

// 'live'  — fast client-side composite (downscaled, not the final file)
// 'final' — the server's real output for the current layout
// 'pending' — an old result is shown while the new one renders
function _setPreviewState(state, text) {
  previewPane.dataset.state = state;
  previewCaption.textContent = text || {
    live:    'Quick preview (reduced size) · Download saves full quality',
    pending: 'Rendering…',
  }[state] || '';
}

const _liveSupported = () => (nRows() === 1 || nCols() === 1) && imgCount() >= 2;

// Called after anything that changes what the output would be: layout,
// rotation/flip, Fit, overlap. Keeps the preview pane from showing a
// result that Download would no longer produce.
function onLayoutChanged() {
  updateStatus();
  if (imgCount() < 2) {
    _previewGen++;  // drop any in-flight live paint
    appBody.classList.remove('has-preview', 'layout-side', 'layout-stack');
    return;
  }
  if (_liveSupported()) scheduleLivePreview();
  else if (appBody.classList.contains('has-preview')) {
    // Back to the layout that was last stitched: the shown result is current.
    if (_matchesLastServerResult(_requestSignature())) {
      _setPreviewState('final', _lastServerResult.caption || ' ');
      return;
    }
    _setPreviewState('pending');
  }
  // Grids with no client-side live preview get the server render instead,
  // debounced so a burst of edits costs one request.
  if (!_liveSupported()) scheduleServerPreview();
}

function buildCell(cell, r, c) {
  const el = document.createElement('div');
  el.className = 'cell';
  el.draggable = true;
  el.tabIndex = 0;
  el.setAttribute('aria-label',
    `${cell.name}, row ${r + 1}, column ${c + 1}. R to rotate, H or V to flip, Delete to remove.`);
  el.addEventListener('keydown', e => {
    if (e.target !== el || e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
    const key = e.key.toLowerCase();
    const transform = key === 'r' ? (e.shiftKey ? 'rotate-ccw' : 'rotate-cw')
                    : key === 'h' ? 'flip-h' : key === 'v' ? 'flip-v' : null;
    if (transform) { e.preventDefault(); applyTransform(r, c, transform); return; }
    if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      removeCell(r, c);
      render();
      // Keep keyboard focus in the grid.
      const next = gridEl.querySelector('.cell:not(.empty)');
      if (next) next.focus();
    }
  });
  el.dataset.r = r;
  el.dataset.c = c;

  const img = document.createElement('img');
  img.src = cell.dataURL;
  img.alt = cell.name;
  el.appendChild(img);

  const overlay = document.createElement('div');
  overlay.className = 'drop-overlay';
  for (const z of ['center','left','right','top','bottom']) {
    const dz = document.createElement('div');
    dz.className = `dz dz-${z}`;
    overlay.appendChild(dz);
  }
  el.appendChild(overlay);

  const rmBtn = document.createElement('button');
  rmBtn.className = 'btn-remove';
  rmBtn.title = 'Remove (Delete)';
  rmBtn.setAttribute('aria-label', 'Remove');
  rmBtn.tabIndex = -1;
  rmBtn.textContent = '×';
  rmBtn.addEventListener('click', e => { e.stopPropagation(); removeCell(r, c); render(); });
  el.appendChild(rmBtn);

  const toolbar = document.createElement('div');
  toolbar.className = 'cell-toolbar';
  toolbar.innerHTML =
    '<button class="btn-transform" tabindex="-1" title="Rotate counter-clockwise (⇧R)" aria-label="Rotate counter-clockwise">↺</button>' +
    '<button class="btn-transform" tabindex="-1" title="Rotate clockwise (R)" aria-label="Rotate clockwise">↻</button>' +
    '<button class="btn-transform" tabindex="-1" title="Flip horizontal (H)" aria-label="Flip horizontal">↔</button>' +
    '<button class="btn-transform" tabindex="-1" title="Flip vertical (V)" aria-label="Flip vertical">↕</button>';
  const [rotateCCW, rotateCW, flipH, flipV] = toolbar.children;
  rotateCCW.addEventListener('click', e => { e.stopPropagation(); applyTransform(r, c, 'rotate-ccw'); });
  rotateCW .addEventListener('click', e => { e.stopPropagation(); applyTransform(r, c, 'rotate-cw'); });
  flipH    .addEventListener('click', e => { e.stopPropagation(); applyTransform(r, c, 'flip-h'); });
  flipV    .addEventListener('click', e => { e.stopPropagation(); applyTransform(r, c, 'flip-v'); });
  el.appendChild(toolbar);

  el.addEventListener('dragstart', e => {
    dragSrc = { r, c };
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', `${r},${c}`);
    requestAnimationFrame(() => el.classList.add('dragging'));
  });
  el.addEventListener('dragend', () => {
    el.classList.remove('dragging');
    dragSrc = null;
    clearAllHighlights();
  });
  el.addEventListener('dragover', e => {
    e.preventDefault();
    if (!dragSrc) return;
    _pendingCellDragover = { overlay, clientX: e.clientX, clientY: e.clientY, el };
    if (_cellDragoverRafPending) return;
    _cellDragoverRafPending = true;
    requestAnimationFrame(() => {
      _cellDragoverRafPending = false;
      if (!_pendingCellDragover) return;
      const p = _pendingCellDragover;
      _pendingCellDragover = null;
      activateZone(p.overlay, zoneAt(p.clientX, p.clientY, p.el));
    });
  });
  el.addEventListener('dragleave', e => {
    if (dragSrc && !el.contains(e.relatedTarget)) deactivateZone(overlay);
  });
  el.addEventListener('drop', e => {
    e.preventDefault();
    deactivateZone(overlay);
    if (e.dataTransfer.files.length) return;
    if (!dragSrc) return;
    const { r: sr, c: sc } = dragSrc;
    if (sr !== r || sc !== c) {
      _finalizeUndo();
      const sameShape = moveCellToZone(sr, sc, r, c, zoneAt(e.clientX, e.clientY, el));
      if (sameShape) { updateCellInPlace(sr, sc); updateCellInPlace(r, c); onLayoutChanged(); }
      else render();
    }
  });

  return el;
}

function buildEmptyCell(r, c) {
  const el = document.createElement('div');
  el.className = 'cell empty';
  el.dataset.r = r;
  el.dataset.c = c;
  el.textContent = '+';
  el.tabIndex = 0;
  el.setAttribute('role', 'button');
  el.setAttribute('aria-label', `Add image at row ${r + 1}, column ${c + 1}`);
  el.addEventListener('dragover', e => { e.preventDefault(); });
  el.addEventListener('click', () => openPicker(r, c));
  el.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openPicker(r, c); }
  });
  return el;
}

function zoneAt(clientX, clientY, el) {
  const rect = el.getBoundingClientRect();
  const nx = (clientX - (rect.left + rect.right)  / 2) / (rect.width  / 2);
  const ny = (clientY - (rect.top  + rect.bottom) / 2) / (rect.height / 2);
  // Small central region snaps to center (replace), everywhere else snaps to nearest side
  if (Math.abs(nx) < 0.33 && Math.abs(ny) < 0.33) return 'center';
  return Math.abs(nx) >= Math.abs(ny)
    ? (nx > 0 ? 'right' : 'left')
    : (ny > 0 ? 'bottom' : 'top');
}

function activateZone(overlay, zone) {
  for (const dz of overlay.querySelectorAll('.dz')) {
    dz.classList.toggle('active', dz.classList.contains(`dz-${zone}`));
  }
}
function deactivateZone(overlay) {
  overlay.querySelectorAll('.dz.active').forEach(d => d.classList.remove('active'));
}
function clearAllHighlights() {
  document.querySelectorAll('.dz.active').forEach(d => d.classList.remove('active'));
  document.querySelectorAll('.cell.empty.drag-over-empty').forEach(d => d.classList.remove('drag-over-empty'));
  dropZoneEl.classList.remove('drag-over');
}

function findNearestZone(clientX, clientY) {
  if (!hasImages()) return { type: 'empty' };
  const allCells = document.querySelectorAll('.cell[data-r][data-c]');
  if (!allCells.length) return { type: 'empty' };
  function distToRect(rect) {
    const dx = Math.max(0, rect.left - clientX, clientX - rect.right);
    const dy = Math.max(0, rect.top  - clientY, clientY - rect.bottom);
    return Math.hypot(dx, dy);
  }
  let nearest = null, minDist = Infinity;
  for (const el of allCells) {
    const rect = el.getBoundingClientRect();
    const d = distToRect(rect);
    if (d < minDist) { minDist = d; nearest = { el, rect }; }
  }
  const r = +nearest.el.dataset.r;
  const c = +nearest.el.dataset.c;
  if (nearest.el.classList.contains('empty')) return { type: 'cell', r, c, zone: 'center', el: nearest.el };
  return { type: 'cell', r, c, zone: zoneAt(clientX, clientY, nearest.el), el: nearest.el };
}

function updateDropHighlight(clientX, clientY) {
  clearAllHighlights();
  const target = findNearestZone(clientX, clientY);
  if (target.type === 'empty') { dropZoneEl.classList.add('drag-over'); return; }
  if (target.el.classList.contains('empty')) {
    target.el.classList.add('drag-over-empty');
  } else {
    const overlay = target.el.querySelector('.drop-overlay');
    if (overlay) activateZone(overlay, target.zone);
  }
}

document.addEventListener('dragenter', e => {
  if (!isFileDrag(e)) return;
  dragEnterCount++;
});
document.addEventListener('dragleave', e => {
  if (!isFileDrag(e)) return;
  dragEnterCount--;
  if (dragEnterCount <= 0) {
    dragEnterCount = 0;
    clearAllHighlights();
  }
});
document.addEventListener('dragover', e => {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  _lastDragX = e.clientX;
  _lastDragY = e.clientY;
  if (!_rafPending) {
    _rafPending = true;
    requestAnimationFrame(() => {
      updateDropHighlight(_lastDragX, _lastDragY);
      _rafPending = false;
    });
  }
});
document.addEventListener('drop', e => {
  if (!e.dataTransfer.files.length) return;
  e.preventDefault();
  dragEnterCount = 0;
  clearAllHighlights();
  const target = findNearestZone(e.clientX, e.clientY);
  if (target.type === 'empty') initFromFiles(e.dataTransfer.files);
  else dropFilesOnCell(target.r, target.c, target.zone, e.dataTransfer.files);
});
document.addEventListener('dragend', () => {
  dragEnterCount = 0;
  clearAllHighlights();
});

// Opens the file picker aimed at an empty cell, or (no target) at the end
// of the grid. The target is always reset, so cancelling a cell's picker
// can't redirect a later browse into that cell.
function openPicker(r, c) {
  if (r === undefined) { delete fileInput.dataset.targetR; delete fileInput.dataset.targetC; }
  else { fileInput.dataset.targetR = r; fileInput.dataset.targetC = c; }
  fileInput.click();
}

dropZoneEl.addEventListener('click', () => openPicker());
dropZoneEl.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openPicker(); } });
document.getElementById('add-images-btn').addEventListener('click', () => openPicker());
fileInput.addEventListener('change', e => {
  if (!e.target.files.length) { fileInput.value = ''; return; }
  const tr = fileInput.dataset.targetR;
  const tc = fileInput.dataset.targetC;
  if (tr !== undefined && tc !== undefined) {
    dropFilesOnCell(+tr, +tc, 'center', e.target.files);
    delete fileInput.dataset.targetR;
    delete fileInput.dataset.targetC;
  } else {
    appendFiles(e.target.files);
  }
  fileInput.value = '';
});

// Edge "+" buttons open a new row/column and go straight to the picker
// for its first slot; cancelling the picker removes the empty slot again.
function addAtEdge(insert, target) {
  if (!hasImages()) return;
  _finalizeUndo();
  insert();
  render();
  const [r, c] = target();
  openPicker(r, c);
}
addLeftBtn.addEventListener('click',   () => addAtEdge(() => insertCol(-1),          () => [0, 0]));
addRightBtn.addEventListener('click',  () => addAtEdge(() => insertCol(nCols() - 1), () => [0, nCols() - 1]));
addTopBtn.addEventListener('click',    () => addAtEdge(() => insertRow(-1),          () => [0, 0]));
addBottomBtn.addEventListener('click', () => addAtEdge(() => insertRow(nRows() - 1), () => [nRows() - 1, 0]));
fileInput.addEventListener('cancel', () => {
  if (fileInput.dataset.targetR === undefined) return;
  delete fileInput.dataset.targetR;
  delete fileInput.dataset.targetC;
  compact();
  render();
});

function _buildFormData() {
  const formData = new FormData();
  const layoutIds = grid.map(row => row.map(cell => cell ? cell.id : null));
  formData.append('layout', JSON.stringify(layoutIds));
  formData.append('fit', fitToggle.checked);
  formData.append('overlap', overlapSlider.value);
  formData.append('overlap_blend', overlapFadeToggle.checked ? 'fade' : 'hard');
  const seen = new Set();
  const transforms = {};
  for (const row of grid) {
    for (const cell of row) {
      if (cell && !seen.has(cell.id)) {
        formData.append(cell.id, cell.file, cell.name);
        if (cell.orient.rot || cell.orient.flip) transforms[cell.id] = cell.orient;
        seen.add(cell.id);
      }
    }
  }
  formData.append('transforms', JSON.stringify(transforms));
  return formData;
}

let _previewObjectURL = null;

// The last full-resolution result from the server, keyed on everything that
// feeds _buildFormData. Download reuses it instead of re-uploading every
// image and re-stitching an identical JPEG. (Never the client-side live
// composite — that one is downscaled.)
let _lastServerResult = null;  // { key, files, blob }

function _requestSignature() {
  const files = [];
  const layoutIds = grid.map(row => row.map(cell => {
    if (cell) files.push(cell.file);
    return cell ? cell.id : null;
  }));
  const key = JSON.stringify([
    layoutIds,
    fitToggle.checked,
    overlapSlider.value,
    overlapFadeToggle.checked,
  ]);
  return { key, files };
}

function _matchesLastServerResult(sig) {
  const last = _lastServerResult;
  return !!last && last.key === sig.key && last.files.length === sig.files.length
    && last.files.every((f, i) => f === sig.files[i]);
}

// ── Live client-side preview ──────────────────────────────────────────────────
const _bitmapCache = {};  // cell.id → ImageBitmap

function _renderZoomStrip({ cv, dw0, ov, cvW, cvH }) {
  const ZOOM   = _zoomLevel;
  const zc     = zoomCanvas;
  // Measure after showing: a hidden panel has no size. The CSS box sets
  // the size; the backing store just matches it so pixels stay square.
  zoomPanel.style.display = 'block';
  zc.width     = Math.max(1, zc.clientWidth);
  zc.height    = Math.max(1, zc.clientHeight);

  // How many source pixels fit in this viewport at the chosen zoom level
  const vpW = Math.ceil(zc.width  / ZOOM);
  const vpH = Math.ceil(zc.height / ZOOM);

  // On first render (or after toggle/reset), center the view on the seam
  const seamL = dw0 - ov;
  if (_zoomOffsetX === null) {
    _zoomOffsetX = Math.round(seamL + ov / 2 - vpW / 2);
    _zoomOffsetY = Math.round((cvH - vpH) / 2);
  }

  const srcX = Math.max(0, Math.min(cvW - vpW, _zoomOffsetX));
  const srcY = Math.max(0, Math.min(cvH - vpH, _zoomOffsetY));

  const zctx = zc.getContext('2d');
  zctx.clearRect(0, 0, zc.width, zc.height);
  zctx.imageSmoothingEnabled = false;
  zctx.drawImage(cv, srcX, srcY, vpW, vpH, 0, 0, zc.width, zc.height);

  // Mark overlap left/right boundaries
  // Accent line on a white under-stroke so it reads on dark and light photos.
  const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#0066cc';
  for (const natX of [seamL, dw0]) {
    const mx = (natX - srcX) * ZOOM;
    if (mx >= 0 && mx <= zc.width) {
      for (const [style, w] of [['#ffffff', 4], [accent, 2]]) {
        zctx.strokeStyle = style; zctx.lineWidth = w;
        zctx.beginPath(); zctx.moveTo(mx, 0); zctx.lineTo(mx, zc.height); zctx.stroke();
      }
    }
  }
  zoomPanel.style.display = 'block';
}

// The live composite is capped at LIVE_MAX_PX on its longest side, so no
// single image is ever drawn larger than that. Caching a bitmap pre-shrunk
// to that bound (plus the natural size, which the layout math needs) means
// each rAF tick resamples a few megapixels instead of full camera frames.
const LIVE_MAX_PX = 2000;

// The server refuses output larger than this per side (MAX_CANVAS_PX in
// concat_jpeg.py); the live preview warns before Download would fail.
const MAX_CANVAS_PX = 16384;

// Turns a full-resolution bitmap into a preview-cache entry, taking
// ownership of `full` (closed here unless it is small enough to keep).
async function _previewEntryFrom(full) {
  const w = full.width, h = full.height;
  const s = LIVE_MAX_PX / Math.max(w, h);
  if (s >= 1) return { bmp: full, w, h };
  // Downscale through a canvas: GPU-backed and far cheaper than
  // createImageBitmap's resizeQuality option, which resamples on the CPU.
  try {
    const cv = Object.assign(document.createElement('canvas'), {
      width: Math.max(1, Math.round(w * s)), height: Math.max(1, Math.round(h * s)),
    });
    const c = cv.getContext('2d');
    c.imageSmoothingQuality = 'high';
    c.drawImage(full, 0, 0, cv.width, cv.height);
    return { bmp: await createImageBitmap(cv), w, h };
  } finally { full.close(); }
}

function _getBitmap(cell) {
  // Cache the promise so concurrent callers share one decode.
  if (!_bitmapCache[cell.id]) {
    const p = _decodeCell(cell).then(_previewEntryFrom);
    _bitmapCache[cell.id] = p;
    p.catch(() => { if (_bitmapCache[cell.id] === p) delete _bitmapCache[cell.id]; });
  }
  return _bitmapCache[cell.id];
}

function _evictBitmap(id) {
  const p = _bitmapCache[id];
  if (!p) return;
  delete _bitmapCache[id];
  p.then(e => e.bmp.close(), () => {});
}

// Reused across frames instead of allocating fresh canvases every tick.
// The live composite is drawn straight into the on-screen canvas, so slider
// feedback lands on the next frame with no JPEG encode/decode round trip.
const _liveCanvas = previewLive;
const _fadeCanvas = document.createElement('canvas');

// Draws one image whose leading `ov` px (along `horizontal` or vertical)
// cross-fade over what is already on ctx. Only the overlap band goes
// through the scratch canvas; the rest is a plain drawImage.
function _drawFaded(ctx, e, x, y, dw, dh, ov, horizontal) {
  const bmp = e.bmp;
  const fx = bmp.width / dw, fy = bmp.height / dh;   // dest px → bitmap px
  const bw = horizontal ? ov : dw, bh = horizontal ? dh : ov;
  if (_fadeCanvas.width !== bw || _fadeCanvas.height !== bh) {
    _fadeCanvas.width = bw; _fadeCanvas.height = bh;
  }
  const tc = _fadeCanvas.getContext('2d');
  tc.globalCompositeOperation = 'source-over';
  tc.clearRect(0, 0, bw, bh);
  tc.drawImage(bmp, 0, 0, bw * fx, bh * fy, 0, 0, bw, bh);
  const grad = horizontal ? tc.createLinearGradient(0, 0, ov, 0) : tc.createLinearGradient(0, 0, 0, ov);
  grad.addColorStop(0, 'rgba(0,0,0,0)');
  grad.addColorStop(1, 'rgba(0,0,0,1)');
  tc.globalCompositeOperation = 'destination-in';
  tc.fillStyle = grad;
  tc.fillRect(0, 0, bw, bh);
  ctx.drawImage(_fadeCanvas, x, y);
  if (horizontal && dw > ov) {
    ctx.drawImage(bmp, ov * fx, 0, bmp.width - ov * fx, bmp.height, x + ov, y, dw - ov, dh);
  } else if (!horizontal && dh > ov) {
    ctx.drawImage(bmp, 0, ov * fy, bmp.width, bmp.height - ov * fy, x, y + ov, dw, dh - ov);
  }
}

async function updateLivePreview() {
  const rows = nRows(), cols = nCols();
  const is1Row = rows === 1 && cols >= 2;
  const is1Col = cols === 1 && rows >= 2;
  if (!hasImages() || (!is1Row && !is1Col)) return;

  const direction = is1Row ? 'horizontal' : 'vertical';
  const cells = is1Row
    ? grid[0].filter(Boolean)
    : grid.map(r => r[0]).filter(Boolean);
  if (cells.length < 2) return;

  const gen = _previewGen;
  let entries;
  try { entries = await Promise.all(cells.map(_getBitmap)); }
  catch (_) { return; }
  // A server result (or another layout) replaced the preview while we were
  // waiting on bitmaps — don't paint a stale low-res composite over it.
  if (gen !== _previewGen) return;

  const isFade  = overlapFadeToggle.checked;
  const horizontal = direction === 'horizontal';

  // Output size of each image. Mirrors _stitch_grid's fit: scale every
  // image to the tallest height (row) or widest width (column).
  if (fitToggle.checked) {
    const maxH = Math.max(...entries.map(e => e.h));
    const maxW = Math.max(...entries.map(e => e.w));
    entries = entries.map(e => horizontal
      ? { ...e, w: e.h === maxH ? e.w : Math.round(e.w * maxH / e.h), h: maxH }
      : { ...e, w: maxW, h: e.w === maxW ? e.h : Math.round(e.h * maxW / e.w) });
  }

  // _stitch_grid clamps overlap to one less than the smallest image, so
  // the slider never offers a value that wouldn't be applied.
  const minSide = Math.min(...entries.map(e => horizontal ? e.w : e.h));
  const maxOv = Math.max(0, minSide - 1);
  if (+overlapSlider.max !== maxOv) overlapSlider.max = maxOv;
  if ((parseInt(overlapSlider.value) || 0) > maxOv) _setOverlapPx(maxOv);
  const overlap = Math.min(Math.max(0, parseInt(overlapSlider.value) || 0), maxOv);

  // Natural canvas size (from full-resolution dimensions)
  let natW, natH;
  if (horizontal) {
    natW = entries.reduce((s, e) => s + e.w, 0) - overlap * (entries.length - 1);
    natH = Math.max(...entries.map(e => e.h));
  } else {
    natW = Math.max(...entries.map(e => e.w));
    natH = entries.reduce((s, e) => s + e.h, 0) - overlap * (entries.length - 1);
  }
  if (natW <= 0 || natH <= 0) return;

  // Scale down to ≤LIVE_MAX_PX on longest side for speed
  const scale = Math.min(1, LIVE_MAX_PX / Math.max(natW, natH));
  const cvW = Math.max(1, Math.round(natW * scale));
  const cvH = Math.max(1, Math.round(natH * scale));
  const ov  = Math.round(overlap * scale);

  const cv  = _liveCanvas;
  if (cv.width !== cvW || cv.height !== cvH) { cv.width = cvW; cv.height = cvH; }
  const ctx = cv.getContext('2d');
  // _stitch_grid fills uncovered canvas with black; match it so the
  // preview doesn't hide bars that Download will contain.
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, cvW, cvH);

  let pos = 0;
  for (let i = 0; i < entries.length; i++) {
    const e  = entries[i];
    const dw = Math.round(e.w * scale);
    const dh = Math.round(e.h * scale);
    if (i > 0) pos -= ov;
    const x = horizontal ? pos : 0, y = horizontal ? 0 : pos;
    if (isFade && i > 0 && ov > 0) _drawFaded(ctx, e, x, y, dw, dh, ov, horizontal);
    else ctx.drawImage(e.bmp, x, y, dw, dh);
    pos += horizontal ? dw : dh;
  }

  // Zoom view of seam edge (horizontal layouts only)
  if (_zoomActive && direction === 'horizontal' && ov > 0) {
    const dw0 = Math.round(entries[0].w * scale);
    _lastZoomParams = { cv, dw0, ov, cvW, cvH };
    _renderZoomStrip(_lastZoomParams);
  } else {
    zoomPanel.style.display = 'none';
  }

  _liveDrawCount++;
  _showPreview(previewLive, cvW >= cvH);
  const uneven = new Set(entries.map(e => horizontal ? e.h : e.w)).size > 1;
  const tooBig = natW > MAX_CANVAS_PX || natH > MAX_CANVAS_PX;
  _setPreviewState('live', tooBig
    ? `Too large to save: ${natW} × ${natH} px. The limit is ${MAX_CANVAS_PX} px per side.`
    : uneven
      ? 'Quick preview · Sizes differ, so gaps save as black. Turn on Fit to fill them.'
      : null);
  updateStatus();
}

// Counts live-canvas paints. A server result whose <img> finishes decoding
// after a newer live paint must not cover it.
let _liveDrawCount = 0;

// Bumped whenever the preview pane gets authoritative content (a server
// result), invalidating any live composite still in flight.
let _previewGen = 0;

function _showPreview(el, landscape) {
  previewLive.style.display = el === previewLive ? '' : 'none';
  previewImg.style.display  = el === previewImg  ? '' : 'none';
  appBody.classList.toggle('layout-side',  !landscape);
  appBody.classList.toggle('layout-stack',  landscape);
  appBody.classList.add('has-preview');
}

let _livePreviewPending = false;
function scheduleLivePreview() {
  if (_livePreviewPending) return;
  _livePreviewPending = true;
  requestAnimationFrame(() => { _livePreviewPending = false; updateLivePreview(); });
}

function _fmtSize(bytes) {
  return bytes >= 1024 * 1024 ? (bytes / 1024 / 1024).toFixed(1) + ' MB'
                              : Math.round(bytes / 1024) + ' KB';
}

// Full-quality server render, used as the preview for grids the live
// composite can't draw. Runs automatically after edits; a sequence number
// drops responses that a newer request has superseded.
let _serverPreviewTimer = null;
let _serverPreviewSeq = 0;
function scheduleServerPreview() {
  clearTimeout(_serverPreviewTimer);
  if (imgCount() < 2) return;
  _serverPreviewTimer = setTimeout(runServerPreview, 400);
}

async function runServerPreview() {
  if (imgCount() < 2) return;
  const seq = ++_serverPreviewSeq;
  const sig = _requestSignature();
  if (appBody.classList.contains('has-preview')) _setPreviewState('pending');
  try {
    const res = await fetch('/preview', { method: 'POST', body: _buildFormData() });
    if (!res.ok) {
      throw new Error(await res.text().catch(() => res.statusText));
    }
    const blob = await res.blob();
    if (seq !== _serverPreviewSeq) return;
    _lastServerResult = { ...sig, blob };
    _previewGen++;
    const liveDrawsAtResult = _liveDrawCount;
    if (_previewObjectURL) URL.revokeObjectURL(_previewObjectURL);
    _previewObjectURL = URL.createObjectURL(blob);
    previewImg.onload = () => {
      if (_liveDrawCount !== liveDrawsAtResult || seq !== _serverPreviewSeq) return;
      const w = previewImg.naturalWidth, h = previewImg.naturalHeight;
      _lastServerResult.caption = `${w} × ${h} px · ${_fmtSize(blob.size)}`;
      _showPreview(previewImg, w >= h);
      if (_matchesLastServerResult(_requestSignature())) _setPreviewState('final', _lastServerResult.caption);
      else _setPreviewState('pending');
    };
    previewImg.src = _previewObjectURL;
  } catch (err) {
    if (seq !== _serverPreviewSeq) return;
    console.error(err);
    showError('Preview failed: ' + err.message);
  }
}

document.addEventListener('keydown', e => {
  const mod = e.metaKey || e.ctrlKey;
  if (!mod) return;
  if (e.key === 'z' && !e.shiftKey && _undo) { e.preventDefault(); undoRemove(); }
  else if (e.key === 'Enter' && imgCount() >= 2) { e.preventDefault(); downloadBtn.click(); }
  else if (e.key === 'o') { e.preventDefault(); openPicker(); }
});

const quitBtn   = document.getElementById('quit-btn');
const stoppedEl = document.getElementById('stopped');
// Two-step: the first click arms the button, a second within 3 s quits.
// (An in-page confirm, not window.confirm.)
let _quitArmTimer = null;
function _disarmQuit() {
  clearTimeout(_quitArmTimer);
  _quitArmTimer = null;
  quitBtn.classList.remove('armed');
  quitBtn.textContent = 'Quit';
}
quitBtn.addEventListener('blur', () => { if (_quitArmTimer) _disarmQuit(); });
quitBtn.addEventListener('click', async () => {
  if (!_quitArmTimer) {
    quitBtn.classList.add('armed');
    quitBtn.textContent = 'Click again to quit';
    _quitArmTimer = setTimeout(_disarmQuit, 3000);
    return;
  }
  clearTimeout(_quitArmTimer);
  _quitArmTimer = null;
  quitBtn.classList.remove('armed');
  quitBtn.disabled = true;
  quitBtn.textContent = 'Quitting…';
  try {
    const res = await fetch('/quit', { method: 'POST', headers: { 'X-Requested-With': 'fetch' } });
    if (!res.ok) throw Object.assign(new Error(`Server error ${res.status}`), { fromServer: true });
  } catch (err) {
    // A network error just means the server went away mid-response — that's
    // the goal. Only an actual HTTP error response means quitting failed.
    if (err.fromServer) {
      showError('Quit failed: ' + err.message);
      quitBtn.disabled = false;
      quitBtn.textContent = 'Quit';
      return;
    }
  }
  stoppedEl.classList.add('visible');
  // Browsers only honour this for tabs with a single history entry (as
  // opened by webbrowser.open) or script-opened ones; otherwise it's a
  // no-op and the "stopped" overlay above stays as the fallback.
  window.close();
});

// Names the download after the first image, e.g. beach.jpg → beach_stitched.jpg.
function _downloadName() {
  const first = grid.flat().find(Boolean);
  const base = first ? first.name.replace(/\.[^.]*$/, '') : '';
  return (base || 'stitched') + (base ? '_stitched' : '') + '.jpg';
}

downloadBtn.addEventListener('click', async () => {
  if (!hasImages() || downloadBtn.disabled) return;
  _statusError = null;
  const prev = downloadBtn.textContent;
  downloadBtn.textContent = 'Downloading…';
  downloadBtn.disabled = true;
  try {
    const sig = _requestSignature();
    let blob;
    if (_matchesLastServerResult(sig)) {
      blob = _lastServerResult.blob;
    } else {
      processingEl.classList.add('visible');
      const res = await fetch('/stitch', { method: 'POST', body: _buildFormData() });
      if (!res.ok) throw new Error(await res.text().catch(() => res.statusText));
      blob = await res.blob();
      _lastServerResult = { ...sig, blob };
    }
    const url  = URL.createObjectURL(blob);
    Object.assign(document.createElement('a'), { href: url, download: _downloadName() }).click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  } catch (err) {
    showError('Download failed: ' + err.message);
  } finally {
    processingEl.classList.remove('visible');
    downloadBtn.textContent = prev;
    downloadBtn.disabled = false;
  }
});

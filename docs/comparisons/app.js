'use strict';
const byId = (id) => document.getElementById(id);
const keys = ['original', 'glc', 'rift'];
const labels = {original: 'Original', glc: 'GLC', rift: 'RIFT'};
const pairs = {three: keys, 'glc-rift': ['glc', 'rift'], 'original-rift': ['original', 'rift'], 'original-glc': ['original', 'glc']};
const state = {scene: null, mode: 'three', view: 'full', positions: [33.3, 66.7], sequence: 0};
let data;
const stage = byId('stage');
const viewer = byId('viewer');
const ranges = [byId('range-0'), byId('range-1')];
const dividers = [byId('divider-0'), byId('divider-1')];
const percent = (value) => Math.round(value) + '%';
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

function resetPositions() {
  state.positions = state.mode === 'three' ? [33.3, 66.7] : [50, 66.7];
}
function setDivider(index, value) {
  if (!Number.isFinite(value)) return;
  if (state.mode !== 'three') state.positions[0] = clamp(value, 0, 100);
  else if (index === 0) state.positions[0] = clamp(value, 0, state.positions[1] - 2);
  else state.positions[1] = clamp(value, state.positions[0] + 2, 100);
  renderPosition();
}
function renderPosition() {
  const active = pairs[state.mode];
  const boundaries = state.mode === 'three' ? [0, state.positions[0], state.positions[1], 100] : [0, state.positions[0], 100];
  for (const key of keys) {
    const index = active.indexOf(key);
    const layer = byId('layer-' + key);
    const tag = byId('label-' + key);
    layer.hidden = index < 0;
    tag.hidden = index < 0 || (boundaries[index + 1] - boundaries[index]) < 13;
    if (index >= 0) {
      const left = boundaries[index], right = boundaries[index + 1];
      layer.style.clipPath = 'inset(0 ' + (100 - right) + '% 0 ' + left + '%)';
      tag.style.left = ((left + right) / 2) + '%';
    }
  }
  for (let index = 0; index < 2; index++) {
    const enabled = index === 0 || state.mode === 'three';
    dividers[index].hidden = !enabled;
    byId('range-group-' + index).hidden = !enabled;
    if (!enabled) continue;
    const value = state.positions[index];
    const name = labels[active[index]] + ' / ' + labels[active[index + 1]];
    dividers[index].style.left = value + '%';
    dividers[index].setAttribute('aria-valuenow', value.toFixed(1));
    dividers[index].setAttribute('aria-valuetext', percent(value));
    dividers[index].setAttribute('aria-label', name + ' divider');
    ranges[index].value = value;
    ranges[index].setAttribute('aria-label', name + ' divider');
    byId('range-name-' + index).textContent = name;
    byId('value-' + index).textContent = percent(value);
  }
  document.querySelector('.range-controls').classList.toggle('single', state.mode !== 'three');
}
function renderRegion() {
  if (!state.scene) return;
  const s = state.scene;
  const crop = state.view === 'detail';
  const [x, y, w, h] = crop ? s.roi : [0, 0, s.width, s.height];
  stage.style.aspectRatio = w + ' / ' + h;
  viewer.style.setProperty('--view-ratio', String(w / h));
  stage.style.setProperty('--image-width', (s.width / w * 100) + '%');
  stage.style.setProperty('--image-height', (s.height / h * 100) + '%');
  stage.style.setProperty('--image-left', (-x / w * 100) + '%');
  stage.style.setProperty('--image-top', (-y / h * 100) + '%');
  byId('region-meta').textContent = crop ? w + ' × ' + h + ' shared detail crop' : '1920 × 1080 full frame';
  byId('detail-view').setAttribute('aria-pressed', String(crop));
  byId('full-view').setAttribute('aria-pressed', String(!crop));
  byId('roi-box').hidden = false;
}
function renderLegend() {
  if (!state.scene) return;
  const active = pairs[state.mode];
  const container = byId('method-legend');
  container.replaceChildren();
  container.style.setProperty('--method-count', active.length);
  for (const key of active) {
    const pill = document.createElement('div');
    pill.className = 'method-pill ' + key;
    const name = document.createElement('strong');
    name.textContent = labels[key];
    const detail = document.createElement('span');
    detail.textContent = key === 'original' ? 'Reference' : state.scene.methods[key].bpp.toFixed(6) + (key === 'glc' ? '* bpp' : ' bpp');
    pill.append(name, detail);
    container.append(pill);
  }
}
async function selectScene(id, updateHash = true) {
  const scene = data.scenes.find((s) => s.id === id) || data.scenes.find((s) => s.id === data.default_scene);
  state.scene = scene;
  const sequence = ++state.sequence;
  if (updateHash && location.hash.slice(1) !== scene.id) history.replaceState(null, '', '#' + scene.id);
  for (const button of document.querySelectorAll('.scene-card')) button.setAttribute('aria-pressed', String(button.dataset.scene === scene.id));
  byId('scene-title').textContent = scene.title;
  byId('scene-meta').textContent = scene.dataset + ' / frame ' + scene.frame + ' / 1080p';
  byId('scene-description').textContent = scene.detail + '. The highlighted area is shared by all three methods.';
  const [x, y, w, h] = scene.roi;
  Object.assign(byId('roi-box').style, {left: (x / scene.width * 100) + '%', top: (y / scene.height * 100) + '%', width: (w / scene.width * 100) + '%', height: (h / scene.height * 100) + '%'});
  byId('context-original').src = scene.methods.original.src;
  for (const key of keys) byId('download-' + key).href = scene.methods[key].src;
  byId('load-error').hidden = true;
  viewer.classList.add('is-loading');
  viewer.setAttribute('aria-busy', 'true');
  renderRegion();
  renderLegend();
  renderPosition();
  try {
    await Promise.all(keys.map(async (key) => {
      const img = byId('image-' + key);
      img.alt = labels[key] + ', ' + scene.title + ', frame ' + scene.frame;
      img.src = scene.methods[key].src;
      await img.decode();
    }));
    if (sequence !== state.sequence) return;
    viewer.classList.remove('is-loading');
    viewer.setAttribute('aria-busy', 'false');
  } catch (error) {
    if (sequence !== state.sequence) return;
    viewer.classList.remove('is-loading');
    viewer.setAttribute('aria-busy', 'false');
    byId('load-error').textContent = 'An image could not be loaded. Reload this page or open the lossless files below.';
    byId('load-error').hidden = false;
  }
}
function createPicker() {
  const picker = byId('scene-picker');
  for (const scene of data.scenes) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'scene-card';
    button.dataset.scene = scene.id;
    button.setAttribute('aria-pressed', 'false');
    const thumb = document.createElement('img');
    thumb.src = scene.thumbnail;
    thumb.alt = '';
    thumb.width = 72; thumb.height = 48;
    const caption = document.createElement('span');
    const title = document.createElement('strong');
    title.textContent = scene.title;
    const meta = document.createElement('small');
    meta.textContent = scene.dataset + ' · frame ' + scene.frame;
    caption.append(title, meta);
    button.append(thumb, caption);
    button.addEventListener('click', () => selectScene(scene.id));
    picker.append(button);
  }
}
function pointerValue(event) {
  const rect = stage.getBoundingClientRect();
  return (event.clientX - rect.left) / rect.width * 100;
}
function bindDrag(element, fixedIndex) {
  let index = null;
  const end = (event) => {
    if (index === null) return;
    if (element.hasPointerCapture(event.pointerId)) element.releasePointerCapture(event.pointerId);
    index = null;
  };
  element.addEventListener('pointerdown', (event) => {
    if (event.button !== 0 || viewer.classList.contains('is-loading')) return;
    event.preventDefault();
    event.stopPropagation();
    const value = pointerValue(event);
    index = fixedIndex;
    if (index === undefined) index = state.mode === 'three' && Math.abs(value - state.positions[1]) < Math.abs(value - state.positions[0]) ? 1 : 0;
    element.setPointerCapture(event.pointerId);
    if (event.pointerType === 'mouse') dividers[index].focus({preventScroll: true});
    setDivider(index, value);
  });
  element.addEventListener('pointermove', (event) => {
    if (index !== null) setDivider(index, pointerValue(event));
  });
  element.addEventListener('pointerup', end);
  element.addEventListener('pointercancel', end);
  element.addEventListener('lostpointercapture', () => { index = null; });
}
for (let index = 0; index < 2; index++) {
  ranges[index].addEventListener('input', () => setDivider(index, Number(ranges[index].value)));
  bindDrag(dividers[index], index);
  dividers[index].addEventListener('keydown', (event) => {
    let value = state.positions[index];
    const step = event.shiftKey ? 5 : 1;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') value -= step;
    else if (event.key === 'ArrowRight' || event.key === 'ArrowUp') value += step;
    else if (event.key === 'Home') value = 0;
    else if (event.key === 'End') value = 100;
    else return;
    event.preventDefault();
    setDivider(index, value);
  });
}
bindDrag(stage);
byId('mode').addEventListener('change', (event) => {
  state.mode = event.target.value;
  resetPositions(); renderLegend(); renderPosition();
});
byId('detail-view').addEventListener('click', () => { state.view = 'detail'; renderRegion(); });
byId('full-view').addEventListener('click', () => { state.view = 'full'; renderRegion(); });
byId('reset').addEventListener('click', () => { resetPositions(); renderPosition(); });
byId('fullscreen').addEventListener('click', async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await byId('comparison-shell').requestFullscreen();
  } catch (error) { /* The normal viewer remains available if fullscreen is unavailable. */ }
});
if (!document.fullscreenEnabled) byId('fullscreen').hidden = true;
document.addEventListener('fullscreenchange', () => byId('fullscreen').setAttribute('aria-label', document.fullscreenElement ? 'Exit fullscreen' : 'Enter fullscreen'));
window.addEventListener('hashchange', () => { if (data) selectScene(location.hash.slice(1), false); });
fetch('data.json').then((response) => {
  if (!response.ok) throw new Error('Metadata unavailable');
  return response.json();
}).then((payload) => {
  data = payload;
  createPicker();
  return selectScene(location.hash.slice(1) || data.default_scene, false);
}).catch(() => {
  viewer.classList.remove('is-loading');
  viewer.setAttribute('aria-busy', 'false');
  byId('scene-title').textContent = 'Comparison unavailable';
  byId('load-error').textContent = 'The comparison metadata could not be loaded. Please reload this page.';
  byId('load-error').hidden = false;
});

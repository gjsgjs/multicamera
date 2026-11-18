// Exercise the asynchronous UI state/data contract without a browser dependency.
// This does not replace a visual test of the native folder picker.
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import assert from 'node:assert/strict';

class Element {
  constructor(tag = '') {
    this.tag = tag; this.children = []; this.hidden = false; this.disabled = false;
    this.value = ''; this.textContent = ''; this.style = {}; this.handlers = {};
    this.lastChild = {textContent: ''}; this.scrollHeight = this.clientHeight = 0; this.scrollTop = 0;
    this.classList = {add() {}, toggle() {}};
  }
  append(...items) {this.children.push(...items);}
  replaceChildren(...items) {this.children = items;}
  setAttribute() {}
  addEventListener(name, callback) {this.handlers[name] = callback;}
  get firstElementChild() {return this.children[0];}
  showModal() {this.open = true;}
  close() {this.open = false;}
}

export async function runTests() {
  const root = new URL('.', import.meta.url);
  const html = await readFile(new URL('static/index.html', root), 'utf8');
  const source = await readFile(new URL('static/app.js', root), 'utf8');
  assert.match(html, /option value="lowlight"/);
  const elements = new Map([...html.matchAll(/id="([^"]+)"/g)].map(match => [match[1], new Element()]));
  const get = id => {assert.ok(elements.has(id), `Missing DOM id ${id}`); return elements.get(id);};
  get('start').append(new Element('span'));
  get('mode').value = 'auto'; get('blend').value = 'sharp'; get('columns').value = '4'; get('preprocess-method').value = 'none';
  const calls = []; let job = null; let failPreprocess = false; let serial = 0;
  const fetch = async (path, options = {}) => {
    calls.push({path, ...options});
    const data = options.body && typeof options.body === 'string' ? JSON.parse(options.body) : {};
    let response;
    if (path === '/api/health') response = {ok: true};
    else if (path === '/api/jobs') {
      job = {id: (++serial).toString(16).padStart(32, '0'), status: 'uploading', files: data.files, logs: [], processed: null}; response = job;
    } else if (/\/files\/\d+$/.test(path)) response = {uploaded: 0};
    else if (path.endsWith('/preprocess')) {
      assert.ok(['dehaze', 'lowlight'].includes(data.method));
      job.preprocess_method = data.method;
      job.status = failPreprocess ? 'preprocess_failed' : 'ready'; job.error = failPreprocess ? 'Preprocessing failed' : null;
      job.processed = failPreprocess ? null : {method: data.method, images: job.files.map((name, i) => ({name, preview: `/processed/${i}/preview`, full: `/processed/${i}`}))}; response = job;
    } else if (path.endsWith('/run')) {
      assert.equal(data.use_processed, true); assert.equal(data.mode, 'grid'); assert.equal(data.columns, 2);
      job.status = 'success'; job.result = {width: 128, height: 48, preview: '/preview', download: '/result'}; response = job;
    } else response = job;
    return {ok: true, json: async () => response};
  };
  let blob = 0;
  class MockURL extends URL {static createObjectURL() {return `blob:${++blob}`;} static revokeObjectURL() {}}
  const context = vm.createContext({document: {getElementById: get, createElement: tag => new Element(tag),
      querySelectorAll: query => get('file-list').children.flatMap(label => label.children.filter(child => child.tag === (query.endsWith('img') ? 'img' : 'input')))},
    fetch, URL: MockURL, URLSearchParams, history: {replaceState() {}}, location: {search: ''}, setTimeout, console});
  new vm.Script(source, {filename: 'app.js'}).runInContext(context);
  await new Promise(resolve => setTimeout(resolve, 0));
  const originals = ['00.png', '01.png', 'ground_truth.png'].map(name => ({name, webkitRelativePath: `folder/${name}`}));
  get('folder-input').onchange({target: {files: originals}});
  assert.equal(get('start').disabled, false);
  get('preprocess-method').value = 'dehaze'; get('preprocess-method').onchange();
  assert.equal(get('start').disabled, true, 'Cannot stitch stale/original pixels before applying dehaze');
  await get('apply-preprocess').onclick();
  assert.equal(get('start').disabled, false);
  assert.equal(get('preprocess-preview-tools').hidden, false);
  assert.equal(get('file-list').children[0].children[0].src, '/processed/0/preview');
  assert.equal(job.files.length, 2, 'Ground-truth image must remain excluded');
  get('show-original').onclick();
  assert.equal(get('file-list').children[0].children[0].src, 'blob:1');
  const uploadCount = calls.filter(call => call.method === 'PUT').length;
  const createCount = calls.filter(call => call.path === '/api/jobs').length;
  get('mode').value = 'grid'; get('columns').value = '2'; get('mode').onchange();
  await get('start').onclick();
  assert.equal(calls.filter(call => call.method === 'PUT').length, uploadCount, 'Stitch must reuse stored processed inputs');
  assert.equal(calls.filter(call => call.path === '/api/jobs').length, createCount);
  assert.match(get('dimensions').textContent, /128/);
  // Method changes invalidate old results; natural previews and original-view
  // toggles must still reuse the saved processed images for stitching.
  get('preprocess-method').value = 'lowlight'; get('preprocess-method').onchange();
  assert.equal(get('start').disabled, true);
  await get('apply-preprocess').onclick();
  assert.equal(job.processed.method, 'lowlight');
  assert.equal(get('start').disabled, false);
  assert.equal(get('show-processed').textContent, '查看自然增强图');
  assert.match(get('processed-tag').textContent, /暗光增强/);
  assert.match(get('stitch-source').textContent, /自然增强图/);
  assert.match(get('preprocess-hint').textContent, /自然增强图/);
  const zoom = get('file-list').children[0].children.find(child => child.tag === 'button');
  zoom.onclick({preventDefault() {}, stopPropagation() {}});
  assert.equal(get('image-dialog-content').src, '/processed/0');
  assert.match(get('image-dialog-title').textContent, /自然增强图/);
  get('close-image').onclick();
  get('show-original').onclick();
  assert.equal(get('file-list').children[0].children[0].src, 'blob:1');
  assert.match(get('stitch-source').textContent, /自然增强图/);
  const lowlightUploads = calls.filter(call => call.method === 'PUT').length;
  await get('start').onclick();
  assert.equal(calls.filter(call => call.method === 'PUT').length, lowlightUploads);
  assert.equal(get('start').disabled, true, 'Consumed preprocessing job needs a new apply');
  get('folder-input').onchange({target: {files: originals}});
  assert.equal(get('start').disabled, true, 'New import invalidates previous processing');
  await get('apply-preprocess').onclick();
  const checkbox = get('file-list').children[0].children.find(child => child.tag === 'input');
  checkbox.checked = false; checkbox.handlers.change();
  checkbox.checked = true; checkbox.handlers.change();
  assert.equal(get('start').disabled, true, 'Changing selection invalidates previous processing');
  failPreprocess = true;
  await get('apply-preprocess').onclick();
  assert.equal(get('start').disabled, true);
  assert.equal(get('error-panel').hidden, false);
  assert.equal(get('error-text').textContent, 'Preprocessing failed');
  assert.match(get('preprocess-hint').textContent, /暗光增强失败/);
  get('preprocess-method').value = 'none'; get('preprocess-method').onchange();
  await get('apply-preprocess').onclick();
  assert.equal(get('start').disabled, false);
  assert.equal(get('preprocess-preview-tools').hidden, true);
  return 'Frontend state tests passed: dehaze/lowlight apply, natural preview/zoom, original toggle, processed-input reuse, invalidation, failure, reset.';
}

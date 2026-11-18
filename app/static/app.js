const $ = id => document.getElementById(id);
let files = [], selected = new Set(), objectUrls = [], busy = false, currentJob = null;
let prepared = null, previewMode = 'original';
const supported = /\.(png|jpe?g|bmp|tiff?|webp)$/i;
const groundTruth = /^(ground[_-]?truth|gt)\./i;
const preprocessing = {
  dehaze: {name: '去雾', result: '去雾结果', pending: '正在估计共享大气光，请稍候…'},
  lowlight: {name: '暗光增强', result: '自然增强图', pending: '正在估计光照并增强暗部，请稍候…'}
};
function processingInfo(method = $('preprocess-method').value) {return preprocessing[method];}
function state(title, badge, progress, label = '') {
  $('status-badge').textContent = title;
  $('status-badge').className = 'badge ' + badge;
  $('progress-title').textContent = title;
  $('progress-label').textContent = label;
  $('progress-bar').style.width = progress + '%';
  ['s1', 's2', 's3'].forEach((id, i) => $(id).classList.toggle('active', progress > [0, 40, 80][i]));
}
function setBusy(value) {
  busy = value;
  ['choose-folder', 'change-folder', 'toggle-all', 'mode', 'blend', 'columns', 'preprocess-method'].forEach(id => $(id).disabled = value);
  document.querySelectorAll('.file-item input').forEach(el => el.disabled = value);
  updateSelection();
  $('start').firstElementChild.textContent = value ? '正在处理…' : '开始拼接';
}
function showError(message) {
  $('error-text').textContent = message;
  let advice = '可展开运行日志查看详情，调整输入或设置后重新尝试。';
  if (/disconnected|overlap/i.test(message)) advice = '部分图片之间没有足够的可靠重叠。若是已知顺序的无重叠裁块，请改用「网格排列」；有重叠的相机画面请检查纹理、覆盖范围和水印。';
  if (/memory|allocate/i.test(message)) advice = '系统可用内存不足，请关闭暂时不用的程序，或减少单次输入数量。';
  if (/folded/i.test(message)) advice = '网格出现翻折，请检查图片是否真实重叠，避免重复纹理或不相关图片混入。';
  $('error-advice').textContent = advice;
  $('error-panel').hidden = false;
}
async function request(url, options = {}) {
  const response = await fetch(url, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `请求失败 (${response.status})`);
  return value;
}
function updateSelection() {
  $('file-count').textContent = selected.size + ' 张已选';
  $('start').disabled = busy || selected.size < 2 || ($('preprocess-method').value !== 'none' && (!prepared || prepared.consumed || prepared.method !== $('preprocess-method').value));
  $('apply-preprocess').disabled = busy || selected.size < 2;
}
function refreshPreviews() {
  document.querySelectorAll('.file-item img').forEach((image, index) => {
    const processed = prepared?.mapping.get(index);
    image.src = previewMode === 'processed' && processed ? processed.preview : objectUrls[index];
  });
  const info = prepared && processingInfo(prepared.method);
  $('stitch-source').textContent = prepared ? `当前拼接输入：${prepared.mapping.size} 张${info.result}（预览切换不改变输入）` : '当前拼接输入：原图';
  if (info) {
    $('processed-tag').textContent = `已完成${info.name}`;
    $('show-processed').textContent = `查看${info.result}`;
  }
}
function invalidatePreprocessing() {
  prepared = null; previewMode = 'original';
  $('preprocess-preview-tools').hidden = true;
  const info = processingInfo();
  $('preprocess-hint').textContent = info ? `请点击「应用」，完成${info.name}后再开始拼接。` : '未使用前处理，拼接将使用原图。';
  refreshPreviews(); updateSelection();
}
function settings() {return {mode: $('mode').value, columns: Number($('columns').value), blend: $('blend').value};}
async function uploadSelection(forPreprocessing = false) {
  const chosen = files.filter((_, index) => selected.has(index));
  if (chosen.length < 2 || chosen.length > 100) throw new Error('请选择 2～100 张图片。');
  const job = await request('/api/jobs', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({files: chosen.map(file => file.name), ...settings(), ...(forPreprocessing ? {mode: 'auto'} : {})})});
  for (let i = 0; i < chosen.length; i++) {
    $('busy-detail').textContent = `${i + 1} / ${chosen.length} · ${chosen[i].name}`;
    state('正在导入图片', 'running', 5 + 30 * i / chosen.length, `${i + 1} / ${chosen.length}`);
    await request(`/api/jobs/${job.id}/files/${i}`, {method: 'PUT', body: chosen[i]});
  }
  return job;
}
function importFolder(list) {
  objectUrls.forEach(url => URL.revokeObjectURL(url)); objectUrls = [];
  const all = [...list];
  files = all.filter(file => supported.test(file.name) && (!file.webkitRelativePath || file.webkitRelativePath.split('/').length === 2))
    .sort((a, b) => a.name.localeCompare(b.name, 'en', {numeric: true}));
  selected = new Set(files.map((file, index) => groundTruth.test(file.name) ? -1 : index).filter(index => index >= 0));
  prepared = null; previewMode = 'original';
  $('folder-name').textContent = all[0]?.webkitRelativePath.split('/')[0] || '已选文件夹';
  $('folder-info').hidden = false; $('selection-toolbar').hidden = !files.length;
  $('choose-folder').hidden = files.length > 0;
  $('error-panel').hidden = true;
  $('result-info').hidden = true; $('result-image').hidden = true; $('empty-state').hidden = false;
  history.replaceState(null, '', '/');
  $('file-list').replaceChildren();
  files.forEach((file, index) => {
    const label = document.createElement('label'); label.className = 'file-item'; label.title = file.name;
    const image = document.createElement('img'); image.alt = ''; image.loading = 'lazy';
    const url = URL.createObjectURL(file); objectUrls.push(url); image.src = url;
    const input = document.createElement('input'); input.type = 'checkbox'; input.checked = selected.has(index);
    input.setAttribute('aria-label', `选择 ${file.name}`);
    input.addEventListener('change', () => {input.checked ? selected.add(index) : selected.delete(index); invalidatePreprocessing();});
    const zoom = document.createElement('button'); zoom.className = 'zoom-image'; zoom.type = 'button';
    zoom.textContent = '⤢'; zoom.setAttribute('aria-label', `放大预览 ${file.name}`);
    zoom.onclick = event => {
      event.preventDefault(); event.stopPropagation();
      const processed = prepared?.mapping.get(index);
      const useProcessed = previewMode === 'processed' && processed;
      $('image-dialog-title').textContent = file.name + (useProcessed ? ` · ${processingInfo(prepared.method).result}` : ' · 原图');
      $('image-dialog-content').src = useProcessed ? processed.full : objectUrls[index];
      $('image-dialog').showModal();
    };
    const name = document.createElement('span'); name.textContent = file.name;
    label.append(image, input, zoom, name); $('file-list').append(label);
  });
  invalidatePreprocessing();
  $('import-hint').textContent = files.length ? `发现 ${files.length} 张同级图片。默认排除 ground_truth / gt 参考图，可手动勾选。` : '该文件夹同一级没有支持的图片，请重新选择。';
  state('待开始', '', 0, `${selected.size} 张已选`);
}
function displayJob(job) {
  const log = $('logs'); const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 50;
  log.textContent = job.logs.length ? job.logs.join('\n') : '后端已启动，正在准备图片与特征匹配…';
  if (nearBottom) log.scrollTop = log.scrollHeight;
  if (job.status === 'preprocessing') {
    const info = processingInfo(job.preprocess_method);
    state(`正在${info.name}`, 'running', 55, '前处理中');
    $('busy-title').textContent = `正在对所选图片${info.name}`;
    const last = [...job.logs].reverse().find(line => /^\d+\/\d+:/.test(line));
    $('busy-detail').textContent = last || info.pending;
    $('preprocess-hint').textContent = '后端正在处理，图片尺寸和编号保持不变。';
  } else if (job.status === 'ready') {
    const info = processingInfo(job.processed.method);
    const indices = files.map((_, index) => index).filter(index => selected.has(index));
    prepared = {id: job.id, method: job.processed.method, mapping: new Map(indices.map((index, order) => [index, job.processed.images[order]])), consumed: false};
    previewMode = 'processed'; refreshPreviews();
    $('preprocess-preview-tools').hidden = false;
    $('preprocess-hint').textContent = `已完成 ${job.files.length} 张图片${info.name}，下方展示${info.result}。拼接会使用这些处理结果。`;
    $('busy-state').hidden = true; $('empty-state').hidden = false;
    state(`${info.name}完成，待拼接`, 'success', 35, `${job.files.length} 张已处理`); setBusy(false);
  } else if (job.status === 'running') {
    const rendering = job.logs.some(line => /Rendering original-size/.test(line));
    state(rendering ? '正在生成全景' : '正在计算拼接', 'running', rendering ? 88 : 55, '运行中');
    $('busy-title').textContent = rendering ? '正在生成原尺寸全景' : '正在匹配与优化画面';
    $('busy-detail').textContent = `已导入 ${job.files.length} 张图片，后端正在处理`;
  } else if (job.status === 'success') {
    $('busy-state').hidden = true; $('empty-state').hidden = true;
    const image = $('result-image'); image.src = job.result.preview; image.hidden = false;
    image.onerror = () => showError('结果文件已生成，但预览加载失败。你仍可点击下载原尺寸 PNG。');
    $('result-info').hidden = false;
    $('dimensions').textContent = `${job.result.width.toLocaleString()} × ${job.result.height.toLocaleString()} px`;
    $('result-detail').textContent = `${(job.result.width * job.result.height / 1e6).toFixed(2)} MP · ${job.files.length} 张输入 · PNG 原尺寸输出`;
    $('download').href = job.result.download;
    state('拼接完成', 'success', 100, '已完成'); setBusy(false);
  } else if (job.status === 'failed' || job.status === 'preprocess_failed') {
    $('busy-state').hidden = true; $('empty-state').hidden = false;
    showError(job.error || '拼接失败，请查看日志。');
    $('logs-details').open = true;
    state(job.status === 'preprocess_failed' ? '前处理失败' : '拼接失败', 'failed', 100, '请检查错误原因');
    if (job.status === 'preprocess_failed') {invalidatePreprocessing(); $('preprocess-hint').textContent = `${processingInfo(job.preprocess_method).name}失败，请查看错误原因并重新应用。`;}
    setBusy(false);
  }
}
async function monitor(id) {
  let failures = 0;
  for (;;) {
    try {
      const job = await request(`/api/jobs/${id}`); failures = 0;
      currentJob = job; displayJob(job);
      if (['success', 'failed', 'ready', 'preprocess_failed'].includes(job.status)) return;
    } catch (error) {
      if (++failures >= 5) throw new Error('与本地服务的连接中断。请确认启动窗口仍在运行，再重新导入。');
      $('busy-detail').textContent = '连接暂时中断，正在重试…';
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}
$('choose-folder').onclick = $('change-folder').onclick = () => {$('folder-input').value = ''; $('folder-input').click();};
$('folder-input').onchange = event => {if (event.target.files.length) importFolder(event.target.files);};
$('toggle-all').onclick = () => {
  selected = selected.size === files.length ? new Set() : new Set(files.map((_, i) => i));
  document.querySelectorAll('.file-item input').forEach((input, index) => input.checked = selected.has(index)); invalidatePreprocessing();
};
$('preprocess-method').onchange = invalidatePreprocessing;
$('show-original').onclick = () => {previewMode = 'original'; refreshPreviews();};
$('show-processed').onclick = () => {previewMode = 'processed'; refreshPreviews();};
$('close-image').onclick = () => $('image-dialog').close();
$('apply-preprocess').onclick = async () => {
  if (busy || selected.size < 2) return;
  history.replaceState(null, '', '/');
  invalidatePreprocessing();
  if ($('preprocess-method').value === 'none') {state('已应用：使用原图', '', 0, `${selected.size} 张`); return;}
  setBusy(true); $('error-panel').hidden = true; $('busy-state').hidden = false; $('empty-state').hidden = true;
  $('result-image').hidden = true; $('result-info').hidden = true;
  const method = $('preprocess-method').value;
  $('busy-title').textContent = `正在导入待${processingInfo(method).name}图片`;
  try {
    const job = await uploadSelection(true);
    await request(`/api/jobs/${job.id}/preprocess`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({method})});
    await monitor(job.id);
  } catch (error) {
    $('busy-state').hidden = true; $('empty-state').hidden = false; showError(error.message);
    state('前处理未完成', 'failed', 0, '请重新应用'); setBusy(false);
  }
};
$('mode').onchange = () => {
  const grid = $('mode').value === 'grid'; $('columns-wrap').hidden = !grid;
  $('mode-hint').textContent = grid ? '按已知布局放置原始像素，不运行特征匹配或 GES 优化。' : '自动检测重叠区域，并进行 HED / GES-GSP 网格优化。';
};
$('start').onclick = async () => {
  if (busy || selected.size < 2) return;
  const method = $('preprocess-method').value;
  const useProcessed = method !== 'none';
  const info = processingInfo(method);
  if (useProcessed && (!prepared || prepared.consumed || prepared.method !== method)) {showError(`请先应用${info.name}，完成后再开始拼接。`); return;}
  const chosen = files.filter((_, index) => selected.has(index));
  if (chosen.length > 100) {showError('单次任务最多支持 100 张图片。'); return;}
  const columns = Number($('columns').value);
  if ($('mode').value === 'grid' && (!Number.isInteger(columns) || columns < 1 || columns > chosen.length)) {showError('网格列数必须在 1 和所选图片数量之间。'); return;}
  setBusy(true); $('error-panel').hidden = true; $('result-info').hidden = true; $('result-image').hidden = true;
  $('empty-state').hidden = true; $('busy-state').hidden = false;
  $('busy-title').textContent = useProcessed ? `正在使用${info.result}开始拼接` : '正在导入图片';
  $('logs').textContent = useProcessed ? `复用已保存的${info.result}，不重新上传原图…` : '正在将所选图片传给本机后端…';
  try {
    const job = useProcessed ? {id: prepared.id} : await uploadSelection();
    await request(`/api/jobs/${job.id}/run`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({...settings(), use_processed: useProcessed})});
    if (useProcessed) {prepared.consumed = true; $('preprocess-hint').textContent = `此组${info.result}已用于拼接。如需再拼接，可重新应用${info.name}创建新任务。`;}
    history.replaceState(null, '', `?job=${job.id}`);
    await monitor(job.id);
  } catch (error) {
    $('busy-state').hidden = true; $('empty-state').hidden = false; showError(error.message);
    state('任务未完成', 'failed', 0, '请重试'); setBusy(false);
  }
};
(async () => {
  try {
    await request('/api/health'); $('connection').classList.add('online'); $('connection').lastChild.textContent = ' 本地服务已连接';
    const id = new URLSearchParams(location.search).get('job');
    if (id && /^[a-f0-9]{32}$/.test(id)) {
      setBusy(true); $('busy-state').hidden = false; $('empty-state').hidden = true;
      await monitor(id);
    }
  } catch (error) {
    $('busy-state').hidden = true; $('empty-state').hidden = false;
    showError(error.message); state('服务或任务不可用', 'failed', 0, '请重新导入或检查服务'); setBusy(false);
  }
})();

function addRipple(e) {
  const target = e.currentTarget;
  const rect = target.getBoundingClientRect();
  const ripple = document.createElement('span');
  ripple.className = 'ripple';
  const size = Math.max(rect.width, rect.height);
  ripple.style.width = ripple.style.height = `${size}px`;
  ripple.style.left = `${e.clientX - rect.left - size / 2}px`;
  ripple.style.top = `${e.clientY - rect.top - size / 2}px`;
  target.appendChild(ripple);
  ripple.addEventListener('animationend', () => ripple.remove());
}

document.querySelectorAll('button, .button').forEach(btn => {
  btn.addEventListener('click', addRipple);
});

function showToast(msg, timeout = 3000) {
  const container = document.getElementById('toast');
  if (!container) return;
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => el.remove(), timeout);
}

const input = document.getElementById('image');
const preview = document.getElementById('preview');
const dropZone = document.getElementById('drop-zone');
const fileNameEl = document.getElementById('file-name');
const fileSizeEl = document.getElementById('file-size');
const form = document.getElementById('upload-form');
const loading = document.getElementById('loading');

const MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10MB (match backend)

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return '';
  const units = ['B','KB','MB','GB'];
  let i = 0; let val = bytes;
  while (val >= 1024 && i < units.length - 1) { val /= 1024; i++; }
  return `${val.toFixed(2)} ${units[i]}`;
}

function showPreview(file) {
  preview.innerHTML = '';
  if (!file) return;
  const img = document.createElement('img');
  img.alt = 'preview';
  img.className = 'preview-img';
  img.src = URL.createObjectURL(file);
  preview.appendChild(img);
  fileNameEl && (fileNameEl.textContent = file.name || '');
  fileSizeEl && (fileSizeEl.textContent = file.size ? `• ${formatBytes(file.size)}` : '');
}

function validateFile(file) {
  if (!file) return false;
  if (file.size > MAX_SIZE_BYTES) { showToast('Tệp quá lớn (tối đa 10MB).'); return false; }
  if (!file.type.startsWith('image/')) { showToast('Vui lòng chọn tệp hình ảnh.'); return false; }
  return true;
}

if (input && preview) {
  input.addEventListener('change', () => {
    const file = input.files && input.files[0];
    if (!validateFile(file)) { input.value = ''; return; }
    showPreview(file);
  });
}

if (dropZone && input) {
  ['dragenter','dragover'].forEach(evt => dropZone.addEventListener(evt, e => {
    e.preventDefault(); e.stopPropagation();
    dropZone.classList.add('dragover');
  }));
  ['dragleave','drop'].forEach(evt => dropZone.addEventListener(evt, e => {
    e.preventDefault(); e.stopPropagation();
    dropZone.classList.remove('dragover');
  }));
  dropZone.addEventListener('drop', e => {
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (!validateFile(file)) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    showPreview(file);
  });
  dropZone.addEventListener('click', () => input.click());
  dropZone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
}

if (form) { form.addEventListener('submit', () => { loading && loading.classList.add('show'); }); }

// Animate bars to their widths on load and bind model info
window.addEventListener('load', () => {
  document.querySelectorAll('.bars .bar span').forEach(span => {
    const w = span.style.width; span.style.width = '0%'; requestAnimationFrame(() => requestAnimationFrame(() => span.style.width = w));
  });
  const thresh = document.getElementById('min_prob');
  const threshVal = document.getElementById('threshVal');
  if (thresh && threshVal) { const update = () => { threshVal.textContent = `${Math.round(parseFloat(thresh.value) * 100)}%`; }; thresh.addEventListener('input', update); update(); }
  const infoMap = (window.__MODEL_INFO__ || {});
  const select = document.getElementById('model');
  const infoRoot = document.getElementById('model-info');
  const setInfo = (key) => {
    const data = infoMap[key] || {}; infoRoot && infoRoot.querySelectorAll('[data-field]').forEach(el => { const field = el.getAttribute('data-field'); if (field && data[field] !== undefined) el.textContent = data[field]; });
  };
  if (select) { setInfo(select.value); select.addEventListener('change', () => setInfo(select.value)); }

  // Tabs
  document.querySelectorAll('.tab-buttons button').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-buttons button').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const id = btn.getAttribute('data-tab');
      const panel = document.getElementById(`panel-${id}`);
      panel && panel.classList.add('active');
    });
  });

  // Modal preview
  const viewBtn = document.getElementById('view-large');
  const modal = document.getElementById('modal');
  const modalImg = document.getElementById('modal-img');
  const resImg = document.getElementById('result-img');
  const openModal = () => { if (!modal || !resImg || !modalImg) return; modalImg.src = resImg.src; modal.classList.add('show'); };
  if (viewBtn) viewBtn.addEventListener('click', openModal);
  if (resImg) resImg.addEventListener('click', openModal);
  modal && modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.remove('show'); });
}); 
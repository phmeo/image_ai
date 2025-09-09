async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (k === 'class') e.className = v;
    else if (k === 'html') e.innerHTML = v;
    else e.setAttribute(k, v);
  });
  children.forEach((c) => e.appendChild(c));
  return e;
}

async function refreshHistory() {
  const list = document.getElementById('history');
  list.innerHTML = 'Loading...';
  try {
    const items = await fetchJSON('/api/history');
    if (!items.length) {
      list.innerHTML = '<div class="card">No history yet.</div>';
      return;
    }
    const container = el('div', { class: 'list' });
    items.forEach((it) => {
      const card = el('div', { class: 'card' });
      card.appendChild(el('div', { html: `<strong>#${it.id}</strong> — ${it.source_filename}` }));
      card.appendChild(el('div', { html: `<small>${it.created_at} — ${it.model} — ${it.duration_ms}ms</small>` }));
      const link = el('a', { href: it.output_url, target: '_blank' }, [document.createTextNode('Open output')]);
      card.appendChild(link);
      container.appendChild(card);
    });
    list.innerHTML = '';
    list.appendChild(container);
  } catch (e) {
    list.innerHTML = 'Failed to load history.';
  }
}

function renderResult(payload) {
  const result = document.getElementById('result');
  result.innerHTML = '';
  const info = el('div', { class: 'card' });
  info.appendChild(el('div', { html: `<strong>Model:</strong> ${payload.model} — <strong>Time:</strong> ${payload.duration_ms}ms` }));
  if (payload.classes && payload.classes.length) {
    const pairs = payload.classes.map((c, i) => `${c} (${(payload.confs?.[i] ?? 0).toFixed(2)})`).join(', ');
    info.appendChild(el('div', { html: `<strong>Detections:</strong> ${pairs}` }));
  }
  result.appendChild(info);

  const isVideo = /\.(mp4|mov|avi|mkv|webm)$/i.test(payload.output_url);
  if (isVideo) {
    const video = el('video', { controls: true });
    video.src = payload.output_url;
    result.appendChild(video);
  } else if (payload.output_url) {
    const img = el('img', { src: payload.output_url, alt: 'Detection result' });
    result.appendChild(img);
  }
}

function attachWebcamControls() {
  const img = document.getElementById('webcam-stream');
  const startBtn = document.getElementById('start-webcam');
  const stopBtn = document.getElementById('stop-webcam');
  const confEl = document.getElementById('cam-conf');
  const iouEl = document.getElementById('cam-iou');

  if (!img || !startBtn || !stopBtn) return;

  startBtn.addEventListener('click', () => {
    const conf = Number(confEl.value || 0.35);
    const iou = Number(iouEl.value || 0.45);
    const url = `/webcam?conf=${encodeURIComponent(conf)}&iou=${encodeURIComponent(iou)}&_t=${Date.now()}`;
    img.src = url;
  });

  stopBtn.addEventListener('click', () => {
    img.src = '';
  });
}

function attachEmotionControls() {
  const form = document.getElementById('emotion-form');
  const status = document.getElementById('emotion-status');
  const result = document.getElementById('emotion-result');
  const submit = document.getElementById('emotion-submit');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    if (!fd.get('file')) return;
    submit.disabled = true;
    status.textContent = 'Analyzing emotion...';
    try {
      const res = await fetchJSON('/api/emotion_detect', { method: 'POST', body: fd });
      result.innerHTML = '';
      const info = el('div', { class: 'card' });
      info.appendChild(el('div', { html: `<strong>Model:</strong> ${res.model} — <strong>Time:</strong> ${res.duration_ms}ms` }));
      if (res.faces && res.faces.length) {
        const list = el('ul');
        res.faces.forEach((f, idx) => {
          const li = el('li', { html: `Face ${idx + 1}: ${f.emotion} (${(f.confidence ?? 0).toFixed(2)})` });
          list.appendChild(li);
        });
        info.appendChild(list);
      }
      result.appendChild(info);
      if (res.output_url) {
        const img = el('img', { src: res.output_url, alt: 'Emotion result' });
        result.appendChild(img);
      }
      status.textContent = '';
      await refreshHistory();
    } catch (err) {
      console.error(err);
      status.textContent = 'Error: ' + err.message;
    } finally {
      submit.disabled = false;
      form.reset();
    }
  });
}

function attachEmotionWebcamControls() {
  const img = document.getElementById('emotion-webcam-stream');
  const start = document.getElementById('start-emotion-webcam');
  const stop = document.getElementById('stop-emotion-webcam');
  if (!img || !start || !stop) return;
  start.addEventListener('click', () => {
    img.src = `/webcam_emotion?_t=${Date.now()}`;
  });
  stop.addEventListener('click', () => {
    img.src = '';
  });
}

async function main() {
  const form = document.getElementById('detect-form');
  const status = document.getElementById('status');
  const submit = document.getElementById('submit');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    if (!fd.get('file')) return;
    submit.disabled = true;
    status.textContent = 'Running detection...';
    try {
      const res = await fetchJSON('/api/detect', { method: 'POST', body: fd });
      renderResult(res);
      status.textContent = '';
      await refreshHistory();
    } catch (err) {
      console.error(err);
      status.textContent = 'Error: ' + err.message;
    } finally {
      submit.disabled = false;
      form.reset();
    }
  });

  attachWebcamControls();
  attachEmotionControls();
  attachEmotionWebcamControls();
  await refreshHistory();
}

window.addEventListener('DOMContentLoaded', main); 
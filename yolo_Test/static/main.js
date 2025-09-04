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
  } else {
    const img = el('img', { src: payload.output_url, alt: 'Detection result' });
    result.appendChild(img);
  }
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

  await refreshHistory();
}

window.addEventListener('DOMContentLoaded', main); 
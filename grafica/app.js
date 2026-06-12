/* ================================================================
   NetMind — app.js
   API base: http://127.0.0.1:8000/

   Campos exactos que espera el backend (FastAPI):
     POST /diagnose       → Form: prompt (str), image (file, opcional)
     POST /diagnose-web   → Form: prompt (str), image (file, opcional)
     POST /feedback       → Form: prompt (str), respuesta (str), titulo (str, opcional)
     POST /train          → sin body
   GET  /status           → keys: status, device, modelo_entrenado,
                                   casos_en_dataset, mensaje
   ================================================================ */

const API = 'https://ildergutierrez12-netmind-802-api.hf.space';

// ─── NAVIGATION ─────────────────────────────────────────────────
const navItems    = document.querySelectorAll('.nav-item');
const sections    = document.querySelectorAll('.section');
const topbarTitle = document.getElementById('topbarTitle');

const sectionTitles = {
  'diagnose':     'Diagnóstico Local',
  'diagnose-web': 'Diagnóstico con Búsqueda Web',
  'feedback':     'Enviar Caso Correcto',
  'train':        'Re-entrenar Modelo',
};

navItems.forEach(item => {
  item.addEventListener('click', () => {
    const target = item.dataset.section;
    navItems.forEach(n => n.classList.remove('active'));
    sections.forEach(s => s.classList.remove('active'));
    item.classList.add('active');
    document.getElementById(`section-${target}`).classList.add('active');
    topbarTitle.textContent = sectionTitles[target] || '';
  });
});

// ─── SERVER STATUS ───────────────────────────────────────────────
const statusDot   = document.getElementById('statusDot');
const statusLabel = document.getElementById('statusLabel');
const statCases   = document.getElementById('statCases');
const statModel   = document.getElementById('statModel');

async function checkStatus() {
  try {
    const res  = await fetch(`${API}/status`, { signal: AbortSignal.timeout(4000) });
    const data = await res.json();

    statusDot.className      = 'status-dot online';
    statusLabel.textContent  = 'En línea';

    // Claves exactas del backend: casos_en_dataset, modelo_entrenado
    statCases.textContent = data.casos_en_dataset ?? '—';
    statModel.textContent = data.modelo_entrenado ? 'listo ✓' : 'sin entrenar';
  } catch {
    statusDot.className      = 'status-dot offline';
    statusLabel.textContent  = 'Sin conexión';
    statCases.textContent    = '—';
    statModel.textContent    = '—';
  }
}

document.getElementById('btnRefresh').addEventListener('click', checkStatus);
checkStatus();
setInterval(checkStatus, 30_000);

// ─── DROPZONE HELPERS ────────────────────────────────────────────
function setupDropzone(dropzoneId, inputId, previewId, previewImgId, innerId) {
  const zone        = document.getElementById(dropzoneId);
  const input       = document.getElementById(inputId);
  const previewWrap = document.getElementById(previewId);
  const previewImg  = document.getElementById(previewImgId);
  const inner       = document.getElementById(innerId);

  function showPreview(file) {
    if (!file || !file.type.startsWith('image/')) return;
    previewImg.src = URL.createObjectURL(file);
    inner.classList.add('hidden');
    previewWrap.classList.remove('hidden');
  }

  input.addEventListener('change', () => {
    if (input.files[0]) showPreview(input.files[0]);
  });

  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('dragover');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) {
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      showPreview(file);
    }
  });
}

document.querySelectorAll('.remove-img').forEach(btn => {
  btn.addEventListener('click', () => {
    const input   = document.getElementById(btn.dataset.target);
    const preview = document.getElementById(btn.dataset.preview);
    const inner   = document.getElementById(btn.dataset.inner);
    input.value   = '';
    preview.classList.add('hidden');
    inner.classList.remove('hidden');
  });
});

setupDropzone('dropzoneDiagnose', 'fileDiagnose', 'previewDiagnose', 'previewImgDiagnose', 'dropzoneDiagnoseInner');
setupDropzone('dropzoneWeb',      'fileWeb',      'previewWeb',      'previewImgWeb',      'dropzoneWebInner');

// ─── RESULT RENDERER ─────────────────────────────────────────────
/**
 * Muestra la respuesta del servidor en el card de resultados.
 * Para /diagnose y /diagnose-web el backend devuelve objetos con
 * las keys: respuesta, respuesta_modelo, contexto_web, mensaje, detail.
 */
function showResult(cardId, data, isError = false) {
  const card = document.getElementById(cardId);
  card.classList.remove('hidden');

  const badgeClass = isError ? 'error' : 'ok';
  const badgeText  = isError ? 'ERROR' : 'OK';

  let html = '';

  if (typeof data === 'string') {
    // Respuesta de texto plano
    html = `<div class="result-body">${escapeHtml(data)}</div>`;

  } else if (data && typeof data === 'object') {

    if (isError) {
      // Errores de FastAPI vienen como { detail: "..." }
      const msg = data.detail ?? JSON.stringify(data, null, 2);
      html = `<div class="result-body result-error">${escapeHtml(String(msg))}</div>`;

    } else if (data.respuesta) {
      // POST /diagnose
      html  = buildBlock('🔍 Diagnóstico', data.respuesta);
      html += buildMeta(data);

    } else if (data.respuesta_modelo) {
      // POST /diagnose-web
      html  = buildBlock('🤖 Modelo local', data.respuesta_modelo);
      if (data.contexto_web) {
        html += buildBlock('🌐 Contexto web', data.contexto_web);
      }
      html += buildMeta(data);

    } else if (data.mensaje) {
      // POST /feedback o /train
      html = `<div class="result-body">${escapeHtml(data.mensaje)}`;
      if (data.id_caso)     html += `\n\n🆔 ID caso   : ${escapeHtml(data.id_caso)}`;
      if (data.total_casos !== undefined)
                             html += `\n📦 Total casos: ${data.total_casos}`;
      if (data.sugerencia)  html += `\n\n💡 ${escapeHtml(data.sugerencia)}`;
      if (data.modelo_path) html += `\n\n📁 ${escapeHtml(data.modelo_path)}`;
      html += '</div>';

    } else {
      // Fallback: JSON completo
      html = `<div class="result-body"><pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre></div>`;
    }
  }

  card.innerHTML = `
    <div class="result-header">
      <span class="result-title">Respuesta del servidor</span>
      <span class="result-badge ${badgeClass}">${badgeText}</span>
    </div>
    ${html}
  `;
}

function buildBlock(title, content) {
  return `
    <div class="result-section-title">${escapeHtml(title)}</div>
    <div class="result-body">${escapeHtml(String(content))}</div>
  `;
}

function buildMeta(data) {
  const parts = [];
  if (data.device)       parts.push(`Dispositivo: ${data.device}`);
  if (data.tiene_imagen !== undefined) parts.push(`Imagen: ${data.tiene_imagen ? 'sí' : 'no'}`);
  if (data.fuente)       parts.push(`Fuente: ${data.fuente}`);
  if (!parts.length) return '';
  return `<div class="result-meta">${parts.map(escapeHtml).join(' · ')}</div>`;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;');
}

// ─── LOADING STATE ────────────────────────────────────────────────
function setLoading(btnId, loading) {
  const btn    = document.getElementById(btnId);
  const text   = btn.querySelector('.btn-text');
  const loader = btn.querySelector('.btn-loader');
  btn.disabled = loading;
  text.classList.toggle('hidden', loading);
  loader.classList.toggle('hidden', !loading);
}

// ─── POST HELPER ─────────────────────────────────────────────────
async function postAPI(endpoint, formData) {
  const res = await fetch(`${API}${endpoint}`, {
    method: 'POST',
    body: formData,
    // No poner Content-Type: el navegador pone el boundary de multipart
  });
  const ct   = res.headers.get('content-type') || '';
  const data = ct.includes('application/json') ? await res.json() : await res.text();
  return { ok: res.ok, data };
}

// ─── DIAGNOSE LOCAL ──────────────────────────────────────────────
// Backend espera Form field: prompt
document.getElementById('btnDiagnose').addEventListener('click', async () => {
  const prompt = document.getElementById('diagnoseText').value.trim();
  if (!prompt) { alert('Por favor escribe una descripción del problema.'); return; }

  setLoading('btnDiagnose', true);
  try {
    const fd = new FormData();
    fd.append('prompt', prompt);                               // ← clave exacta del backend
    const imgFile = document.getElementById('fileDiagnose');
    if (imgFile.files[0]) fd.append('image', imgFile.files[0]);

    const { ok, data } = await postAPI('/diagnose', fd);
    showResult('resultDiagnose', data, !ok);
  } catch (err) {
    showResult('resultDiagnose', `No se pudo conectar con el servidor.\n${err.message}`, true);
  } finally {
    setLoading('btnDiagnose', false);
    checkStatus();
  }
});

// ─── DIAGNOSE WEB ────────────────────────────────────────────────
// Backend espera Form field: prompt
document.getElementById('btnDiagnoseWeb').addEventListener('click', async () => {
  const prompt = document.getElementById('diagnoseWebText').value.trim();
  if (!prompt) { alert('Por favor escribe una descripción del problema.'); return; }

  setLoading('btnDiagnoseWeb', true);
  try {
    const fd = new FormData();
    fd.append('prompt', prompt);                               // ← clave exacta del backend
    const imgFile = document.getElementById('fileWeb');
    if (imgFile.files[0]) fd.append('image', imgFile.files[0]);

    const { ok, data } = await postAPI('/diagnose-web', fd);
    showResult('resultDiagnoseWeb', data, !ok);
  } catch (err) {
    showResult('resultDiagnoseWeb', `No se pudo conectar con el servidor.\n${err.message}`, true);
  } finally {
    setLoading('btnDiagnoseWeb', false);
    checkStatus();
  }
});

// ─── FEEDBACK ────────────────────────────────────────────────────
// Backend espera Form fields: prompt (req), respuesta (req), titulo (opt)
document.getElementById('btnFeedback').addEventListener('click', async () => {
  const prompt    = document.getElementById('fbPrompt').value.trim();
  const respuesta = document.getElementById('fbRespuesta').value.trim();
  const titulo    = document.getElementById('fbTitulo').value.trim();

  if (!prompt || !respuesta) {
    alert('El prompt y la respuesta correcta son obligatorios.');
    return;
  }

  setLoading('btnFeedback', true);
  try {
    const fd = new FormData();
    fd.append('prompt',    prompt);                            // ← clave exacta
    fd.append('respuesta', respuesta);                        // ← clave exacta
    if (titulo) fd.append('titulo', titulo);                  // ← clave exacta (opcional)

    const { ok, data } = await postAPI('/feedback', fd);
    showResult('resultFeedback', data, !ok);

    if (ok) {
      document.getElementById('fbPrompt').value    = '';
      document.getElementById('fbRespuesta').value = '';
      document.getElementById('fbTitulo').value    = '';
    }
  } catch (err) {
    showResult('resultFeedback', `No se pudo conectar con el servidor.\n${err.message}`, true);
  } finally {
    setLoading('btnFeedback', false);
    checkStatus();
  }
});

// ─── TRAIN ───────────────────────────────────────────────────────
document.getElementById('btnTrain').addEventListener('click', async () => {
  if (!confirm('¿Iniciar un nuevo ciclo de entrenamiento?')) return;

  setLoading('btnTrain', true);
  try {
    const { ok, data } = await postAPI('/train', new FormData());
    showResult('resultTrain', data, !ok);
  } catch (err) {
    showResult('resultTrain', `No se pudo conectar con el servidor.\n${err.message}`, true);
  } finally {
    setLoading('btnTrain', false);
    checkStatus();
  }
});
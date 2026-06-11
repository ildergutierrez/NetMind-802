/* ================================================================
   NetMind — upload-app.js
   Interfaz de carga de datos de entrenamiento
   API base: http://127.0.0.1:8000
   ================================================================ */

const API = 'http://127.0.0.1:8000';
let newCasesCount = 0; // casos enviados en esta sesión

// ─── TABS ────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => {
      p.classList.remove('active');
      p.classList.add('hidden');
    });
    tab.classList.add('active');
    const panel = document.getElementById(`tab-${tab.dataset.tab}`);
    panel.classList.remove('hidden');
    panel.classList.add('active');
  });
});

// ─── SERVER STATUS ───────────────────────────────────────────────
const pillDot   = document.getElementById('pillDot');
const pillLabel = document.getElementById('pillLabel');
const pillCases = document.getElementById('pillCases');

async function checkStatus() {
  try {
    const res  = await fetch(`${API}/status`, { signal: AbortSignal.timeout(4000) });
    const data = await res.json();
    pillDot.className   = 'pill-dot online';
    pillLabel.textContent = 'En línea';
    pillCases.textContent = `· ${data.casos_en_dataset ?? '?'} casos`;
  } catch {
    pillDot.className   = 'pill-dot offline';
    pillLabel.textContent = 'Sin conexión';
    pillCases.textContent = '';
  }
}
checkStatus();
setInterval(checkStatus, 30_000);

// ─── LOG ─────────────────────────────────────────────────────────
const logPanel = document.getElementById('logPanel');
const logBody  = document.getElementById('logBody');

function addLog(msg, type = 'info') {
  logPanel.classList.remove('hidden');
  const now   = new Date().toLocaleTimeString('es', { hour12: false });
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML = `<span class="log-time">${now}</span><span class="log-msg ${type}">${escHtml(msg)}</span>`;
  logBody.prepend(entry);
}

document.getElementById('btnClearLog').addEventListener('click', () => {
  logBody.innerHTML = '';
  logPanel.classList.add('hidden');
});

// ─── TRAIN BANNER ────────────────────────────────────────────────
const trainBanner = document.getElementById('trainBanner');
const trainCount  = document.getElementById('trainBannerCount');

function bumpNewCases(n = 1) {
  newCasesCount += n;
  trainCount.textContent = `${newCasesCount} caso(s) nuevo(s) pendientes de entrenamiento.`;
  trainBanner.style.display = 'flex';
}

setLoading2('btnTrainFromUpload', false);
document.getElementById('btnTrainFromUpload').addEventListener('click', async () => {
  setLoading2('btnTrainFromUpload', true);
  try {
    const res  = await fetch(`${API}/train`, { method: 'POST' });
    const data = await res.json();
    if (res.ok) {
      addLog('Re-entrenamiento completado. ' + (data.mensaje ?? ''), 'ok');
      newCasesCount = 0;
      trainBanner.style.display = 'none';
    } else {
      addLog('Error en entrenamiento: ' + (data.detail ?? JSON.stringify(data)), 'error');
    }
  } catch (err) {
    addLog('Sin conexión con el servidor: ' + err.message, 'error');
  } finally {
    setLoading2('btnTrainFromUpload', false);
    checkStatus();
  }
});

// ─── HELPER: enviar un caso vía POST /feedback ───────────────────
async function enviarCaso({ titulo, prompt, respuesta }) {
  const fd = new FormData();
  fd.append('prompt',    prompt.trim());
  fd.append('respuesta', respuesta.trim());
  if (titulo && titulo.trim()) fd.append('titulo', titulo.trim());

  const res  = await fetch(`${API}/feedback`, { method: 'POST', body: fd });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail ?? JSON.stringify(data));
  return data;
}

// ─── HELPERS GENÉRICOS ───────────────────────────────────────────
function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function setLoading2(btnId, loading) {
  const btn    = document.getElementById(btnId);
  if (!btn) return;
  const text   = btn.querySelector('.btn-text');
  const loader = btn.querySelector('.btn-loader');
  btn.disabled = loading;
  if (text)   text.classList.toggle('hidden', loading);
  if (loader) loader.classList.toggle('hidden', !loading);
}

function setupDropzone({ zoneId, inputId, innerId, loadedId, fileNameId, fileMetaId, removeId, onFile }) {
  const zone   = document.getElementById(zoneId);
  const input  = document.getElementById(inputId);
  const inner  = document.getElementById(innerId);
  const loaded = document.getElementById(loadedId);
  const fnEl   = document.getElementById(fileNameId);
  const fmEl   = document.getElementById(fileMetaId);
  const rmBtn  = document.getElementById(removeId);

  function showFile(file) {
    if (fnEl) fnEl.textContent = file.name;
    if (fmEl) fmEl.textContent = (file.size / 1024).toFixed(1) + ' KB';
    inner.classList.add('hidden');
    loaded.classList.remove('hidden');
    onFile(file);
  }

  input.addEventListener('change', () => { if (input.files[0]) showFile(input.files[0]); });

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) {
      const dt = new DataTransfer(); dt.items.add(file); input.files = dt.files;
      showFile(file);
    }
  });

  if (rmBtn) {
    rmBtn.addEventListener('click', () => {
      input.value = '';
      loaded.classList.add('hidden');
      inner.classList.remove('hidden');
    });
  }
}

// ================================================================
//  TAB 1 — CSV
// ================================================================

// Plantilla CSV descargable
document.getElementById('btnDownloadTemplate').addEventListener('click', () => {
  const csv = `titulo,prompt,respuesta\n"Fallo OSPF área backbone","El router no propaga rutas OSPF al área backbone","DIAGNÓSTICO: Fallo en adyacencia OSPF\n\nCAUSAS:\n1. Temporizadores Hello/Dead distintos\n\nSOLUCIÓN:\nrouter ospf 1\n network 192.168.1.0 0.0.0.255 area 0\n\nVERIFICACIÓN:\nshow ip ospf neighbor"\n"DHCP sin IPs a clientes","Los clientes no reciben IP del servidor DHCP en otra VLAN","DIAGNÓSTICO: Falta ip helper-address\n\nCAUSAS:\n1. Router no reenvía DHCP broadcast\n\nSOLUCIÓN:\ninterface Gi0/0\n ip helper-address 10.0.0.1\n\nVERIFICACIÓN:\nshow ip dhcp binding"\n`;
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'plantilla_netmind.csv';
  a.click();
});

let csvRows = [];

setupDropzone({
  zoneId: 'dropzoneCSV', inputId: 'fileCSV', innerId: 'dzCSVInner',
  loadedId: 'dzCSVLoaded', fileNameId: 'csvFileName', fileMetaId: 'csvFileMeta',
  removeId: 'csvRemove',
  onFile: parseCSV
});

function parseCSV(file) {
  const reader = new FileReader();
  reader.onload = e => {
    const text = e.target.result;
    csvRows = csvTextToRows(text);
    renderCSVTable(csvRows);
  };
  reader.readAsText(file, 'UTF-8');
}

/** Parsea CSV sencillo con soporte de comillas y separador , o ; */
function csvTextToRows(text) {
  const lines = text.split(/\r?\n/).filter(l => l.trim());
  if (lines.length < 2) return [];

  const sep   = lines[0].includes(';') ? ';' : ',';
  const header = parseCsvLine(lines[0], sep).map(h => h.toLowerCase().trim());

  const iT = header.indexOf('titulo');
  const iP = header.indexOf('prompt');
  const iR = header.indexOf('respuesta');

  if (iP === -1 || iR === -1) {
    alert('El CSV debe tener las columnas "prompt" y "respuesta".');
    return [];
  }

  return lines.slice(1).map((line, idx) => {
    const cols = parseCsvLine(line, sep);
    return {
      _id:      idx,
      titulo:   iT >= 0 ? (cols[iT] ?? '').trim() : '',
      prompt:   (cols[iP] ?? '').trim(),
      respuesta:(cols[iR] ?? '').trim(),
      status:   'pending'
    };
  }).filter(r => r.prompt && r.respuesta);
}

function parseCsvLine(line, sep) {
  const cols = [];
  let cur = '', inQ = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQ && line[i+1] === '"') { cur += '"'; i++; }
      else inQ = !inQ;
    } else if (ch === sep && !inQ) {
      cols.push(cur); cur = '';
    } else { cur += ch; }
  }
  cols.push(cur);
  return cols;
}

function renderCSVTable(rows) {
  const wrap  = document.getElementById('csvPreviewWrap');
  const tbody = document.getElementById('csvTbody');
  const count = document.getElementById('csvRowCount');

  if (!rows.length) { wrap.classList.add('hidden'); return; }

  count.textContent = rows.length;
  tbody.innerHTML   = '';
  rows.forEach(row => {
    const tr = document.createElement('tr');
    tr.id = `csv-row-${row._id}`;
    tr.innerHTML = `
      <td class="col-check"><input type="checkbox" class="csv-check" data-id="${row._id}" checked /></td>
      <td title="${escHtml(row.titulo)}">${escHtml(row.titulo || '—')}</td>
      <td title="${escHtml(row.prompt)}">${escHtml(row.prompt.substring(0,80))}${row.prompt.length>80?'…':''}</td>
      <td title="${escHtml(row.respuesta)}">${escHtml(row.respuesta.substring(0,80))}${row.respuesta.length>80?'…':''}</td>
      <td><span class="status-badge pending" id="csv-status-${row._id}">pendiente</span></td>
    `;
    tbody.appendChild(tr);
  });

  wrap.classList.remove('hidden');
  updateCSVSelectedCount();

  document.getElementById('csvCheckAll').addEventListener('change', e => {
    document.querySelectorAll('.csv-check').forEach(cb => cb.checked = e.target.checked);
    updateCSVSelectedCount();
  });
  document.querySelectorAll('.csv-check').forEach(cb =>
    cb.addEventListener('change', updateCSVSelectedCount));
}

function updateCSVSelectedCount() {
  const n = document.querySelectorAll('.csv-check:checked').length;
  document.getElementById('csvSelectedCount').textContent = `${n} seleccionados`;
}

document.getElementById('btnCSVSelectAll').addEventListener('click', () => {
  document.querySelectorAll('.csv-check').forEach(cb => cb.checked = true);
  document.getElementById('csvCheckAll').checked = true;
  updateCSVSelectedCount();
});
document.getElementById('btnCSVDeselectAll').addEventListener('click', () => {
  document.querySelectorAll('.csv-check').forEach(cb => cb.checked = false);
  document.getElementById('csvCheckAll').checked = false;
  updateCSVSelectedCount();
});

document.getElementById('btnSendCSV').addEventListener('click', async () => {
  const checked = [...document.querySelectorAll('.csv-check:checked')].map(cb => Number(cb.dataset.id));
  if (!checked.length) { alert('Selecciona al menos un caso.'); return; }

  setLoading2('btnSendCSV', true);
  let ok = 0, err = 0;

  for (const id of checked) {
    const row    = csvRows.find(r => r._id === id);
    const badge  = document.getElementById(`csv-status-${id}`);
    const trEl   = document.getElementById(`csv-row-${id}`);
    badge.className = 'status-badge sending'; badge.textContent = 'enviando…';
    try {
      await enviarCaso(row);
      badge.className = 'status-badge ok'; badge.textContent = '✓ enviado';
      trEl.classList.add('row-sent');
      addLog(`CSV → OK: ${row.titulo || row.prompt.substring(0,50)}`, 'ok');
      ok++;
    } catch (e) {
      badge.className = 'status-badge error'; badge.textContent = '✕ error';
      addLog(`CSV → ERROR: ${e.message}`, 'error');
      err++;
    }
  }

  setLoading2('btnSendCSV', false);
  addLog(`CSV: ${ok} enviados, ${err} errores.`, ok > 0 ? 'ok' : 'error');
  if (ok > 0) bumpNewCases(ok);
  checkStatus();
});

// ================================================================
//  TAB 2 — PDF
// ================================================================
let pdfFile = null;

setupDropzone({
  zoneId: 'dropzonePDF', inputId: 'filePDF', innerId: 'dzPDFInner',
  loadedId: 'dzPDFLoaded', fileNameId: 'pdfFileName', fileMetaId: 'pdfFileMeta',
  removeId: 'pdfRemove',
  onFile: f => {
    pdfFile = f;
    document.getElementById('pdfConfig').classList.remove('hidden');
    document.getElementById('pdfExtractedWrap').classList.add('hidden');
  }
});

document.getElementById('pdfSeparator').addEventListener('change', e => {
  const custom = document.getElementById('pdfSepCustom');
  custom.classList.toggle('hidden', e.target.value !== 'custom');
});

document.getElementById('btnExtractPDF').addEventListener('click', async () => {
  if (!pdfFile) { alert('Carga un archivo PDF primero.'); return; }

  setLoading2('btnExtractPDF', true);

  // Leemos el PDF como texto en el cliente usando FileReader
  // y dividimos por el separador elegido.
  // (Para extracción real el backend necesitaría un endpoint /extract-pdf;
  //  aquí lo procesamos en cliente como texto plano y dejamos al usuario editar)
  const reader = new FileReader();
  reader.onload = e => {
    const raw = e.target.result;
    const sepSel = document.getElementById('pdfSeparator').value;
    const sepCustom = document.getElementById('pdfSepCustom').value;
    const sep = sepSel === 'custom' ? sepCustom :
                sepSel === '\n\n'   ? '\n\n'     : sepSel;

    // Limpiar texto extraído (remover null bytes y caracteres raros de PDF)
    const clean = raw.replace(/\u0000/g, '').replace(/\r\n/g, '\n');
    const chunks = clean.split(sep).map(c => c.trim()).filter(c => c.length > 20);

    renderPDFCases(chunks);
    setLoading2('btnExtractPDF', false);
    addLog(`PDF procesado: ${chunks.length} secciones encontradas.`, 'info');
  };
  reader.onerror = () => {
    addLog('Error al leer el PDF en el cliente.', 'error');
    setLoading2('btnExtractPDF', false);
  };
  reader.readAsText(pdfFile, 'UTF-8');
});

let pdfCases = [];

function renderPDFCases(chunks) {
  const wrap       = document.getElementById('pdfExtractedWrap');
  const container  = document.getElementById('pdfCasesContainer');
  const countEl    = document.getElementById('pdfCaseCount');
  const tituloBase = document.getElementById('pdfTituloBase').value.trim() || 'Caso PDF';

  pdfCases = chunks.map((text, i) => ({
    _id: i, titulo: `${tituloBase} ${i+1}`, prompt: '', respuesta: text, status: 'pending'
  }));

  countEl.textContent = pdfCases.length;
  container.innerHTML = '';

  pdfCases.forEach(c => {
    const card = document.createElement('div');
    card.className = 'case-card';
    card.id = `pdf-card-${c._id}`;
    card.innerHTML = `
      <input type="checkbox" class="pdf-check" data-id="${c._id}" checked />
      <div class="case-card-body">
        <div class="case-title">
          <input type="text" class="field-input" id="pdf-titulo-${c._id}"
            value="${escHtml(c.titulo)}" placeholder="Título del caso" style="margin-bottom:8px;" />
        </div>
        <div class="case-fields">
          <div class="case-field">
            <label>Prompt (descripción del problema)</label>
            <textarea id="pdf-prompt-${c._id}" rows="2"
              placeholder="Escribe el prompt que describe el problema de red..."></textarea>
          </div>
          <div class="case-field">
            <label>Respuesta extraída del PDF</label>
            <textarea id="pdf-resp-${c._id}" rows="5">${escHtml(c.respuesta)}</textarea>
          </div>
        </div>
      </div>
      <div class="case-status">
        <span class="status-badge pending" id="pdf-status-${c._id}">pendiente</span>
      </div>
    `;
    container.appendChild(card);
  });

  // Update selection styling
  container.querySelectorAll('.pdf-check').forEach(cb => {
    cb.addEventListener('change', () => {
      const card = document.getElementById(`pdf-card-${cb.dataset.id}`);
      card.classList.toggle('selected', cb.checked);
      updatePDFSelectedCount();
    });
    const card = document.getElementById(`pdf-card-${cb.dataset.id}`);
    card.classList.toggle('selected', cb.checked);
  });

  updatePDFSelectedCount();
  wrap.classList.remove('hidden');
}

function updatePDFSelectedCount() {
  const n = document.querySelectorAll('.pdf-check:checked').length;
  document.getElementById('pdfSelectedCount').textContent = `${n} seleccionados`;
}

document.getElementById('btnPDFSelectAll').addEventListener('click', () => {
  document.querySelectorAll('.pdf-check').forEach(cb => {
    cb.checked = true;
    document.getElementById(`pdf-card-${cb.dataset.id}`)?.classList.add('selected');
  });
  updatePDFSelectedCount();
});
document.getElementById('btnPDFDeselectAll').addEventListener('click', () => {
  document.querySelectorAll('.pdf-check').forEach(cb => {
    cb.checked = false;
    document.getElementById(`pdf-card-${cb.dataset.id}`)?.classList.remove('selected');
  });
  updatePDFSelectedCount();
});

document.getElementById('btnSendPDF').addEventListener('click', async () => {
  const checked = [...document.querySelectorAll('.pdf-check:checked')].map(cb => Number(cb.dataset.id));
  if (!checked.length) { alert('Selecciona al menos un caso.'); return; }

  // Validate prompts are filled
  const missing = checked.filter(id => !document.getElementById(`pdf-prompt-${id}`)?.value.trim());
  if (missing.length) {
    alert(`Hay ${missing.length} caso(s) sin prompt. Por favor completa el campo "Prompt" de cada caso seleccionado.`);
    return;
  }

  setLoading2('btnSendPDF', true);
  let ok = 0, err = 0;

  for (const id of checked) {
    const titulo   = document.getElementById(`pdf-titulo-${id}`)?.value.trim() || '';
    const prompt   = document.getElementById(`pdf-prompt-${id}`)?.value.trim() || '';
    const respuesta= document.getElementById(`pdf-resp-${id}`)?.value.trim()   || '';
    const badge    = document.getElementById(`pdf-status-${id}`);

    badge.className = 'status-badge sending'; badge.textContent = 'enviando…';
    try {
      await enviarCaso({ titulo, prompt, respuesta });
      badge.className = 'status-badge ok'; badge.textContent = '✓ enviado';
      addLog(`PDF → OK: ${titulo || prompt.substring(0,50)}`, 'ok');
      ok++;
    } catch (e) {
      badge.className = 'status-badge error'; badge.textContent = '✕ error';
      addLog(`PDF → ERROR: ${e.message}`, 'error');
      err++;
    }
  }

  setLoading2('btnSendPDF', false);
  addLog(`PDF: ${ok} enviados, ${err} errores.`, ok > 0 ? 'ok' : 'error');
  if (ok > 0) bumpNewCases(ok);
  checkStatus();
});

// ================================================================
//  TAB 3 — MANUAL
// ================================================================
const manualPrompt   = document.getElementById('manualPrompt');
const manualRespuesta = document.getElementById('manualRespuesta');

manualPrompt.addEventListener('input', () => {
  document.getElementById('promptChars').textContent = manualPrompt.value.length;
});
manualRespuesta.addEventListener('input', () => {
  document.getElementById('respChars').textContent = manualRespuesta.value.length;
});

document.getElementById('btnManualClear').addEventListener('click', () => {
  document.getElementById('manualTitulo').value    = '';
  manualPrompt.value    = '';
  manualRespuesta.value = '';
  document.getElementById('promptChars').textContent = '0';
  document.getElementById('respChars').textContent   = '0';
  const res = document.getElementById('manualResult');
  res.className = 'result-inline hidden';
  res.textContent = '';
});

document.getElementById('btnManualSend').addEventListener('click', async () => {
  const titulo    = document.getElementById('manualTitulo').value.trim();
  const prompt    = manualPrompt.value.trim();
  const respuesta = manualRespuesta.value.trim();
  const resEl     = document.getElementById('manualResult');

  if (!prompt)    { showInlineResult(resEl, 'El campo Prompt es obligatorio.', 'error'); return; }
  if (!respuesta) { showInlineResult(resEl, 'La respuesta correcta es obligatoria.', 'error'); return; }

  setLoading2('btnManualSend', true);
  try {
    const data = await enviarCaso({ titulo, prompt, respuesta });
    showInlineResult(resEl, `✓ ${data.mensaje ?? 'Caso guardado.'} (ID: ${data.id_caso})`, 'ok');
    addLog(`Manual → OK: ${titulo || prompt.substring(0,50)}`, 'ok');
    bumpNewCases(1);
    checkStatus();
    // Limpiar tras éxito
    setTimeout(() => {
      document.getElementById('btnManualClear').click();
    }, 1800);
  } catch (e) {
    showInlineResult(resEl, `✕ Error: ${e.message}`, 'error');
    addLog(`Manual → ERROR: ${e.message}`, 'error');
  } finally {
    setLoading2('btnManualSend', false);
  }
});

function showInlineResult(el, msg, type) {
  el.textContent = msg;
  el.className   = `result-inline ${type}`;
}

// ================================================================
//  TAB 4 — BATCH JSON
// ================================================================
let batchData = null;

setupDropzone({
  zoneId: 'dropzoneBatch', inputId: 'fileBatch', innerId: 'dzBatchInner',
  loadedId: 'dzBatchLoaded', fileNameId: 'batchFileName', fileMetaId: null,
  removeId: 'batchRemove',
  onFile: file => {
    const reader = new FileReader();
    reader.onload = e => {
      document.getElementById('batchJSON').value = e.target.result;
      validateBatch();
    };
    reader.readAsText(file, 'UTF-8');
  }
});

function validateBatch() {
  const raw   = document.getElementById('batchJSON').value.trim();
  const panel = document.getElementById('batchValidate');
  const icon  = document.getElementById('validateIcon');
  const msg   = document.getElementById('validateMsg');
  const btn   = document.getElementById('btnSendBatch');

  panel.classList.remove('hidden');
  batchData = null;

  if (!raw) {
    icon.textContent = '⚠️'; msg.textContent = 'El campo está vacío.';
    btn.disabled = true; return;
  }

  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) throw new Error('El JSON debe ser un array [ ... ]');

    const invalid = parsed.filter(c => !c.prompt || !c.respuesta);
    if (invalid.length) {
      icon.textContent = '⚠️';
      msg.textContent  = `${invalid.length} objeto(s) sin "prompt" o "respuesta". Corrígelos antes de enviar.`;
      btn.disabled = true; return;
    }

    batchData    = parsed;
    icon.textContent = '✅';
    msg.textContent  = `${parsed.length} casos válidos listos para enviar.`;
    btn.disabled = false;
  } catch (e) {
    icon.textContent = '❌';
    msg.textContent  = `JSON inválido: ${e.message}`;
    btn.disabled = true;
  }
}

document.getElementById('btnValidateBatch').addEventListener('click', validateBatch);
document.getElementById('batchJSON').addEventListener('input', () => {
  document.getElementById('batchValidate').classList.remove('hidden');
});

document.getElementById('btnSendBatch').addEventListener('click', async () => {
  if (!batchData || !batchData.length) { alert('Valida el JSON primero.'); return; }

  setLoading2('btnSendBatch', true);
  const resEl = document.getElementById('batchResult');
  let ok = 0, err = 0;

  for (const caso of batchData) {
    try {
      await enviarCaso({
        titulo:    caso.titulo    ?? '',
        prompt:    caso.prompt,
        respuesta: caso.respuesta
      });
      addLog(`Batch → OK: ${caso.titulo || caso.prompt.substring(0,50)}`, 'ok');
      ok++;
    } catch (e) {
      addLog(`Batch → ERROR: ${e.message}`, 'error');
      err++;
    }
  }

  setLoading2('btnSendBatch', false);
  showInlineResult(
    resEl,
    `Proceso completado: ${ok} enviados correctamente, ${err} con error.`,
    ok > 0 ? 'ok' : 'error'
  );
  resEl.classList.remove('hidden');
  if (ok > 0) bumpNewCases(ok);
  checkStatus();
});

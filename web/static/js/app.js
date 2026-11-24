(() => {
  const $ = s => document.querySelector(s);
  const $$ = s => [...document.querySelectorAll(s)];
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
  const PREVIEW_ROWS = 6;
  const typeIcons = Object.fromEntries($$('.type-btn').map(b => [b.dataset.type, b.querySelector('img').src]));

  const state = { mode: 'scout', type: null, file: null, loading: false, result: null };

  // ---- Tabs ----
  $$('.tab').forEach(tab => tab.addEventListener('click', () => {
    state.mode = tab.dataset.mode;
    $$('.tab').forEach(t => t.setAttribute('aria-selected', t === tab));
    $('#mode-scout').hidden = state.mode !== 'scout';
    $('#mode-upload').hidden = state.mode !== 'upload';
    showError('');
    refresh();
  }));

  // ---- Type picker ----
  $$('.type-btn').forEach(btn => btn.addEventListener('click', () => {
    state.type = state.type === btn.dataset.type ? null : btn.dataset.type;
    $$('.type-btn').forEach(b => b.setAttribute('aria-pressed', b.dataset.type === state.type));
    refresh();
  }));

  // ---- Autocomplete slots ----
  const X_ICON = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.75" stroke-linecap="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>';
  $$('.slot').forEach(slot => {
    const input = slot.querySelector('input');
    const db = window.CARD_DB[slot.dataset.db] || [];
    const clear = Object.assign(document.createElement('button'), { type: 'button', className: 'clear-btn', innerHTML: X_ICON, hidden: true });
    clear.setAttribute('aria-label', 'Clear');
    const menu = Object.assign(document.createElement('div'), { className: 'suggest', hidden: true });
    slot.append(clear, menu);

    const render = () => {
      clear.hidden = !input.value;
      const q = input.value.trim().toLowerCase();
      let list = db.filter(([n]) => !q || n.toLowerCase().includes(q));
      if (slot.dataset.db === 'pokemon' && state.type)
        list = [...list.filter(c => c[1] === state.type), ...list.filter(c => c[1] !== state.type)];
      list = list.slice(0, 6);
      menu.innerHTML = list.length
        ? list.map(([n, sub]) => `<button type="button" data-name="${esc(n)}"><span>${esc(n)}</span><span class="sub">${typeIcons[sub] ? `<img src="${typeIcons[sub]}" alt="">` : ''}${esc(sub)}</span></button>`).join('')
        : '<div class="empty">No match in the card list.</div>';
    };
    input.addEventListener('focus', () => { render(); menu.hidden = false; });
    input.addEventListener('input', () => { render(); menu.hidden = false; });
    input.addEventListener('blur', () => setTimeout(() => (menu.hidden = true), 120));
    menu.addEventListener('mousedown', e => {
      const b = e.target.closest('button[data-name]');
      if (!b) return;
      e.preventDefault();
      input.value = b.dataset.name;
      menu.hidden = true;
      clear.hidden = false;
    });
    clear.addEventListener('click', () => { input.value = ''; clear.hidden = true; input.focus(); });
  });

  // ---- File upload + client-side preview ----
  const drop = $('#dropzone'), fileInput = $('#file-input');
  drop.addEventListener('click', () => fileInput.click());
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
  drop.addEventListener('drop', e => { e.preventDefault(); drop.classList.remove('drag'); readFile(e.dataTransfer.files[0]); });
  fileInput.addEventListener('change', () => { readFile(fileInput.files[0]); fileInput.value = ''; });
  $('#remove-file').addEventListener('click', () => setFile(null));

  async function readFile(f) {
    if (!f) return;
    const ext = f.name.split('.').pop().toLowerCase();
    if (!['csv', 'xlsx'].includes(ext)) return setFile(null, `"${f.name}" isn't a CSV or XLSX file.`);
    try {
      let rows;
      if (window.XLSX) {
        const wb = XLSX.read(await f.arrayBuffer(), { type: 'array' });
        rows = XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]], { header: 1, defval: '' });
      } else {
        rows = (await f.text()).split(/\r?\n/).filter(Boolean).map(l => l.split(','));
      }
      rows = rows.filter(r => r.some(c => String(c).trim()));
      if (rows.length < 2) throw new Error('empty');
      setFile({ raw: f, name: f.name, cols: rows[0].map(String), rows: rows.slice(1) });
    } catch {
      setFile(null, `Couldn't read "${f.name}". Check that it has a header row and at least one card.`);
    }
  }

  function setFile(file, err = '') {
    state.file = file;
    showError(err);
    drop.hidden = !!file;
    $('#file-info').hidden = !file;
    if (file) {
      $('#file-name').textContent = file.name;
      $('#file-summary').textContent = `${file.rows.length} rows · ${file.cols.length} columns`;
      $('#preview-head').innerHTML = '<tr>' + file.cols.map(c => `<th>${esc(c)}</th>`).join('') + '</tr>';
      $('#preview-body').innerHTML = file.rows.slice(0, PREVIEW_ROWS)
        .map(r => '<tr>' + file.cols.map((_, i) => `<td>${esc(r[i] ?? '')}</td>`).join('') + '</tr>').join('');
      $('#preview-more').textContent = file.rows.length > PREVIEW_ROWS ? `Showing ${PREVIEW_ROWS} of ${file.rows.length} rows` : '';
    }
    refresh();
  }

  function showError(msg) {
    $('#form-error').textContent = msg;
    $('#form-error').hidden = !msg;
  }

  // ---- Generate ----
  function refresh() {
    const scout = state.mode === 'scout';
    const ready = scout ? !!state.type : !!state.file;
    $('#generate').disabled = !ready || state.loading;
    $('#generate').textContent = state.loading ? 'Running model…' : 'Build my counter deck';
    $('#hint').textContent = ready
      ? (scout ? 'More known cards raise the model’s confidence.' : 'The full list gives the most accurate recommendation.')
      : (scout ? 'Pick the enemy type to continue.' : 'Upload a file to continue.');
  }

  $('#generate').addEventListener('click', async () => {
    state.loading = true; showError(''); refresh();
    try {
      let res;
      if (state.mode === 'scout') {
        res = await fetch('/api/recommend/scout', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            type: state.type,
            pokemon: $$('input[name="pokemon"]').map(i => i.value),
            trainer: $('input[name="trainer"]').value,
            item: $('input[name="item"]').value
          })
        });
      } else {
        const fd = new FormData();
        fd.append('file', state.file.raw);
        res = await fetch('/api/recommend/upload', { method: 'POST', body: fd });
      }
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Something went wrong.');
      renderResult(data);
    } catch (e) {
      showError(e.message);
    } finally {
      state.loading = false; refresh();
    }
  });

  function renderResult(r) {
    state.result = r;
    $('#res-icon').src = r.icon;
    $('#res-title').textContent = r.title;
    $('#res-reason').textContent = r.reason;
    $('#res-basis').innerHTML = r.basis.map(b => `<span class="tag tag-neutral">${esc(b)}</span>`).join('');
    $('#res-win').textContent = r.winPct == null ? '—' : r.winPct + '%';
    $('#res-conf').textContent = r.confidence;
    $('#res-groups').innerHTML = r.groups.map(g => `
      <div class="deck-col"><h4>${esc(g.name)}</h4>
        ${g.cards.map(c => `<div class="deck-row"><span class="n">${c.n}</span><span>${esc(c.name)}</span></div>`).join('')}
      </div>`).join('');
    $('#copy-list').textContent = 'Copy list';
    const section = $('#results');
    section.hidden = false;
    $('#res-bar').style.width = '0';
    requestAnimationFrame(() => requestAnimationFrame(() => ($('#res-bar').style.width = (r.winPct ?? 0) + '%')));
    window.scrollTo({ top: section.getBoundingClientRect().top + window.scrollY - 24, behavior: 'smooth' });
  }

  $('#copy-list').addEventListener('click', () => {
    if (!state.result) return;
    const txt = state.result.groups.map(g => g.name + '\n' + g.cards.map(c => `${c.n} ${c.name}`).join('\n')).join('\n\n');
    navigator.clipboard?.writeText(txt);
    $('#copy-list').textContent = 'Copied';
  });

  refresh();
})();

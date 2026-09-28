async function sendForm(form, url) {
  const fd = new FormData(form);
  const r = await fetch(url, {method:'POST', body:fd});
  const data = await r.json();
  if (!r.ok) throw new Error(JSON.stringify(data));
  return data;
}

document.getElementById('galleryForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const out = document.getElementById('galleryOut');
  try { out.textContent = JSON.stringify(await sendForm(e.target, '/api/v1/gallery/items'), null, 2); }
  catch(err){ out.textContent = String(err); }
});

document.getElementById('queryForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const status = document.getElementById('status');
  const results = document.getElementById('results');
  results.innerHTML='';
  try {
    const d = await sendForm(e.target, '/api/v1/query');
    status.className = d.refused ? 'bad' : 'good';
    status.textContent = d.refused ? `Отказ: уверенного совпадения нет (top=${d.top_score ?? '—'})` : `Совпадения найдены, top-score=${d.top_score.toFixed(4)}`;
    d.candidates.forEach((c, idx) => {
      const el = document.createElement('div'); el.className='result';
      el.innerHTML = `<div><strong>#${idx+1} ${c.image_id}</strong><br><small>${c.vehicle_id ?? 'vehicle_id скрыт/не задан'}</small></div><div>${c.score.toFixed(4)}</div>`;
      results.appendChild(el);
    });
  } catch(err){ status.className='bad'; status.textContent=String(err); }
});

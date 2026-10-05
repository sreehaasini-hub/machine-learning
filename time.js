// Re-fetches chart specs from /api/time-series whenever a filter changes.
const ids = {state: 'f-state', crop: 'f-crop', season: 'f-season'};
async function refresh() {
  const p = new URLSearchParams();
  for (const [k, id] of Object.entries(ids)) p.set(k, document.getElementById(id).value);
  try {
    const r = await fetch('/api/time-series?' + p);
    const d = await r.json();
    renderCharts(d.charts);
    document.getElementById('trend-notes').innerHTML = d.notes.map(n => `<li>${n}</li>`).join('');
  } catch (e) { console.error(e); }
}
Object.values(ids).forEach(id => document.getElementById(id).addEventListener('change', refresh));
document.getElementById('f-reset').addEventListener('click', () => {
  Object.values(ids).forEach(id => document.getElementById(id).value = ''); refresh();
});

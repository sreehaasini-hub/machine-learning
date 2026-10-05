// Renders every <div class="chart" data-chart="key"> from the JSON embedded by the server.
const PLOT_CFG = {responsive: true, displaylogo: false};
function renderCharts(specs) {
  document.querySelectorAll('.chart[data-chart]').forEach(el => {
    const s = specs[el.dataset.chart];
    if (s) Plotly.react(el, s.data, s.layout, PLOT_CFG);
  });
}
renderCharts(JSON.parse(document.getElementById('chart-data').textContent));
document.getElementById('menuBtn').addEventListener('click', () =>
  document.getElementById('sidebar').classList.toggle('open'));

const fs = require('fs');
const glob = require('glob');

function readReports(patterns) {
  const files = patterns.reduce((acc, p) => acc.concat(glob.sync(p)), []);
  return files.map(f => ({ path: f, json: JSON.parse(fs.readFileSync(f, 'utf8')) }));
}

function summarize(reportJson) {
  return {
    scenariosCreated: reportJson.aggregate?.scenariosCreated || 0,
    requestsCompleted: reportJson.aggregate?.requestsCompleted || 0,
    codes: reportJson.aggregate?.codes || {},
    latency: reportJson.aggregate?.latency || {}
  };
}

function aggregateAll(reports) {
  const combined = {
    totalRequests: 0,
    codes: {},
    latency: { min: Infinity, max: 0 },
    perReport: []
  };
  reports.forEach(r => {
    const s = summarize(r.json);
    combined.perReport.push({ file: r.path, summary: s });
    combined.totalRequests += s.requestsCompleted;
    Object.keys(s.codes).forEach(code => {
      combined.codes[code] = (combined.codes[code] || 0) + s.codes[code];
    });
    if (s.latency.min) combined.latency.min = Math.min(combined.latency.min, s.latency.min);
    if (s.latency.max) combined.latency.max = Math.max(combined.latency.max, s.latency.max);
  });
  if (combined.latency.min === Infinity) combined.latency.min = 0;
  return combined;
}

function writeHtmlReport(aggregate, outPath = './reports/summary.html') {
  const html = `<!doctype html>
<html>
<head><meta charset="utf-8"><title>Artillery Summary</title></head>
<body>
  <h1>Artillery Aggregate Report</h1>
  <p>Total requests: ${aggregate.totalRequests}</p>
  <h2>Status codes</h2>
  <ul>
    ${Object.keys(aggregate.codes).map(k => `<li>${k}: ${aggregate.codes[k]}</li>`).join('\n')}
  </ul>
  <h2>Latency</h2>
  <p>min: ${aggregate.latency.min} ms</p>
  <p>max: ${aggregate.latency.max} ms</p>
</body>
</html>`;
  fs.writeFileSync(outPath, html);
  console.log(`✅ Report written to ${outPath}`);
}

const reports = readReports(process.argv.slice(2));
const aggregate = aggregateAll(reports);
writeHtmlReport(aggregate);

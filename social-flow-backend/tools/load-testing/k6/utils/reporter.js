const fs = require('fs');
const glob = require('glob');

function readReports(patterns) {
  const files = patterns.reduce((acc, p) => acc.concat(glob.sync(p)), []);
  return files.map(f => ({ path: f, json: JSON.parse(fs.readFileSync(f, 'utf8')) }));
}

function summarize(report) {
  return {
    requests: report.metrics?.http_reqs?.count || 0,
    failures: report.metrics?.http_req_failed?.count || 0,
    latency95: report.metrics?.http_req_duration?.['p(95)'] || 0
  };
}

function aggregate(reports) {
  return reports.map(r => ({ file: r.path, summary: summarize(r.json) }));
}

const reports = readReports(process.argv.slice(2));
console.log('Aggregated k6 reports:', JSON.stringify(aggregate(reports), null, 2));

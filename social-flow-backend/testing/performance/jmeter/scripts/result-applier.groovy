/**
 * result-applier.groovy
 *
 * Example JSR223 Listener script that runs at the end of the test and
 * applies simple assertions (e.g., percent of errors < X, p95 latency < Y).
 *
 * This script reads the result CSV/JTL and computes metrics. It's a demonstration;
 * for robust CI gating parse the JMeter HTML report or the generated metrics export.
 */

import org.apache.jorphan.logging.LoggingManager
import groovy.json.JsonBuilder
import java.nio.file.Files
import java.nio.file.Paths
import groovy.xml.MarkupBuilder

def log = LoggingManager.getLoggerForClass()

// location of results - assume props contains results location
def resultsFile = props.getProperty('results_file') ?: vars.get('results_file') ?: './reports/results.jtl'
def outputDir = props.getProperty('report_output_dir') ?: './reports'

log.info("result-applier: processing resultsFile=${resultsFile}")

if (!Files.exists(Paths.get(resultsFile))) {
    log.warn("Results file not found: ${resultsFile}")
    return
}

// Simple parser: count total samples, errors and compute percent errors
def total = 0
def errors = 0
def latencies = []
Files.newBufferedReader(Paths.get(resultsFile)).eachLine { line ->
    // JTL may be XML or CSV depending on configuration. This expects CSV; adapt if XML JTL.
    if (line.trim().length() == 0) return
    if (line.startsWith('<?xml') || line.startsWith('<testResults')) return
    // crude CSV parse: split by comma, but JTL contains commas in fields - in production use proper CSV library.
    def parts = line.split(',')
    // sample typical CSV: timeStamp,elapsed,label,responseCode,threadName,success,bytes,grpThreads,allThreads,URL,Latency
    total++
    def success = parts.size() > 5 ? parts[5].trim() : 'true'
    def elapsed = parts.size() > 1 ? parts[1].toInteger() : 0
    latencies << elapsed
    if (success.toLowerCase() != 'true') {
        errors++
    }
}
def errorPct = total > 0 ? (errors * 100.0f / total) : 0
def p95 = 0
if (latencies.size() > 0) {
    latencies.sort()
    p95 = latencies[(latencies.size() * 95 / 100) as int]
}
def summary = ['total': total, 'errors': errors, 'errorPct': errorPct, 'p95': p95]
log.info("Summary: ${summary}")
// Optionally fail the test by setting a property or writing a file for CI to consume
def summaryJson = new JsonBuilder(summary).toPrettyString()
Files.write(Paths.get(outputDir, 'summary.json'), summaryJson.getBytes('UTF-8'))

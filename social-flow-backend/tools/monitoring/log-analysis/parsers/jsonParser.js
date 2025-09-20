/**
 * JSON log parser
 * Converts structured JSON logs into normalized objects.
 */
export function parseJSONLog(line) {
  try {
    const obj = JSON.parse(line);
    return {
      timestamp: obj.timestamp || new Date().toISOString(),
      level: obj.level || "INFO",
      message: obj.message || line,
      service: obj.service || "unknown",
      meta: obj
    };
  } catch {
    return null; // not valid JSON
  }
}

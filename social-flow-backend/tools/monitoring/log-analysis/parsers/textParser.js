import moment from "moment";
import config from "../config/log.config.json" assert { type: "json" };

/**
 * Text log parser
 * Uses regex & keyword matching to classify logs.
 */
export function parseTextLog(line) {
  let level = "INFO";
  if (config.errorKeywords.some(k => line.includes(k))) level = "ERROR";
  else if (config.warnKeywords.some(k => line.includes(k))) level = "WARN";

  return {
    timestamp: moment().format(config.timeFormat),
    level,
    message: line,
    service: "unknown",
    meta: {}
  };
}

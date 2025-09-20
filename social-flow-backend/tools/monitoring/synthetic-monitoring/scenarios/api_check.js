import axios from "axios";
import { getConfig } from "../utils/httpClient.js";

export default async function run() {
  const start = Date.now();
  try {
    const res = await axios.get(`${process.env.API_BASE}/health`, getConfig());
    const latency = Date.now() - start;
    return {
      name: "API Health Check",
      status: res.status === 200 ? "PASS" : "FAIL",
      latency,
      details: res.data
    };
  } catch (err) {
    return {
      name: "API Health Check",
      status: "FAIL",
      error: err.message
    };
  }
}

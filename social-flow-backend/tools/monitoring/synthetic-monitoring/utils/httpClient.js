import config from "../config/synthetic.config.json" assert { type: "json" };

export function getConfig() {
  return {
    headers: {
      Authorization: `Bearer ${process.env.API_KEY || ""}`
    },
    timeout: config.timeout
  };
}

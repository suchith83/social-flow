import axios from "axios";

const client = axios.create({
  baseURL: process.env.ELASTIC_URL,
  auth: {
    username: process.env.ELASTIC_USER,
    password: process.env.ELASTIC_PASS
  }
});

export async function fetchLogs(index = "logs-*", size = 100) {
  const { data } = await client.post(`/${index}/_search`, {
    size,
    sort: [{ "@timestamp": { order: "desc" } }],
    query: { match_all: {} }
  });
  return data.hits.hits.map(hit => hit._source);
}

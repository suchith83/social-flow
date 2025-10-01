# common/libraries/node/utils

Collection of advanced, reusable utility modules for Node.js services.

## Modules
- asyncPool: concurrent mapping over arrays/async-iterables
- asyncQueue: bounded async queue with workers
- retry: generic retry helper with backoff
- backoff: several backoff strategies (exponential, fibonacci)
- debounceThrottle: wrappers for debounce and throttle, promise-aware
- memoize: TTL + LRU memoization
- deep: deepClone, deepMerge, getPath, setPath
- guards: runtime assertions and guards
- env: safe environment variable parsing
- fs: atomic writes and JSON helpers
- time: high-resolution timing and sleep
- uuid: uuidv4 and shortId
- safeJson: safe stringify/parse for logs and external data

## Usage
```js
const utils = require('common/libraries/node/utils');

const { pool } = utils.asyncPool;
await pool(items, 10, async (item) => { ... });

const id = utils.uuid.uuidv4();

await utils.fs.writeJsonAtomic('/tmp/data.json', { foo: 'bar' });

const result = await utils.retry(async () => await makeNetworkCall(), { attempts: 5 });

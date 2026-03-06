(function (global) {
    const CSV_URL = 'spreadsheet_data.csv';
    const STORAGE_DATA_KEY = 'golfModel:bettingData:data';
    const STORAGE_META_KEY = 'golfModel:bettingData:meta';
    const STORAGE_VERSION = 'v1'; // bump when schema changes

    let dataPromise = null;
    let metaCache = null;

    function getStorageKey(key) {
        return `${key}:${STORAGE_VERSION}`;
    }

    function readFromStorage(key) {
        try {
            return localStorage.getItem(getStorageKey(key));
        } catch (_) {
            return null;
        }
    }

    function writeToStorage(key, value) {
        try {
            localStorage.setItem(getStorageKey(key), value);
            return true;
        } catch (_) {
            try {
                localStorage.removeItem(getStorageKey(key));
            } catch (_) {
                /* noop */
            }
            return false;
        }
    }

    function clearStorage() {
        try {
            localStorage.removeItem(getStorageKey(STORAGE_DATA_KEY));
            localStorage.removeItem(getStorageKey(STORAGE_META_KEY));
        } catch (_) {
            /* noop */
        }
    }

    async function fetchMeta() {
        if (metaCache) return metaCache;
        try {
            const response = await fetch(CSV_URL, { method: 'HEAD', cache: 'no-store' });
            if (!response.ok) return null;
            const meta = {
                lastModified: response.headers.get('Last-Modified'),
                etag: response.headers.get('ETag')
            };
            metaCache = meta;
            return meta;
        } catch (_) {
            return null;
        }
    }

    function metaToKey(meta) {
        if (!meta) return null;
        if (meta.etag) return `etag:${meta.etag}`;
        if (meta.lastModified) return `last-modified:${meta.lastModified}`;
        return null;
    }

    function getStoredData() {
        const raw = readFromStorage(STORAGE_DATA_KEY);
        const metaRaw = readFromStorage(STORAGE_META_KEY);
        if (!raw) return null;
        try {
            const parsed = JSON.parse(raw);
            const metaSnapshot = metaRaw ? JSON.parse(metaRaw) : null;
            return { data: parsed, metaSnapshot };
        } catch (_) {
            clearStorage();
            return null;
        }
    }

    function storeData(data, metaSnapshot) {
        const ok = writeToStorage(STORAGE_DATA_KEY, JSON.stringify(data));
        if (!ok) return;
        if (metaSnapshot) {
            writeToStorage(STORAGE_META_KEY, JSON.stringify(metaSnapshot));
        }
    }

    async function fetchAndParseCsv() {
        if (typeof Papa === 'undefined') {
            throw new Error('PapaParse is required before loading betting data.');
        }
        const response = await fetch(CSV_URL, { cache: 'no-store' });
        if (!response.ok) throw new Error(`Failed to fetch ${CSV_URL}: ${response.status}`);
        const csvText = await response.text();
        const parsed = Papa.parse(csvText, {
            header: true,
            skipEmptyLines: 'greedy',
            dynamicTyping: false,
            delimiter: ',',
            quoteChar: '"',
            escapeChar: '"',
            newline: '\n',
            transformHeader: header => header.trim()
        });
        return parsed.data;
    }

    async function internalLoad(forceRefresh) {
        const stored = getStoredData();
        const remoteMeta = await fetchMeta();
        const remoteMetaKey = metaToKey(remoteMeta);
        if (!forceRefresh && stored) {
            const storedMetaKey = metaToKey(stored.metaSnapshot);
            if (remoteMetaKey && storedMetaKey && remoteMetaKey === storedMetaKey) {
                return stored.data;
            }
            if (!remoteMetaKey) {
                return stored.data;
            }
        }

        const data = await fetchAndParseCsv();
        const metaSnapshot = remoteMeta || { storedAt: new Date().toISOString() };
        storeData(data, metaSnapshot);
        return data;
    }

    async function load(options = {}) {
        if (options.forceRefresh) {
            dataPromise = null;
            return internalLoad(true);
        }
        if (!dataPromise) {
            dataPromise = internalLoad(false).catch(err => {
                dataPromise = null;
                throw err;
            });
        }
        return dataPromise;
    }

    async function getMeta() {
        const remote = await fetchMeta();
        if (remote) return remote;
        const stored = getStoredData();
        if (stored && stored.metaSnapshot) return stored.metaSnapshot;
        return null;
    }

    function clearCache() {
        clearStorage();
        dataPromise = null;
        metaCache = null;
    }

    global.BettingData = {
        load,
        getMeta,
        clearCache
    };
})(window);

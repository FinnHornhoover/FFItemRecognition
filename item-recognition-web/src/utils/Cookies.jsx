/**
 * Get a cookie value by name.
 * @param {string} name - Cookie name
 * @returns {string|null} Cookie value or null
 */
export function getCookie(name) {
  const match = document.cookie.match(new RegExp('(^| )' + name + '=([^;]+)'));
  return match ? decodeURIComponent(match[2]) : null;
}

/**
 * Set a cookie.
 * @param {string} name - Cookie name
 * @param {string} value - Cookie value (will be URI-encoded)
 * @param {object} options - { maxAgeSeconds, path }
 */
export function setCookie(name, value, options = {}) {
  const { maxAgeSeconds = 365 * 24 * 60 * 60, path = '/' } = options;
  const encoded = encodeURIComponent(value);
  document.cookie = `${name}=${encoded}; path=${path}; max-age=${maxAgeSeconds}; SameSite=Lax`;
}

/** @param {string} name */
function deleteCookie(name) {
  document.cookie = `${name}=; path=/; max-age=0`;
}

/** Cookie value ~4KB max; leave margin for name and attributes. */
const MAX_COOKIE_VALUE_ENCODED_LEN = 3800;

/** First chunk; continuation keys are `retrobution_inventory_cont_1`, `retrobution_inventory_cont_2`, ... */
export const INVENTORY_COOKIE_NAME = 'retrobution_inventory';

const INVENTORY_COOKIE_BASE = 'retrobution_inventory';
const MAX_CONT_CHUNK_INDEX = 100;

function inventoryContCookieName(index) {
  return `${INVENTORY_COOKIE_BASE}_cont_${index}`;
}

function clearInventoryChunkCookies() {
  deleteCookie(INVENTORY_COOKIE_NAME);
  for (let i = 1; i <= MAX_CONT_CHUNK_INDEX; i++) {
    deleteCookie(inventoryContCookieName(i));
  }
}

/**
 * Split compact dump on item boundaries (4 "::"-separated fields per item) so chunks rejoin safely.
 * @param {string} dump
 * @returns {string[]}
 */
function splitDumpIntoChunks(dump) {
  if (!dump) return [];
  const parts = dump.split('::');
  if (parts.length % 4 !== 0) {
    if (encodeURIComponent(dump).length <= MAX_COOKIE_VALUE_ENCODED_LEN) return [dump];
    return [dump];
  }
  const items = [];
  for (let i = 0; i < parts.length; i += 4) {
    items.push(parts.slice(i, i + 4).join('::'));
  }
  const chunks = [];
  let group = [];
  for (const item of items) {
    const candidate = group.length === 0 ? item : `${group.join('::')}::${item}`;
    if (encodeURIComponent(candidate).length <= MAX_COOKIE_VALUE_ENCODED_LEN) {
      group.push(item);
    } else {
      if (group.length) {
        chunks.push(group.join('::'));
      }
      if (encodeURIComponent(item).length > MAX_COOKIE_VALUE_ENCODED_LEN) {
        chunks.push(item);
        group = [];
      } else {
        group = [item];
      }
    }
  }
  if (group.length) chunks.push(group.join('::'));
  return chunks;
}

/**
 * Join cookie chunks: each chunk is a full sequence of items; items are separated by "::".
 * Between chunks we must insert "::" so the last field of chunk N is not glued to the first field of chunk N+1.
 */
function joinInventoryChunks(main, getContPart) {
  let full = main;
  for (let i = 1; i <= MAX_CONT_CHUNK_INDEX; i++) {
    const part = getContPart(i);
    if (part == null || part === '') break;
    full += `::${part}`;
  }
  return full;
}

/**
 * Raw serialized inventory: compact (possibly split across `retrobution_inventory` + `retrobution_inventory_cont_*`),
 * or legacy JSON only in the main `retrobution_inventory` cookie. `parseInventoryCookie` accepts both.
 */
export function getInventorySerialized() {
  const main = getCookie(INVENTORY_COOKIE_NAME);
  if (main == null || main === '') return '';

  const trimmed = main.trimStart();
  if (trimmed.startsWith('[')) {
    return main;
  }
  return joinInventoryChunks(main, (i) => getCookie(inventoryContCookieName(i)));
}

/**
 * Persist serialized inventory across one or more cookies when over the per-cookie size limit.
 */
export function setInventorySerialized(dump) {
  clearInventoryChunkCookies();
  if (dump == null || dump === '') return;

  const chunks = splitDumpIntoChunks(dump);
  if (chunks.length === 0) return;

  setCookie(INVENTORY_COOKIE_NAME, chunks[0]);
  for (let i = 1; i < chunks.length; i++) {
    setCookie(inventoryContCookieName(i), chunks[i]);
  }
}

/**
 * Compact inventory cookie: type_id::item_id::qty::price::...
 * Each label is type_id::item_id (two "::"-separated segments); the cookie
 * repeats that pattern with quantity and price for every row.
 */

function parseInventoryCookieCompact(raw) {
  const parts = raw.split('::');
  if (parts.length === 0 || parts.length % 4 !== 0) return [];

  const items = [];
  for (let i = 0; i < parts.length; i += 4) {
    const typeId = parts[i];
    const itemId = parts[i + 1];
    const qty = parseInt(parts[i + 2], 10);
    const price = parts[i + 3];
    if (!typeId || !itemId || Number.isNaN(qty) || price == null || price === '') continue;
    items.push({
      label: `${typeId}::${itemId}`,
      quantity: qty,
      price,
    });
  }
  return items;
}

/**
 * @param {string|null} raw
 * @returns {{ label: string, quantity: number, price: string }[]}
 */
export function parseInventoryCookie(raw) {
  if (!raw) return [];
  const trimmed = raw.trim();
  if (!trimmed) return [];

  if (trimmed.startsWith('[')) {
    try {
      const parsed = JSON.parse(trimmed);
      if (!Array.isArray(parsed) || parsed.length === 0) return [];
      return parsed
        .map(({ label, quantity, price }) => ({
          label,
          quantity: typeof quantity === 'number' ? quantity : 1,
          price: typeof price === 'string' ? price : '5k',
        }))
        .filter((row) => row.label);
    } catch {
      return [];
    }
  }

  return parseInventoryCookieCompact(trimmed);
}

/**
 * @param {{ label: string, quantity: number, price: string }[]} rows
 * @returns {string}
 */
export function dumpInventoryCookie(rows) {
  if (!rows.length) return '';
  const chunks = [];
  for (const { label, quantity, price } of rows) {
    const segs = label.split('::');
    if (segs.length !== 2) continue;
    const [typeId, itemId] = segs;
    const q = typeof quantity === 'number' && !Number.isNaN(quantity) ? quantity : 1;
    const p = price != null && price !== '' ? String(price) : '5k';
    chunks.push(`${typeId}::${itemId}::${q}::${p}`);
  }
  return chunks.join('::');
}

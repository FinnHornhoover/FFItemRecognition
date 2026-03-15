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

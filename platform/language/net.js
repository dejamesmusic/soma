// where the language's requests go (Nov 26, /api/run). In the browser: this site's /api, with the signed-in token.
// The server's headless language (server/run.js) points it at logOS itself, with the caller's own credentials, so a
// command run over http can do exactly what its caller could, never more.
export const net = {
  base: '/api',
  headers() {
    let t = null; try { t = localStorage.getItem('logos_session_token'); } catch (_) { /* blocked */ }
    return t ? { Authorization: `Bearer ${t}` } : {};
  },
  writes: 0,          // requests that change something (the server's question replay checks it)
};

export function auth(init = {}) {
  const headers = { ...net.headers(), ...(init.headers || {}) };
  if (init.body && !headers['Content-Type']) headers['Content-Type'] = 'application/json';
  if (init.method && init.method !== 'GET') net.writes += 1;
  return { ...init, credentials: 'include', headers };
}

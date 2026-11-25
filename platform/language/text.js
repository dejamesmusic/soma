// The terminal's text kit (Nov 19, James's design): everything printed is strict ASCII in a monospaced grid; a segment
// with `cmd` is bold and is a command (a link); `href` is a link out of the site. Pages are functions of the width.

export const S = (t, o) => Object.assign({ t: String(t ?? '') }, o || {});
// bold: a command. language v1 (Nov 25): a command shows its slash wherever it's printed, so what you see is what
// you'd type; an answer to a question (y, n, yes, no) and a link named for its thing (a model's name) don't
const ANSWERS = /^(y|n|yes|no)$/i;
const commandLike = (t, c) => /^[a-z]/i.test(t) && !ANSWERS.test(t) && !ANSWERS.test(c)
  && c.toLowerCase().startsWith(t.split(/\s+/)[0].toLowerCase());
export const K = (t, cmd) => { const c = String(cmd ?? t); const shown = String(t ?? ''); return S(commandLike(shown, c) ? `/${shown}` : shown, { cmd: c }); };
export const X = (t, href) => S(t, { href });                       // bold: a link out
export const Dm = (t) => S(t, { cls: 'dm' });
export const Hi = (t) => S(t, { cls: 'hi' });
export const Am = (t) => S(t, { cls: 'am' });
export const line = (...xs) => xs.flat().filter((x) => x !== null && x !== undefined && x !== false)
  .map((x) => (typeof x === 'string' || typeof x === 'number' ? S(x) : x));
export const blank = () => [];
// 1.5.2 (Nov 22, James): an image or a video a model read or showed. show: draw it now (images by default); otherwise a
// link that draws it when clicked (videos by default). Only http(s) addresses; the terminal draws them, not ascii.
export const M = (m, show) => S('', { media: { kind: m.kind === 'video' ? 'video' : 'image', src: String(m.src || ''), alt: String(m.alt || ''), show: !!show } });

// strict ascii: typographic characters fold to their ascii forms, anything else becomes '?'
const FOLD = { '‘': "'", '’': "'", '“': '"', '”': '"', '–': '-', '—': '-', '…': '...',
  '·': '|', '→': '->', '←': '<-', '×': 'x', '≈': '~', ' ': ' ', '•': '*', '≤': '<=', '≥': '>=' };
// symbols and emoji (the app's status marks, a pasted smiley) have no ascii form: they drop out, where an unknown
// letter prints '?'
const SYMBOL = /[\p{Extended_Pictographic}\p{So}←-⇿⌀-⏿■-➿️‍]\s?/gu;
export function ascii(s) {
  return String(s ?? '').replace(SYMBOL, (m) => { const f = FOLD[m.trim()]; return f === undefined ? '' : f + m.slice(m.trim().length); }).normalize('NFKD').replace(/[̀-ͯ]/g, '')
    .replace(/[^\x09\x0a\x0d\x20-\x7e]/g, (c) => FOLD[c] ?? '?');
}

export const cut = (s, n) => (s.length > n ? s.slice(0, Math.max(0, n - 1)) + '~' : s);
export const pad = (s, n) => { s = cut(String(s), n); return s + ' '.repeat(Math.max(0, n - s.length)); };
export const rpad = (s, n) => { s = cut(String(s), n); return ' '.repeat(Math.max(0, n - s.length)) + s; };
const width = (c) => Math.min(c, 78);

export function wrap(segs, w, indent = '', hang) {
  const toks = [];
  for (const g of segs) for (const p of ascii(g.t).split(/(\s+)/)) if (p) toks.push({ ...g, t: p });
  const first = indent; const rest = hang ?? indent;
  const out = []; let cur = first ? [S(first)] : []; let lead = first.length; let w0 = lead;
  const flush = () => { out.push(cur); cur = rest ? [S(rest)] : []; lead = rest.length; w0 = lead; };
  for (const tk of toks) {
    if (/^\s+$/.test(tk.t)) { if (w0 > lead && w0 + 1 <= w) { cur.push({ ...tk, t: ' ', cmd: undefined, href: undefined }); w0 += 1; } continue; }
    let t = tk.t;
    while (t.length > w - lead) { if (w0 > lead) flush(); cur.push({ ...tk, t: t.slice(0, w - lead) }); t = t.slice(w - lead); flush(); }
    if (w0 + t.length > w && w0 > lead) { while (cur.length && cur[cur.length - 1].t === ' ') { cur.pop(); w0 -= 1; } flush(); }
    cur.push({ ...tk, t }); w0 += t.length;
  }
  if (cur.length > (rest ? 1 : 0) || out.length === 0) out.push(cur);
  return out;
}
export const P = (c, ...segs) => wrap(segs.flat().map((x) => (typeof x === 'string' ? S(x) : x)), width(c));
export const H = (c, title, note) => {
  const t = ascii(title).toUpperCase();
  const head = wrap([Hi(t), ...(note ? [Dm('   ' + note)] : [])], width(c));
  return [...head, line(Dm('='.repeat(Math.min(t.length, c))))];
};
export const kv = (k, v, w = 13) => line(Dm(pad(k, w)), ...(Array.isArray(v) ? v : [S(v)]));
export function buttons(c, items) {
  return wrap(items.filter(Boolean).flatMap((b, i, a) => [b[2] ? X('[ ' + b[0] + ' ]', b[2]) : K('[ ' + b[0] + ' ]', b[1]), S(i < a.length - 1 ? '  ' : '')]), c);
}

// columns: [{ head, w (fixed) | flex, get(row) -> segment or string, min (drop below this width) }]
export function table(c, rows, colsSpec) {
  const shown = colsSpec.filter((k) => !k.min || c >= k.min);
  const fixed = shown.filter((k) => !k.flex).reduce((a, k) => a + k.w, 0);
  const flexW = Math.max(10, c - fixed - 1);
  const ws = shown.map((k) => (k.flex ? flexW : k.w));
  const out = [line(Dm(shown.map((k, i) => pad(k.head, ws[i])).join('').trimEnd())), line(Dm('-'.repeat(Math.min(c, ws.reduce((a, b) => a + b, 0)))))];
  for (const r of rows) {
    const segs = [];
    shown.forEach((k, i) => {
      let g = k.get(r); if (typeof g !== 'object') g = S(g);
      const t = cut(ascii(g.t), ws[i] - (i < shown.length - 1 ? 1 : 0));
      segs.push({ ...g, t }); if (i < shown.length - 1) segs.push(S(' '.repeat(ws[i] - t.length)));
    });
    out.push(segs);
  }
  return out;
}

export function progress(frac, w = 20) {
  const n = Math.max(0, Math.min(w, Math.round(frac * w)));
  return '[' + '#'.repeat(n) + '.'.repeat(w - n) + ']';
}

// a curve as characters: points [[x, y]], y falling is good; labels on the y axis at the given values
export function plot(c, pts, { y0, y1, x1, ylabels, xlabels, xunit = '' }) {
  const W = Math.max(24, Math.min(64, c - 8)); const Hh = 9;
  const grid = Array.from({ length: Hh }, () => Array(W).fill(' '));
  for (let x = 0; x < W; x += 1) {
    const t = ((x + 0.5) / W) * x1; if (t < pts[0][0] || t > pts[pts.length - 1][0]) continue;
    const k = pts.findIndex((p) => p[0] >= t); const [a, b] = k > 0 ? [pts[k - 1], pts[k]] : [pts[0], pts[0]];
    const v = a[1] + (b[1] - a[1]) * ((t - a[0]) / Math.max(1e-9, b[0] - a[0]));
    const r = Math.round(((y1 - v) / (y1 - y0)) * (Hh - 1)); if (r >= 0 && r < Hh) grid[r][x] = '*';
  }
  const lab = {}; for (const v of ylabels) lab[Math.round(((y1 - v) / (y1 - y0)) * (Hh - 1))] = v.toFixed(1);
  const out = grid.map((row, r) => line(Dm(pad(lab[r] || '', 4) + ' |'), Hi(row.join('').replace(/ +$/, ''))));
  out.push(line(Dm('     +' + '-'.repeat(W))));
  let xl = '      ';
  for (const v of xlabels) { const pos = 6 + Math.round((v / x1) * W); xl = pad(xl, pos) + (v ? v + xunit : '0'); }
  out.push(line(Dm(xl)));
  return out;
}

// ---- markdown, printed: headings, paragraphs, lists, quotes, code, tables, links ----
function inline(text, linkFor) {
  const segs = []; const re = /(\[([^\]]+)\]\(([^)\s]+)\)|\*\*([^*]+)\*\*|`([^`]+)`|\*([^*\s][^*]*)\*|_([^_\s][^_]*)_)/g;
  let i = 0; let m;
  while ((m = re.exec(text))) {
    if (m.index > i) segs.push(S(text.slice(i, m.index)));
    if (m[2]) segs.push(linkFor(m[2], m[3]));
    else if (m[4]) segs.push(Hi(m[4]));
    else if (m[5]) segs.push(Hi(m[5]));
    else segs.push(S(m[6] || m[7]));
    i = m.index + m[0].length;
  }
  if (i < text.length) segs.push(S(text.slice(i)));
  return segs;
}
export function markdown(c, md, linkFor) {
  const w = width(c); const out = [];
  const lines = ascii(md).replace(/\r/g, '').split('\n');
  let para = []; let i = 0;
  const flushPara = () => { if (para.length) { out.push(...wrap(inline(para.join(' '), linkFor), w)); out.push(blank()); para = []; } };
  while (i < lines.length) {
    const l = lines[i];
    if (/^```/.test(l)) {
      flushPara(); i += 1;
      while (i < lines.length && !/^```/.test(lines[i])) { out.push(line(Dm('  ' + cut(lines[i], c - 2)))); i += 1; }
      i += 1; out.push(blank()); continue;
    }
    const h = l.match(/^(#{1,6})\s+(.*)$/);
    if (h) { flushPara(); out.push(...H(c, h[2].replace(/[*`]/g, ''))); out.push(blank()); i += 1; continue; }
    if (/^\s*\|.*\|\s*$/.test(l)) {
      flushPara(); const rows = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
        const cells = lines[i].trim().replace(/^\||\|$/g, '').split('|').map((x) => x.trim().replace(/\*\*/g, ''));
        if (!cells.every((x) => /^:?-{2,}:?$/.test(x))) rows.push(cells);
        i += 1;
      }
      const n = Math.max(...rows.map((r) => r.length));
      const ws = Array.from({ length: n }, (_, k) => Math.min(Math.max(...rows.map((r) => (r[k] || '').length)) + 2, Math.max(8, Math.floor(c / n))));
      rows.forEach((r, ri) => {
        out.push(line(ri === 0 ? Dm(r.map((x, k) => pad(x, ws[k])).join('').trimEnd()) : S(r.map((x, k) => pad(x, ws[k])).join('').trimEnd())));
        if (ri === 0) out.push(line(Dm('-'.repeat(Math.min(c, ws.reduce((a, b) => a + b, 0))))));
      });
      out.push(blank()); continue;
    }
    const li = l.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
    if (li) {
      flushPara();
      const bullet = /\d/.test(li[2]) ? li[2] : '-'; const ind = ' '.repeat(Math.min(6, li[1].length));
      const lead = ind + bullet + ' ';
      let body = li[3]; i += 1;
      while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*+]|\d+[.)])\s/.test(lines[i])) { body += ' ' + lines[i].trim(); i += 1; }
      out.push(...wrap(inline(body, linkFor), w, lead, ' '.repeat(lead.length)));
      if (!(i < lines.length && /^\s*([-*+]|\d+[.)])\s/.test(lines[i]))) out.push(blank());
      continue;
    }
    const q = l.match(/^>\s?(.*)$/);
    if (q) { flushPara(); out.push(...wrap(inline(q[1], linkFor).map((g) => (g.cmd || g.href ? g : { ...g, cls: 'dm' })), w, '| ', '| ')); i += 1; continue; }
    if (/^\s*(---|\*\*\*|___)\s*$/.test(l)) { flushPara(); out.push(line(Dm('-'.repeat(Math.min(c, 40))))); out.push(blank()); i += 1; continue; }
    if (/^!\[/.test(l)) { flushPara(); const im = l.match(/^!\[([^\]]*)\]\(([^)\s]+)\)/); if (im) out.push(line(X('[image: ' + (im[1] || 'figure') + ']', im[2]))); i += 1; continue; }
    if (!l.trim()) { flushPara(); i += 1; continue; }
    para.push(l.trim()); i += 1;
  }
  flushPara();
  while (out.length && !out[out.length - 1].length) out.pop();
  return out;
}

export const fmtBytes = (n) => {
  if (!n) return '0 B'; const u = ['B', 'kB', 'MB', 'GB', 'TB']; let i = 0; let v = n;
  while (v >= 1000 && i < u.length - 1) { v /= 1000; i += 1; }
  return `${v >= 100 || i === 0 ? Math.round(v) : v.toFixed(1)} ${u[i]}`;
};
export const fmtCount = (n) => (!n ? '0' : n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${Math.round(n / 1e6)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n));
export const fmtDate = (iso) => (iso ? ascii(new Date(iso).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })).toLowerCase() : '');
export const fmtDur = (s) => {
  s = Math.round(s || 0);
  if (s < 60) return `${s} s`; if (s < 3600) return `${Math.floor(s / 60)} min ${s % 60} s`;
  return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;
};

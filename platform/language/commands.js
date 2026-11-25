import HELP_ROWS from './help.json';
// logOS's commands (Nov 19). Each one prints a page; each page is ASCII; anything bold is another command. A command
// that is a place has a path, so the address bar follows the terminal and every address prints its page.
import {
  S, K, X, M, Dm, Hi, Am, line, blank, ascii, pad, cut, P, H, kv, buttons, table, progress, plot, markdown,
  fmtBytes, fmtCount, fmtDate, fmtDur, wrap,
} from './text';
import { runCloud } from './cloud';
import { libraryPage, libraryCommand } from './library';
import { uploadPage } from './upload';
import { loadCmd, thisComputer } from './places';
import { sendToMachine } from './transfer';
import { adminCommand } from './admin';
import { filesCommand, sessionFiles } from './files';
import { net, auth } from './net';

const LIVE = new Set(['soma.somatic_core']);
async function get(path) { const r = await fetch(`${net.base}${path}`, auth()); if (!r.ok) throw Object.assign(new Error((await r.json().catch(() => ({}))).detail || `logOS answered ${r.status}`), { status: r.status }); return r.json(); }
async function post(path, body) { const r = await fetch(`${net.base}${path}`, auth({ method: 'POST', body: body ? JSON.stringify(body) : undefined })); const d = await r.json().catch(() => ({})); if (!r.ok) throw Object.assign(new Error(d.detail || `logOS answered ${r.status}`), { status: r.status }); return d; }
async function request(path, method, body) { const r = await fetch(`${net.base}${path}`, auth({ method, body: body ? JSON.stringify(body) : undefined })); const d = await r.json().catch(() => ({})); if (!r.ok) throw new Error(d.detail || `logOS answered ${r.status}`); return d; }
async function text(path) { const r = await fetch(`${net.base}${path}`, auth()); if (!r.ok) throw new Error(`logOS answered ${r.status}`); return r.text(); }

// one terminal stream per account (Nov 25, James): what this terminal draws goes to logOS, and /history backfills
export async function keepStream(blocks) { try { await post('/me/terminal', { blocks }); } catch (_) { /* offline: the next flush tries again */ } }

const norm = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
function lev(a, b) {
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i]); for (let j = 1; j <= b.length; j += 1) d[0][j] = j;
  for (let i = 1; i <= a.length; i += 1) for (let j = 1; j <= b.length; j += 1) d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
  return d[a.length][b.length];
}
const macName = (n) => ascii(String(n || 'your machine').replace(/^soma( harness)? on /i, '')) || 'your machine';

// ---------- paths ----------
// a command's address, when it is a place (for a link's real href: cmd-click opens a new shell there); null if not
export function commandToPath(cmd) {
  const t = String(cmd || '').trim(); const low = t.toLowerCase(); let m;
  if (/^(home|~)$/.test(low)) return '/';
  if (/^(help|\?)$/.test(low)) return '/help';
  if (/^(models|ls)$/.test(low)) return '/models';
  if (/^(public models|browse)$/.test(low)) return '/models';
  if (/^(my models|library)$/.test(low)) return '/library';
  if ((m=t.match(/^library\s+(.+)$/i))) return `/library/${encodeURIComponent(m[1])}`;
  if ((m = t.match(/^search\s+(.+)$/i))) return `/models?q=${encodeURIComponent(m[1])}`;
  if ((m = t.match(/^open\s+([0-9a-f]{32})$/i))) return `/models/${m[1]}`;
  if (/^(developments|reading)$/.test(low)) return '/developments';
  if ((m = t.match(/^(?:developments|reading)\s+([a-z0-9][a-z0-9-]*[a-z])$/i))) return `/developments/${m[1]}`;
  if (/^runpod setup$/.test(low)) return '/runpod';
  if (/^settings$/.test(low)) return '/settings';
  if (/^(get soma|download soma)$/.test(low)) return '/download';
  if (/^(runtime|spec)$/.test(low)) return '/runtime';
  if ((m = low.match(/^runtime\s+([a-z0-9.-]+)$/))) return `/runtimes/${m[1]}`;
  if ((m = t.match(/^whois\s+@?(\S+)$/i))) return `/u/${m[1]}`;
  if (/^chat$/.test(low)) return '/library';
  if (/^machines$/.test(low)) return '/machines';
  if (/^(cloud|runpod)$/.test(low)) return '/cloud';
  if (/^link$/.test(low)) return '/link';
  if (/^upload$/.test(low)) return '/upload';
  if (/^admin$/.test(low)) return '/admin';
  if (/^(about|info)$/.test(low)) return `/${low}`;
  return null;
}

export function pathToCommand(pathname, search = '') {
  const p = decodeURIComponent(pathname.replace(/\/+$/, '')) || '/';
  const q = new URLSearchParams(search);
  let m;
  if (p === '/') return 'home';
  if (p === '/help') return 'help';
  if (p === '/models' || p === '/checkpoints' || p === '/creators' || p === '/feed/following') return q.get('q') || q.get('search') ? `search ${q.get('q') || q.get('search')}` : 'public models';
  if(p==='/library') return 'my models';
  if(p.startsWith('/library/')) return `library ${p.slice(9)}`;
  if ((m = p.match(/^\/(models|checkpoints)\/(.+)$/))) return `open ${m[2]}`;
  if (p === '/developments' || p === '/blog') return 'developments';
  if (p === '/runpod') return 'runpod setup';
  if (p === '/settings') return 'settings';
  if (p === '/paper') return 'read time-is-all-you-need';
  if ((m = p.match(/^\/developments\/(.+)$/))) return `developments ${m[1]}${q.get('version') ? `?version=${q.get('version')}` : ''}`;
  if (['/download', '/downloads', '/train', '/credits'].includes(p) || p.startsWith('/runs/')) return 'get soma';
  if (['/runtime', '/runtimes', '/specs', '/runtimes/current'].includes(p)) return 'runtime';
  if ((m = p.match(/^\/(runtimes|specs)\/(.+)$/))) return `runtime ${m[2]}`;
  if ((m = p.match(/^\/u\/(.+)$/))) return `whois ${m[1]}`;
  if (p === '/talk' || p === '/chat' || p === '/workspace' || p === '/data' || p.startsWith('/remote/')) return 'my models';
  if (p === '/machines') return 'machines';
  if (p === '/cloud') return 'cloud';
  if (p === '/cloud/history') return 'cloud history';
  if (p === '/link') return q.get('code') ? `link ${q.get('code')}` : 'link';
  if (p === '/upload') return 'upload';
  if (p === '/admin') return 'admin';
  if (p === '/about') return 'about';
  if (p === '/research' || p === '/lineage') return 'developments';      // retired Nov 25: old links
  if (p === '/privacy') return 'privacy';
  if (p === '/notices') return 'notices';
  if (p === '/info') return 'info';
  return `goto ${p}`;
}

// ---------- pages ----------
const BANNER = [' _              ___  ____', '| | ___   __ _ / _ \\/ ___|', '| |/ _ \\ / _` | | | \\___ \\', '| | (_) | (_| | |_| |___) |',
  '|_|\\___/ \\__, |\\___/|____/', '         |___/'];

function linkFor(ctx) {
  return (label, href) => {
    if (/^https?:\/\//.test(href) && !/^https?:\/\/(www\.)?logossoma\.com/.test(href)) return X(label, href);
    const path = href.replace(/^https?:\/\/(www\.)?logossoma\.com/, '');
    if (path.startsWith('/api/') || /\.(md|json|txt|pdf|zip|dmg|gz)$/.test(path)) return X(label, path);
    return K(label, pathToCommand(path.split('?')[0], path.includes('?') ? path.slice(path.indexOf('?')) : ''));
  };
}

function home(ctx) {
  // the home page (Nov 25, James): the title and the doors, nothing explained. /help has the words
  const u = ctx.user();
  const unseen = u ? ctx.st.unseen || 0 : 0;
  if (u) get('/me/notices?limit=50').then((n) => { ctx.st.unseen = n.unseen; }).catch(() => {});
  const doors = u ? ['my models', 'public models', 'developments', 'help'] : ['public models', 'developments', 'sign in', 'help'];
  return (c) => [
    ...(c >= 30 ? BANNER.map((l) => line(Hi(l))) : [line(Hi('logOS'))]), line(Dm('where minds meet')), blank(),
    ...(unseen ? [line(Am(`${unseen} new: `), K('notices')), blank()] : []),
    ...(doors.join('    ').length + doors.length < c ? [line(...doors.flatMap((k, i) => [K(k), ...(i < doors.length - 1 ? [S('    ')] : [])]))]
      : doors.map((k) => line(K(k)))),
  ];
}

function helpRows(ctx) {
  return [...HELP_ROWS, ...(ctx?.user?.()?.role === 'admin' ? [['admin, admin users, admin models', 'the numbers, the people, the models'], ['admin verify|unverify|promote|demote @name', 'a person'], ['admin delete <model id>', 'a model, for good'], ['/post, /post paste', 'publish a versioned development']] : [])];
}
function help(ctx) {
  const rows = helpRows(ctx);
  return (c) => [...H(c, 'help'), ...P(c, 'a command starts with /. bold is a command: click it or type it. anything else is written to your stream, or said to the model you are talking to. up and down: what you typed. tab completes. esc leaves. ctrl+l clears.'),
    ...rows.flatMap(([k, v]) => { if (k === null) return [blank(), line(Hi(v))];
      const w = Math.min(30, Math.floor(c * 0.44)); const first = k.split(',')[0].trim().replace(/\|\S+/g, '');
      // a command that takes something ("open <model|n>", "whois @name") is typed into the prompt for you to finish
      const cmd = k.startsWith('/sign') ? 'sign in' : /[<[]|@name/.test(first) ? first.replace(/\s*([<[]|@name).*$/, ' ') : first;
      if (c < 60) return [line('  ', K(k, cmd)), ...wrap([Dm(v)], c, '    ')];          // narrow: one beneath the other
      const col = ' '.repeat(w + 2);                                                  // descriptions wrap in their own column
      if (k.length > w - 1) return [line('  ', K(k, cmd)), ...wrap([Dm(v)], c, col)];   // long: the words beneath
      return wrap([Dm(v)], c - col.length).map((l, i) => (i ? line(S(col), ...l) : line('  ', K(k, cmd), S(' '.repeat(Math.max(1, w - k.length))), ...l))); })];
}

function modelRows(list) {
  return [
    { head: '#', w: 4, get: (r) => Dm(String(r._n)) },
    { head: 'model', flex: true, get: (r) => K(r.name || r.checkpoint_id, `open ${r.checkpoint_id}`) },
    { head: 'by', w: 16, min: 52, get: (r) => S('@' + (r.creator_username || '?')) },
    { head: 'size', w: 8, get: (r) => S(r.parameters ? fmtCount(r.parameters) : '') },
    { head: 'read', w: 9, min: 64, get: (r) => S(r.tokens_seen ? fmtCount(r.tokens_seen) : r.bytes_seen ? fmtBytes(r.bytes_seen) : '') },
    { head: 'stars', w: 6, min: 74, get: (r) => S(String(r.star_count || 0)) },
  ];
}

// search (Nov 23, James): the commands that match as well as the models: every word must appear in a command or what it does
function matchingCommands(ctx, q) {
  const words = String(q || '').toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  return helpRows(ctx).filter(([k, v]) => k !== null && words.every((w) => `${k} ${v}`.toLowerCase().includes(w)));
}
const runnable = (k) => {                          // the part of a help row you can click: up to its first placeholder
  const first = k.split(',')[0].trim();
  const cut = first.indexOf('<'); const bar = first.indexOf('[');
  const end = Math.min(...[cut, bar].filter((x) => x >= 0), first.length);
  return end < first.length ? first.slice(0, end) : first;
};
async function searchAll(ctx, q) {
  const cmds = matchingCommands(ctx, q);
  const shown = await models(ctx, q);
  return (c) => [
    ...(cmds.length ? [...H(c, `commands matching '${ascii(q)}'`, `${cmds.length}`),
      ...cmds.flatMap(([k, v]) => wrap([K(k, runnable(k)), Dm(`   ${v}`)], Math.min(c, 100), '  ', '      ')), blank()] : []),
    ...shown(c)];
}

async function models(ctx, q) {
  const params = new URLSearchParams({ sort: 'stars', limit: '60' }); if (q) params.set('search', q);
  const list = await get(`/checkpoints?${params}`);
  list.forEach((r, i) => { r._n = i + 1; });
  ctx.st.list = list.map((r) => ({ kind: 'model', id: r.checkpoint_id, name: r.name }));
  ctx.remember(list);
  return (c) => {
    const out = [...H(c, q ? `public models matching '${q}'` : 'public models', `${list.length} ${list.length === 1 ? 'snapshot' : 'snapshots'}`)];
    if (!list.length) return [...out, line(Dm('nothing matches. try '), K('models'), Dm(' for all of them.'))];
    out.push(...table(c, list, modelRows(list)));
    return out;
  };
}

async function findModel(ctx, q) {
  q = (q || '').trim();
  if (!q || /^(it|this)$/.test(q)) return ctx.st.here?.kind === 'model' ? ctx.st.here.id : null;
  if (/^\d+$/.test(q)) { const it = ctx.st.list[+q - 1]; if (it?.kind === 'model') return it.id; }
  if (/^[0-9a-f]{32}$/.test(q)) return q;
  const n = norm(q);
  const known = ctx.known();
  const hit = known.find((m) => norm(m.name) === n) || known.find((m) => norm(m.name).includes(n));
  if (hit) return hit.checkpoint_id;
  const found = await get(`/checkpoints?${new URLSearchParams({ search: q, limit: '5' })}`).catch(() => []);
  ctx.remember(found);
  return found[0]?.checkpoint_id || null;
}

async function open(ctx, q) {
  const id = await findModel(ctx, q);
  if (!id) return { render: notFound(ctx, `open ${q}`), path: null };
  const m = await get(`/checkpoints/${id}`);
  const revs = await get(`/checkpoints/${id}/revisions`).catch(() => []);
  ctx.st.here = { kind: 'model', id, name: m.name };
  ctx.remember([m]);
  const u = ctx.user();
  const shape = m.hidden_dim ? `${m.hidden_dim} x ${m.n_layers || m.config?.depth || '?'}` : '';
  return {
    path: `/models/${id}`,
    sugg: [`talk to ${m.name}`, m.starred_by_me ? 'unstar' : 'star', 'download', 'models'],
    render: (c) => [
      ...H(c, m.name, m.visibility && m.visibility !== 'public' ? m.visibility : ''),
      line(Dm('by '), K('@' + m.creator_username, `whois ${m.creator_username}`), m.starred_by_me ? Am('   * starred') : null),
      blank(),
      ...(m.description || m.og_tagline ? [...P(c, m.description || m.og_tagline), blank()] : []),
      ...buttons(c, [[m.visibility==='public'?'add to my models':'talk to it',m.visibility==='public'?`add ${m.checkpoint_id}`:`talk to ${m.name}`], [m.starred_by_me ? 'unstar' : 'star', m.starred_by_me ? 'unstar' : 'star'], ['download snapshot', 'download']]),
      ...(m.visibility==='public'?[line(Dm('frozen publication'))]:[]),
      blank(),
      m.parameters && kv('parameters', `${fmtCount(m.parameters)}${shape ? `  (${shape})` : ''}`),
      (m.tokens_seen || m.bytes_seen) && kv('read', [m.tokens_seen ? `${fmtCount(m.tokens_seen)} tokens` : '', m.bytes_seen ? `${fmtBytes(m.bytes_seen)} of text` : ''].filter(Boolean).join(', ')),
      kv('runtime', [S(`${(m.runtime_id || '').replace(/^soma\./, '').replace(/_/g, ' ')} ${m.runtime_version || ''}  `), LIVE.has(m.runtime_id) ? K('spec', 'runtime') : Dm('posthumous')]),
      m.file_size_bytes && kv('file', fmtBytes(m.file_size_bytes)),
      m.licence && kv('licence', m.licence),
      kv('stars', `${m.star_count || 0}   downloads ${m.download_count || 0}`),
      m.created_at && kv('since', fmtDate(m.created_at)),
      ...(m.corpus_tags?.length ? [kv('tags', m.corpus_tags.flatMap((t, i) => [K(t, `search ${t}`), S(i < m.corpus_tags.length - 1 ? ' ' : '')]))] : []),
      blank(), ...H(c, 'revisions', `${revs.length}`),
      ...(revs.length ? table(c, revs.slice(0, 10), [
        { head: 'saved', w: 14, get: (r) => S(fmtDate(r.created_at)) },
        { head: 'read', w: 12, get: (r) => S(r.bytes_seen ? fmtBytes(r.bytes_seen) : '') },
        { head: 'runtime', w: 10, min: 50, get: (r) => S(r.runtime_version || '') },
        { head: 'revision', flex: true, get: (r) => (r.head ? Hi(r.revision_id.slice(0, 12) + '  head') : Dm(r.revision_id.slice(0, 12))) },
      ]) : [line(Dm('  none yet'))]),
      ...(u ? [] : [blank(), line(Dm('star it, fork it or talk to it: '), K('sign in'))]),
    ].filter(Boolean),
  };
}

async function devList(ctx) {
  const d = await get('/developments');
  const list = d.developments || [];
  ctx.st.list = list.map((x) => ({ kind: 'dev', id: x.slug }));
  ctx.rememberDevs(list);
  return (c) => [...H(c, 'developments', 'the lab notebook, and the paper'),
    ...list.flatMap((x, i) => { const ls = wrap([K(x.title, `developments ${x.slug}`)], Math.min(c, 78) - 4, '    '); ls[0] = [Dm(pad(String(i + 1), 4)), ...ls[0].slice(1)];
      return [...ls, ...(x.occurred_on || x.published_at ? [line(Dm('    ' + fmtDate(x.occurred_on || x.published_at) + (x.verdict ? '  |  ' + cut(ascii(x.verdict), Math.max(10, c - 22)) : '')))] : [])]; })];
}

async function devPage(ctx, q) {
  const versionMatch = q.trim().match(/\?version=([1-9]\d*)$/);
  const version = versionMatch ? versionMatch[1] : null;
  let slug = q.trim().replace(/\?version=[1-9]\d*$/, '');
  if (/^\d+$/.test(slug)) {
    const it = ctx.st.list[+slug - 1];
    if (it?.kind === 'dev') slug = it.id;
    else { const all = (await get('/developments')).developments || []; ctx.rememberDevs(all); if (all[+slug - 1]) slug = all[+slug - 1].slug; }   // "read 2" from anywhere
  }
  else { const n = norm(slug); const hit = ctx.knownDevs().find((d) => norm(d.slug) === n) || ctx.knownDevs().find((d) => norm(d.title).includes(n) || norm(d.slug).includes(n)); if (hit) slug = hit.slug; }
  let d;
  try { d = await get(`/developments/${encodeURIComponent(slug)}${version ? `?version=${version}` : ''}`); } catch (e) {
    if (e.status !== 404) throw e;
    if (!ctx.knownDevs().length) ctx.rememberDevs((await get('/developments').catch(() => ({}))).developments || []);
    return { render: notFound(ctx, `developments ${q}`), sugg: ['developments', 'help'] };
  }
  ctx.st.here = { kind: 'dev', id: d.slug };
  const lf = linkFor(ctx);
  return {
    path: `/developments/${d.slug}${version ? `?version=${version}` : ''}`, sugg: ['developments', 'models'],
    render: (c) => [
      ...H(c, 'development', fmtDate(d.occurred_on || d.published_at)), ...P(c, Hi(d.title)),
      line(Dm('by '), ...(d.authors?.length ? d.authors : [{ username: d.published_by }]).flatMap((a, i, all) => [a.username ? K('@' + a.username, `whois ${a.username}`) : S(ascii(a.name || a)), S(i < all.length - 1 ? ', ' : '')])),
      blank(),
      ...(d.verdict ? [...wrap([Dm('verdict  '), S(d.verdict)], Math.min(c, 78), '', '         '), blank()] : []),
      ...(d.evidence ? [...wrap([Dm('evidence '), Dm(d.evidence)], Math.min(c, 78), '', '         '), blank()] : []),
      ...markdown(c, d.current_evidence_note || '', lf),
      ...markdown(c, d.body_md || d.summary || '', lf),
      ...(d.files?.length ? [blank(), ...H(c, 'files'), ...d.files.map((f) => line('  ', X(f.name, f.url), Dm(f.bytes ? '  ' + fmtBytes(f.bytes) : '')))] : []),
      blank(), line(Dm('more: '), K('developments'), Dm('   back: '), K('back')),
    ],
  };
}

async function download(ctx) {
  const [d, copy, rooms] = await Promise.all([get('/downloads'), text('/content/downloads.md').catch(() => ''), text('/content/app.md').catch(() => '')]);
  const apps = d.apps || []; const app = apps[0];
  const current = (d.runtimes || []).find((r) => r.status === 'latest');
  const lf = linkFor(ctx);
  const body = copy.replace(/^# .*\n/, '');
  return (c) => [
    ...H(c, app ? `soma ${app.version}` : 'soma', apps.length > 1 ? 'mac and windows' : `for ${app?.label || 'mac'}`),
    ...markdown(c, rooms, lf), blank(),
    ...(apps.length ? buttons(c, apps.map((a) => [`download for ${a.label}`, null, a.url])) : [line(Dm('the app is between releases; try again soon.'))]), blank(),
    ...apps.flatMap((a) => [kv(a.platform, `${a.name}  ${fmtBytes(a.size_bytes)}`),
      ...(a.sha256 ? wrap([Dm(pad('  sha256', 13)), S(a.sha256)], c, '', ' '.repeat(13)) : [])]),
    app?.released && kv('released', app.released),
    current && kv('runtime', [S(`${current.name} ${current.runtime_version}  `), K('spec', 'runtime')]),
    blank(),
    ...markdown(c, body, lf).filter(Boolean),
    blank(), ...wrap([Dm(pad('signing key', 13)), S('0lLdgB3LtdBakH70/B8uQZmPYb0pxyOvVXullhq2jNU=')], c, '', ' '.repeat(13)),
    ...P(c, Dm('current releases are signed with this ed25519 key. archived runtimes have checksums, not signatures.')),
  ].filter(Boolean);
}

async function runtime(ctx, which) {
  const id = which || 'current';
  let r; let spec;
  try { [r, spec] = await Promise.all([get(`/runtimes/${encodeURIComponent(id)}`), text(`/runtimes/${encodeURIComponent(id)}/spec.md`).catch(() => '')]); } catch (e) {
    if (e.status !== 404) throw e;
    return { render: () => [line(Dm(`no runtime '${ascii(id)}'. there are: `), K('somatic core', 'runtime'), S('  '), K('rma v1', 'runtime rma-v1'), S('  '), K('v14', 'runtime v14'), S('  '), K('v12.3', 'runtime v12.3'))], sugg: ['runtime', 'help'] };
  }
  const lf = linkFor(ctx);
  return {
    path: id === 'current' || r.status === 'latest' ? '/runtime' : `/runtimes/${r.id}`, sugg: ['get soma', 'models'],
    render: (c) => [
      ...H(c, `${r.name} ${r.runtime_version || ''}`, r.status === 'latest' ? 'the runtime' : r.status),
      ...P(c, r.one_liner || r.headline || ''), blank(),
      ...H(c, 'releases'), ...table(c, (r.releases || []).slice(0, 12), [
        { head: 'version', w: 16, get: (x) => (x.version === r.runtime_version ? Hi(x.version) : S(x.version)) },
        { head: 'released', w: 14, get: (x) => Dm(x.released_on || '') },
        { head: 'files', flex: true, get: (x) => (x.files?.[0] ? X(x.files[0].name, x.files[0].url) : Dm('')) },
      ]), blank(),
      ...(r.status === 'latest' ? [] : [line(Dm('the current runtime: '), K('runtime')), blank()]),
      ...H(c, 'spec'), ...markdown(c, spec, lf),
      blank(), line(Dm('earlier runtimes: '), K('rma v1', 'runtime rma-v1'), S('  '), K('v14', 'runtime v14'), S('  '), K('v12.3', 'runtime v12.3')),
    ],
  };
}

async function whois(ctx, name) {
  let u;
  try { u = await get(`/users/${encodeURIComponent(name.replace(/^@/, ''))}`); } catch (e) {
    if (e.status !== 404) throw e;
    return { render: () => [line(Dm(`no one here is called @${ascii(name.replace(/^@/, ''))}.`))], sugg: ['models', 'help'] };
  }
  const ms = u.checkpoints || [];
  ctx.remember(ms); ctx.st.list = ms.map((m) => ({ kind: 'model', id: m.checkpoint_id }));
  return {
    path: `/u/${u.username}`, sugg: ['models', 'developments'],
    render: (c) => [...H(c, '@' + u.username, u.verified ? 'verified' : ''),
      ...(u.bio ? P(c, u.bio) : []), line(Dm(`since ${fmtDate(u.created_at)}   ${u.checkpoint_count || 0} models   ${u.total_stars || 0} stars   ${u.total_downloads || 0} downloads`)),
      ...(u.links?.length ? [line(...u.links.flatMap((l) => [X(l.label || l.url, l.url), S('  ')]))] : []),
      blank(), ...H(c, 'models'),
      ...(ms.length ? ms.map((m, i) => line(Dm(pad(String(i + 1), 4)), K(m.name, `open ${m.checkpoint_id}`), Dm(m.parameters ? '  ' + fmtCount(m.parameters) : ''))) : [line(Dm('  none public yet'))])],
  };
}

async function page(ctx, name, title) {
  const md = await text(`/content/${name}.md`);
  return (c) => markdown(c, md || `# ${title}`, linkFor(ctx));
}

function info() {
  return (c) => [...H(c, 'for agents'), ...P(c, 'logOS is readable by machines as well as people. start here:'), blank(),
    line('  ', X('/llms.txt', '/llms.txt'), Dm('            the handshake')), line('  ', X('/api/manifest.json', '/api/manifest.json'), Dm('  every endpoint')),
    line('  ', X('/api/developments.md', '/api/developments.md'), Dm('  the full linked notebook')),
    line('  ', X('/api/site-info', '/api/site-info'), Dm('      what this is, in brief')), line('  ', X('/api/runtimes', '/api/runtimes'), Dm('       every runtime and release'))];
}

function notFound(ctx, q) {
  const known = ctx.known(); const words = q.toLowerCase(); const verb = words.split(/\s+/)[0];
  const ms = known.filter((m) => (m.name || '').toLowerCase().includes(words) || lev(norm(q), norm(m.name)) <= 2).slice(0, 4);
  const ds = ctx.knownDevs().filter((d) => d.title.toLowerCase().includes(words)).slice(0, 3);
  const verbs = ['models', 'open', 'developments', 'help', 'search', 'talk', 'star', 'fork', 'download', 'clear', 'history', 'back', 'runtime', 'whoami', 'whois', 'machines', 'link', 'about', 'info', 'upload'];
  const close = verbs.map((k) => [k, lev(verb, k)]).filter((x) => x[1] <= 2 && x[1] > 0).sort((a, b) => a[1] - b[1])[0];
  // a known command whose thing wasn't found says that; an unknown word is an unknown command
  const rest = q.split(/\s+/).slice(1).join(' ');
  const head = verbs.includes(verb) && rest ? `nothing called '${cut(ascii(rest), 40)}' to ${verb}.` : `no command '${cut(ascii(q), 40)}'.`;
  return (c) => [line(Dm(head)), ...(close && !verbs.includes(verb) ? [line(Dm('did you mean '), K(q.replace(new RegExp('^' + verb), close[0])), Dm('?'))] : []),
    ...(ms.length || ds.length ? [blank(), line(Dm('things that match:')), ...ms.map((m) => line('  ', K(m.name, `open ${m.checkpoint_id}`))),
      ...ds.map((d) => wrap([K(d.title, `developments ${d.slug}`)], Math.min(c, 78) - 2, '  ')[0])] : []),
    blank(), line(Dm('or '), K(`search ${cut(ascii(q), 30)}`), Dm('. type '), K('help'), Dm(' for everything you can say.'))];
}

// ---------- you, your machines, linking ----------
function signIn(ctx, then) {
  const inApp = !!window.soma?.signInStart;
  ctx.ask(inApp ? 'sign in through your browser? [y/n] ' : 'sign in with google? [y/n] ', (a) => {
    if (inApp && /^(y|yes|ok|sure|go)$/i.test(a)) {
      ctx.login({ onCode: (code) => ctx.print(() => [line(Dm('approve it in your browser. code '), Hi(code))]) })
        .then((u) => { ctx.print(() => [line(Dm('signed in as '), K('@' + u.username, `whois ${u.username}`))]); if (then) ctx.run(then); })
        .catch((e) => ctx.print(() => [line(Am(ascii(e.message || 'sign-in did not finish')))]));
      return ctx.print(() => [line(Dm('opening your browser ...'))]);
    }
    if (/^(y|yes|ok|sure|go)$/i.test(a)) { if (then) try { sessionStorage.setItem('logos:term:then', then); } catch (_) { /* */ } ctx.login(); return ctx.print(() => [line(Dm('to google ...'))]); }
    ctx.suggest(['models', 'developments', 'help']);
    return ctx.print(() => [line(Dm('not signed in. you can still browse: '), K('models'), Dm(', '), K('developments'), Dm('.'))]);
  }, ['yes', 'no']);
  return (c) => [line(inApp ? 'sign in through your browser. type ' : 'sign in with google. type '), line(K('y', 'yes'), Dm(' or '), K('n', 'no'))];
}

async function machines(ctx) {
  if (!ctx.user()) return signIn(ctx, 'machines');
  const apps = await get('/me/apps');
  ctx.st.apps = apps.apps || apps || [];
  return (c) => [...H(c, 'your machines'),
    ...(ctx.st.apps.length ? ctx.st.apps.map((a) => line(Hi(pad(macName(a.name), 26)), a.online ? S('online   ') : Dm('offline  '), K('unlink', `unlink ${a.token_id}`)))
      : [line(Dm('none linked.'))]),
    blank(), line(K('link <code>', 'link '), '  ', K('cloud'), '  ', K('get soma'))];
}

async function link(ctx, code) {
  if (!ctx.user()) return signIn(ctx, code ? `link ${code}` : 'link');
  if (!code) return machines(ctx);                     // the linked apps are your machines; one list, with unlink on each
  const v = code.toUpperCase().replace(/[^A-Z0-9]/g, ''); const formatted = v.length > 4 ? `${v.slice(0, 4)}-${v.slice(4, 8)}` : v;
  let pending;
  try { pending = await get(`/auth/device/${encodeURIComponent(formatted)}`); } catch (e) {
    return (c) => [line(Am(`no app is waiting with the code ${ascii(formatted)}.`)), ...P(c, Dm('codes last ten minutes; sign in again in the app for a fresh one.'))];
  }
  const prompt = pending.kind === 'shell' ? `sign ${ascii(pending.client_name || 'the soma app')} in as @${ctx.user().username}? it can do all you can here. [y/n] `
    : `link ${ascii(pending.client_name || 'this app')} to @${ctx.user().username}? [y/n] `;
  ctx.ask(prompt, async (a) => {
    const decision = /^(y|yes)$/i.test(a) ? 'approve' : 'deny';
    try { const r = await post(`/auth/device/${pending.user_code}/${decision}`); return ctx.print(() => [line(decision === 'approve' ? Hi(`${ascii(r.client_name)} is signed in as @${ctx.user().username}.`) : Dm(`${ascii(r.client_name)} was refused.`))]); }
    catch (e) { return ctx.print(() => [line(Am(ascii(e.message)))]); }
  }, ['yes', 'no']);
  return (c) => [line(Dm('code '), Hi(pending.user_code)), line(Dm('from '), S(ascii(pending.client_name || 'an app'))), line(K('y', 'yes'), Dm(' or '), K('n', 'no'))];
}

// ---------- chat: the soma app's session, driven from here ----------
const MESSAGE_MAX = 4000;                    // what logOS carries as words; longer goes as a document
// a file onto the machine holding the open model, into its data folder: the soma app on this machine, or a transfer
// through logOS to anywhere else. its name there, or null (and why, printed)
async function putOnMachine(ctx, ch, file) {
  if (ch?.mac === LOCAL) {
    const added = window.soma?.addFiles ? await window.soma.addFiles([file]) : [];
    const f = added[0]; if (!f) { await ctx.print(() => [line(Am('the app could not take it.'))]); return null; }
    const took = await commandResult(ctx, 'data.take', { name: f.name, replace: true });
    if (!took.ok) { await ctx.print(() => [line(Am(`${ascii(f.name)}: ${ascii(took.error || 'it could not be moved in')}`))]); return null; }
    return f.name;
  }
  const token = ch.mac.startsWith(SHARED) ? ctx.st.chat?.token : ch.mac;
  const statusOf = ch.mac.startsWith(SHARED) ? (id) => get(`/shared/${ch.mac.slice(SHARED.length)}/commands/${id}`) : null;
  let name; try { name = await sendToMachine(ctx, { get, post }, token, file, statusOf); } catch (_) { return null; }
  for (let i = 0; i < 30 && !(ctx.st.chat?.s?.data || []).some((x) => x.name === name); i += 1) await new Promise((z) => setTimeout(z, 1000));
  if (!(ctx.st.chat?.s?.data || []).some((x) => x.name === name)) { await ctx.print(() => [line(Am(`${ascii(name)} hasn't reached the machine yet. `), K('/files'))]); return null; }
  return name;
}
function readLine(r) {
  const frac = r.total ? Math.min(1, r.position / r.total) : 0; const live = !r.finished;
  const bpb = r.finished ? r.final_bpb : r.bpb;
  const parts = [`${(frac * 100).toFixed(1)}%`, `${fmtBytes(r.position)} of ${fmtBytes(r.total)}`];
  if (r.finished) { parts.push(fmtDur(r.seconds)); if (r.stopped) parts.push('stopped early'); }
  else { if (r.bps > 0) parts.push(`${fmtBytes(r.bps)}/s`); if (r.eta) parts.push(`${fmtDur(r.eta)} left`); }
  return (c) => [
    line(Dm((r.finished ? 'read ' : r.paused ? 'paused ' : 'reading ')), Hi(ascii(r.name)), Dm(r.learned ? '' : '  (no weight updates)')),
    line(S(progress(frac, Math.max(10, Math.min(30, c - 40)))), S('  '), bpb != null ? (live ? Hi(`${bpb.toFixed(4)} bpb`) : S(`${bpb.toFixed(4)} bpb`)) : null),
    line(Dm(parts.join('  |  ')), ...(live ? [S('   '), ...readKeys(r)] : [])),
  ];
}
// a running read's own controls, by its number (Nov 25): /stop 2 stops read 2 where it is; /resume 2 carries on
const readKeys = (r) => { const n = r.n || 1; return [K(`${r.paused ? '/resume' : '/stop'} ${n}`), S(' '), K(`/skip ${n}`)]; };
// your reads that are running, oldest first
const liveReads = (ch) => (ch?.s?.turns || []).filter((t) => t.kind === 'read' && t.read && !t.read.finished).map((t) => t.read);
// one status row per running read: number, file, progress, what's left, its controls
function readRows(ch) {
  return liveReads(ch).map((r) => {
    const frac = r.total ? Math.min(1, r.position / r.total) : 0;
    const left = r.paused ? 'paused' : r.eta ? `${fmtDur(r.eta)} left` : r.bps > 0 ? `${fmtBytes(r.bps)}/s` : 'starting';
    return [Dm(`${r.n || '-'}  `), Hi(ascii(r.name).slice(0, 28)), Dm('  '), S(progress(frac, 10)), Dm(`  ${(frac * 100).toFixed(1)}%  ${left}   `), ...readKeys(r)];
  });
}
// /lanes (Nov 25, James): what the model is doing for you, read by read, without scrolling back up
function lanesView(ch) {
  const s = ch?.s || {}; const reads = liveReads(ch);
  return (c) => {
    const head = s.lanes ? `${s.lanes_busy ?? '?'} of ${s.lanes} in use` : 'one at a time';
    const out = [...H(c, 'lanes', head)];
    if (!reads.length) out.push(line(Dm('no reads running. '), K('/read ', '/read '), Dm('<file> starts one.')));
    for (const r of reads) out.push(blank(), ...readLine(r)(c).map((l, i) => (i === 0 ? [S(`${r.n || '-'}  `), ...l] : [S('   '), ...l])));
    const st = sessionStatus(ch);
    out.push(blank(), line(Dm('your chat: '), ...(st ? st.segs.filter((g) => g.cmd === undefined) : [Dm('ready')])));
    const auto = s.auto_read || {};
    if (auto.lanes) {
      out.push(blank(), line(Dm(`auto read: ${auto.lanes} lane${auto.lanes === 1 ? '' : 's'}`)));
      for (const a of auto.reading || []) out.push(line(Dm(`  ${a.slot + 1}  `), S(ascii(a.file || '...')), Dm(a.bpb != null ? `  ${a.bpb.toFixed(3)} bpb` : '')));
    }
    if ((s.queue || []).length) out.push(line(Dm(`waiting for a lane: ${s.queue.length}`)));
    if (!s.lanes) out.push(line(Dm('this model answers one thing at a time; reads wait their turn.')));
    return out;
  };
}
// 1.5.2 (Nov 22, James): images show by default, videos are a click; /images and /videos change it, per chat
const MEDIA_KEY = 'logos.media';
let mediaNow = { images: true, videos: false };
function mediaPrefs(chatId) {
  let all = {}; try { all = JSON.parse(localStorage.getItem(MEDIA_KEY) || '{}'); } catch (_) { /* blocked */ }
  return { images: true, videos: false, ...(all[chatId] || {}) };
}
function setMediaPref(chatId, key, on) {
  let all = {}; try { all = JSON.parse(localStorage.getItem(MEDIA_KEY) || '{}'); } catch (_) { /* blocked */ }
  all[chatId || '-'] = { ...mediaPrefs(chatId || '-'), [key]: on };
  try { localStorage.setItem(MEDIA_KEY, JSON.stringify(all)); } catch (_) { /* blocked */ }
  mediaNow = all[chatId || '-'];
}
const mediaLine = (m) => line(Dm('    '), M(m, m.kind === 'video' ? mediaNow.videos : mediaNow.images));
// a page as text, its pictures and videos where they appear: [image: alt](url)
function pageLines(c, text, media) {
  const known = new Map((media || []).map((m) => [m.src, m]));
  const out = [];
  const RE = /\[(image|video): ([^\]]*)\]\((https?:\/\/[^)\s]+)\)/g;
  let last = 0; let mt;
  const prose = (s) => { for (const para of s.split('\n')) if (para.trim()) out.push(...wrap([Dm(para.trim())], Math.min(c, 100), '    ', '    ')); };
  while ((mt = RE.exec(text))) {
    prose(text.slice(last, mt.index));
    out.push(mediaLine(known.get(mt[3]) || { kind: mt[1], src: mt[3], alt: mt[2] }));
    last = RE.lastIndex;
  }
  prose(text.slice(last));
  for (const m of media || []) if (!text.includes(m.src)) out.push(mediaLine(m));
  return out;
}

// somatic core 1.5.0 (Nov 22): what a model asked for while it wrote: a search, a page, or code it ran
function skillLines(c, s, who) {
  const verb = { search: 'searched for', fetch: 'read', run: s.ok == null ? 'is running' : 'ran' }[s.kind] || 'asked for';
  const what = s.kind === 'run' ? '' : ascii(String(s.what || '')).slice(0, Math.max(20, c - 30));
  const how = s.ok === false ? `  (${ascii(s.note || 'refused')})` : s.kind === 'search' && s.results != null ? `  (${s.results} results)`
    : s.bytes ? `  (${fmtBytes(s.bytes)})` : '';
  const out = [line(Dm(`  ${ascii(who)} ${verb} `), s.kind === 'fetch' && s.ok ? X(what, String(s.what)) : S(what), Dm(how))];
  if (s.kind === 'run') {
    for (const l of String(s.what || '').split('\n').slice(0, 12)) out.push(line(Dm('    | '), S(ascii(l).slice(0, c - 8))));
    if (s.output) for (const l of String(s.output).split('\n').slice(0, 12)) out.push(line(Dm('    > '), Dm(ascii(l).slice(0, c - 8))));
  }
  if (s.kind === 'fetch' && s.ok && (s.excerpt || (s.media || []).length)) out.push(...pageLines(c, String(s.excerpt || ''), s.media));
  return out;
}
function requestTag(t) {
  if (t.state === 'queued') return Dm(`  (queued${t.ahead ? `, ${t.ahead} ahead` : ', next'})`);
  if (t.state === 'now') return Am(t.ask ? '  (reading now)' : '  (answering now)');
  if (t.state === 'done') return Dm(t.ask ? '  [done]' : '  [read]');
  if (t.state === 'stopped') return Am('  (stopped)');
  if (t.state === 'failed') return Am('  (failed)');
  return null;
}
function turnRender(t) {
  if (t.kind === 'user' && t.ask) {
    return (c) => wrap([Dm(`${ascii(t.you || 'you')}> `), Dm(t.ask === 'score' ? 'score ' : 'read '), Hi(t.text || ''), ...(t.learned === false && t.ask === 'read' ? [Dm(' (no weight updates)')] : []),
      ...(requestTag(t) ? [requestTag(t)] : [])], Math.min(c, 90), '', '  ');
  }
  if (t.kind === 'marker') return (c) => [line(t.text.startsWith('error') ? Am(ascii(t.text)) : Dm('-- ' + ascii(t.text) + ' --'))];
  if (t.kind === 'read') return readLine(t.read);
  if (t.kind === 'skill') return (c) => skillLines(c, t.skill || {}, t.speaker_name || t.speaker || 'soma');
  // speakers are names (Nov 25, James): the model's, and yours
  const who = t.kind === 'model' ? `${t.speaker_name || t.speaker || 'soma'}${t.comment ? ` (reading ${t.file || 'a file'})` : ''}` : (t.you || 'you');
  const tag = t.kind === 'user' ? requestTag(t) : null;
  return (c) => [...wrap([Dm(ascii(who) + '> '), t.kind === 'model' ? S(t.text || '...') : Hi(t.text || ''), ...(tag ? [tag] : [])], Math.min(c, 90), '', '  '),
    ...(t.media || []).map(mediaLine)];
}

// this machine, in the soma app: its engine answers directly (signed out and offline too); anywhere else, through logOS
const LOCAL = 'local';
const here = () => window.soma?.local;
const SHARED = 'shared:';                                  // a model someone shared with you: logOS relays for its machine
async function sessionOf(mac, rev, wait) {
  if (mac === LOCAL) return here().session(rev, wait);
  if (mac.startsWith(SHARED)) {
    const r = await fetch(`${net.base}/shared/${mac.slice(SHARED.length)}/session?rev=${rev}&wait=${wait}`, auth());
    if (!r.ok) throw new Error(`logOS answered ${r.status}`);
    return r.json();
  }
  const r = await fetch(`${net.base}/me/apps/${mac}/session?rev=${rev}&wait=${wait}`, auth());
  if (!r.ok) throw new Error(`logOS answered ${r.status}`);
  return r.json();
}
async function localModels() {
  if (!here()) return [];
  try { return ((await here().presence()).checkpoints || []).filter((c) => c.chat_ok); } catch (_) { return []; }
}
async function localTalk(ctx, name) {
  if (ctx.st.chat) ctx.st.chat.stopped = true;
  const ch = ctx.st.chat = { mac: LOCAL, name: 'this machine', blocks: new Map(), s: {} };
  try { ch.s = (await here().session(-1, 0)).session || {}; } catch (_) { ch.s = {}; }
  setBaseline(ch, 0);
  const already = ch.s.loaded && ch.s.model === name;
  if (!already) here().command('session.load', { checkpoint: name, where: 'mac' }).catch((e) => ctx.print(() => [line(Am(ascii(e.message)))]));
  ctx.setModel(name); chatPoll(ctx);
  return { path: '/talk', sugg: ['/pick', '/learn on', '/files', '/exit'],
    render: () => [line(Dm(already ? 'talking to ' : 'loading '), Hi(ascii(name)), Dm(' on this machine ...'))] };
}
export async function localModelsPage(ctx) {
  const ms = await localModels();
  return { path: '/library', sugg: ms.length ? [`talk to ${ms[0].name}`, 'sign in'] : ['sign in', 'public models'],
    render: (c) => [...H(c, 'on this machine'), blank(),
      ...ms.map((m) => line(K(ascii(m.name), `talk to ${m.name}`), Dm(`  ${ascii(m.runtime || '')}`))),
      ...(ms.length ? [] : [line(Dm('none yet.'))]), blank(), line(K('sign in'), Dm(' for your library'))] };
}

// resolves once the page is visible (at once if it already is)
const shown = () => (document.visibilityState !== 'hidden' ? Promise.resolve() : new Promise((z) => {
  const on = () => { if (document.visibilityState !== 'hidden') { document.removeEventListener('visibilitychange', on); z(); } };
  document.addEventListener('visibilitychange', on);
}));
export function chatPoll(ctx) {
  const ch = ctx.st.chat;
  if (!ch || ch.polling) return;
  ch.polling = true;
  (async () => {
    let seen = -1;
    while (ctx.st.chat === ch && !ch.stopped) {
      if (ch.mac !== LOCAL) await shown();         // a tab nobody is looking at asks nothing; it catches up when shown
      try {
        const d = await sessionOf(ch.mac, seen, 20); if (ctx.st.chat !== ch || ch.stopped) break;
        if (d.machine?.token_id) ch.token = d.machine.token_id;           // a shared model's machine: where /pick sends files
        seen = d.rev; ch.s = d.session || {}; mediaNow = mediaPrefs(ch.s.chat || '-'); showTurns(ctx, ch);
      } catch (_) { await new Promise((z) => setTimeout(z, 2000)); }
    }
  })();
}
// the app sends its transcript capped to the latest turns, so a turn is new by its id, never by its position: the
// ids already there when you started are the baseline, and every other id prints (and reprints as it grows)
function turnKey(t, i) { return t.id !== undefined && t.id !== null ? `${t.id}` : `${t.kind}:${t.at || i}`; }
function setBaseline(ch, keepLast = 0) {
  const turns = ch.s.turns || [];
  ch.baseline = new Set(turns.slice(0, Math.max(0, turns.length - keepLast)).map(turnKey));
}
function sessionStatus(ch) {
  // what the model is doing now, where it's up to, and what's waiting (Nov 23, James)
  const s = ch.s || {}; const model = ascii(s.model || ctx0(ch));
  const q = (s.queue || []).length;
  const waiting = q ? [Dm(`   |  ${q} waiting`)] : [];
  if (s.ending || (s.loaded && /saving, then|then saving/.test(s.status || ''))) {
    return { key: 'ending', spin: true, segs: [Am(`${model}: ${ascii(/saving|stopping/.test(s.status || '') ? s.status : 'saving, then stopping')}`)] };
  }
  if (!s.loaded) {
    if (!s.opening && !/loading|starting/.test(s.status || '')) return null;
    return { key: 'load', spin: true, segs: [Dm(ascii(/loading|starting|opening/.test(s.status || '') ? s.status : `loading ${model}`))] };
  }
  if (s.stopped) return { key: 'stopped', segs: [Am(`${model}: stopped`), Dm('   '), K('/resume'), Dm(' or a new message carries on'), ...waiting], notimer: true };
  if (/opening chats|^loading .*chat/.test(s.status || '')) return { key: `chats:${s.status}`, spin: true, segs: [Dm(`${model}: ${ascii(s.status)}`), ...waiting] };
  if (s.busy) return { key: `busy:${s.status}`, spin: true, segs: [Hi(model), Dm(`: ${ascii(s.status || 'working')}`), ...waiting], extra: [K('/stop')] };
  return { key: 'ready', segs: [Dm(`${model}: ready`), ...waiting], notimer: true };
}
const ctx0 = (ch) => ch.name || 'the model';
function showStatus(ctx, ch) {
  if (ctx.setRows) ctx.setRows(ch.stopped ? null : readRows(ch));
  if (!ctx.setStatus) return;
  const st = ch.stopped ? null : sessionStatus(ch);
  if (!st) { ch.statusKey = null; return ctx.setStatus(null); }
  if (st.key !== ch.statusKey) { ch.statusKey = st.key; ch.statusSince = Date.now(); }
  ctx.setStatus({ spin: st.spin, notimer: st.notimer, since: ch.statusSince, segs: [...st.segs, ...(st.extra ? [Dm('   '), ...st.extra] : [])] });
}
// what each turn is, in the stream (Nov 26): so a program can follow only replies, or only reads
const TURN_KIND = { user: 'you', model: 'reply', read: 'read', skill: 'skill', marker: 'note' };
function showTurns(ctx, ch) {
  const turns = ch.s.turns || [];
  if (!ch.baseline) setBaseline(ch, 8);                                     // a little context the first time
  const place = new Map((ch.s.queue || []).map((x, i) => [x.turn, i + (ch.s.busy ? 1 : 0)]));
  turns.forEach((t, i) => {
    const key = turnKey(t, i);
    if (ch.baseline.has(key)) return;
    const shown = { ...t, speaker_name: ch.s.model || ch.name, you: ctx.user()?.username, ...(t.state === 'queued' ? { ahead: place.get(t.id) || 0 } : {}) };
    const sig = JSON.stringify([t.text, t.read, t.state, shown.ahead, (t.media || []).length]);
    const b = ch.blocks.get(key);
    if (!b) ch.blocks.set(key, { sig, block: ctx.printLive(turnRender(shown), { kind: TURN_KIND[t.kind] || 'note', model: ch.s.model || ch.name }) });
    else if (b.sig !== sig) { b.sig = sig; ctx.update(b.block, turnRender(shown)); }
  });
  showStatus(ctx, ch);
  if (ch.s.model && ctx.st.model !== ch.s.model && ch.s.loaded) ctx.setModel(ch.s.model);
  if (!ch.s.loaded && ctx.st.model && !ch.s.opening) ctx.setModel(null);
}

// ---------- your name (Nov 25, James): picked by you, never taken from your email ----------
export function chooseName(ctx) {
  ctx.ask('username> ', async (a) => {
    const name = a.trim().toLowerCase().replace(/^@/, '');
    if (!name) return chooseName(ctx);
    const c = await get(`/users/check?u=${encodeURIComponent(name)}`).catch(() => ({}));
    if (!c.available) { await ctx.print(() => [line(Am(ascii(c.reason || 'taken')))]); return chooseName(ctx); }
    try {
      const r = await request('/users/me', 'PATCH', { username: name });
      if (ctx.user()) Object.assign(ctx.user(), { username: r.username, username_chosen: true });
      return ctx.print(() => [line(Hi(`@${ascii(r.username)}`))]);
    } catch (e) { await ctx.print(() => [line(Am(ascii(e.message)))]); return chooseName(ctx); }
  }, []);
}

// ---------- your terminal (Nov 22): what `home` runs, and what logOS opens with ----------
async function setPref(ctx, key, value) {
  if (!ctx.user()) return signIn(ctx, 'settings');
  const r = await request('/users/me', 'PATCH', { prefs: { [key]: value } });
  if (ctx.user()) ctx.user().prefs = r.prefs || {};
  return settingsPage(ctx);
}

function settingsPage(ctx) {
  if (!ctx.user()) return signIn(ctx, 'settings');
  const u = ctx.user(); const p = u.prefs || {};
  return { path: '/settings', sugg: [],
    render: (c) => [...H(c, 'settings'),
      line(Dm('username  '), Hi(`@${ascii(u.username)}`), Dm('   '), K('set username', 'set username')),
      line(Dm('home      '), S(p.home ? ascii(p.home) : '-'), Dm('   '), K('set home <command>', 'set home ')),
      line(Dm('start     '), S(p.start ? ascii(p.start) : '-'), Dm('   '), K('set start <command>', 'set start '))] };
}

// ---------- RunPod, from nothing to a linked key (Nov 22) ----------
const RUNPOD = { console: 'https://console.runpod.io', billing: 'https://console.runpod.io/user/billing', keys: 'https://console.runpod.io/user/credentials',
  docs: 'https://docs.runpod.io/get-started/credentials' };
function runpodSetup(ctx) {
  return { path: '/runpod', sugg: ['runpod key', 'cloud'],
    render: (c) => [...H(c, 'set up runpod', 'about five minutes'),
      ...P(c, 'models run on gpus rented from runpod, on your own account. runpod bills your account for compute and persistent storage; logOS does not process those payments.'), blank(),
      ...P(c, Hi('1. '), X('make a runpod account', RUNPOD.console), Dm('.')),
      ...P(c, Hi('2. '), X('add credit', RUNPOD.billing), Dm('. turn on a low-balance alert.')),
      ...P(c, Hi('3. '), X('create an api key', RUNPOD.keys), Dm(': settings, api keys, create. name: logOS. permissions: read/write on pods and network volumes, read on billing (so logOS can rescue models before credit runs out). copy it; it is shown once.')),
      ...P(c, Hi('4. '), K('runpod key'), Dm(', then paste it. stored encrypted. never shown again. '), K('runpod forget'), Dm(' removes it.')),
      blank(),
      ...P(c, Dm('then: '), K('cloud offers'), Dm(' shows what a gpu costs now; '), K('move <model>', 'move '), Dm(' puts a model on a volume in a data center; '), K('talk to <model>', 'talk to '), Dm(' starts it.')),
      ...P(c, Dm('idle machines save and stop after 20 minutes. none runs past 8 hours unless asked (or warm). runpod caps new accounts at $80/h.'))] };
}

// ---------- /post (admins, Nov 22): a markdown file becomes a development ----------
async function postDev(ctx, how) {
  if (ctx.user()?.role !== 'admin') return { render: () => [line(Dm('posting developments is for admins.'))] };
  const send = async (markdown, from) => {
    try {
      const r = await post('/admin/developments/post', { markdown });
      ctx.suggest([`developments ${r.slug}`, 'developments']);
      return ctx.print(() => [line(Hi('posted '), K(ascii(r.title), `developments ${r.slug}`)), line(Dm(`version ${r.version} of ${r.slug}, ${r.visibility === 'admin' ? 'admins only' : 'public'}, from ${from}.`))]);
    } catch (e) { return ctx.print(() => [line(Am(ascii(e.message)))]); }
  };
  if (/^paste$/i.test(how || '')) {
    let md = ''; try { md = await navigator.clipboard.readText(); } catch (_) { return { render: () => [line(Am('the browser would not share the clipboard; choose the file instead: ')), K('/post')] }; }
    await send(md, 'the clipboard');
    return true;
  }
  if (!ctx.pick) return { render: () => [line(Dm('choose a file: this terminal can\'t open a file picker here. '), K('/post paste'), Dm(' reads the clipboard.'))] };
  ctx.pick(async (files) => { for (const f of files) await send(await f.text(), f.name); }, '.md,.markdown,.txt');
  return { render: (c) => [...P(c, Dm('choose a .md file. a +++ toml header (title, summary, kind, verdict, evidence, thread, tags, visibility) is read as tools/logos_development.py reads it; without one, the first "# heading" is the title. or '), K('/post paste'), Dm(' to post the clipboard.'))] };
}

// ---------- chats (Nov 22, James): the weights are the model; every store belongs to one chat ----------
const when = (iso) => (iso ? ascii(new Date(iso).toLocaleString('en-AU', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })) : '');
function chatLines(c, list, current, inTalk) {
  return list.map((x, i) => line(Dm(pad(String(i + 1), 3)), x.chat_id === current ? Hi('* ') : S('  '),
    K(ascii(x.name || 'chat'), inTalk ? `/chat ${i + 1}` : `chat history ${x.chat_id}`),
    Dm(`  ${x.model && !inTalk ? `${ascii(x.model)}, ` : ''}${x.sessions || 0} session${x.sessions === 1 ? '' : 's'}${x.bytes ? `, ${fmtBytes(x.bytes)}` : ''}${x.last_used || x.last_used_at ? `, ${when(x.last_used || x.last_used_at)}` : ''}`)));
}

async function chatsPage(ctx, model) {
  if (!ctx.user()) return signIn(ctx, 'chats');
  let d; try { d = await get(`/me/chats${model ? `?model=${encodeURIComponent(model)}` : ''}`); } catch (e) { return { render: () => [line(Am(ascii(e.message)))] }; }
  ctx.st.list = d.chats.map((x) => ({ kind: 'chat', id: x.chat_id }));
  return { path: null, sugg: d.chats.length ? [`chat history ${d.chats[0].chat_id}`, 'my models'] : ['my models'],
    render: (c) => [...H(c, model ? `your chats with ${ascii(model)}` : 'your chats', 'one per model, the same on every device; the weights are the model'),
      ...(d.chats.length ? chatLines(c, d.chats, null, false) : [line(Dm('none yet. '), K('talk to <model>', 'talk to '), Dm(' starts one.'))]),
      blank(), line(K('chat history <n>', 'chat history '), S('   '), K('delete chat <n>', 'delete chat '))] };
}

function chatRef(ctx, ref) {                       // a number from the last list, or a chat id
  const s = String(ref || '').trim();
  if (/^\d+$/.test(s)) { const it = ctx.st.list[+s - 1]; if (it?.kind === 'chat') return it.id; }
  return s;
}

function historyLines(c, events) {
  const out = [];
  for (const e of events) {
    if (e.kind === 'session_start') out.push(blank(), line(Dm(`--- ${when(e.at)}${e.data?.who ? `  @${ascii(e.data.who)}` : ''}`)));
    else if (e.kind === 'user') out.push(...wrap([Hi('you  '), S(e.text)], Math.min(c, 100), '', '     '));
    else if (e.kind === 'model') out.push(...wrap([Dm(e.data?.comment ? 'it*  ' : 'it   '), S(e.text)], Math.min(c, 100), '', '     '), ...(e.data?.media || []).map(mediaLine));
    else if (e.kind === 'skill') out.push(...skillLines(c, { ...(e.data || {}), what: e.text }, 'it'));
    else if (e.kind === 'read') out.push(line(Dm(`     ${e.data?.score ? 'scored' : e.data?.learned ? 'learned from' : 'read'} ${ascii(e.data?.file || '')}${e.data?.bpb != null ? `, ${Number(e.data.bpb).toFixed(3)} bpb` : ''}`)));
    else if (e.kind === 'session_end') out.push(line(Dm(`--- saved${e.data?.learned ? '; the weights learned from it' : ''}`)));
    else if (e.text) out.push(line(Dm(`     ${ascii(e.text)}`)));
  }
  return out;
}

async function chatHistory(ctx, ref) {
  if (!ctx.user()) return signIn(ctx, 'chats');
  const id = chatRef(ctx, ref);
  let d; try { d = await get(`/me/chats/${encodeURIComponent(id)}/history`); } catch (e) { return { render: () => [line(Am(ascii(e.message))), line(K('chats'))] }; }
  return { path: null, sugg: [`talk to ${d.chat.model}`, 'chats'],
    render: (c) => [...H(c, `${ascii(d.chat.name)}`, `with ${ascii(d.chat.model)}; ${d.chat.sessions || 0} sessions`),
      ...(d.events.length ? historyLines(c, d.events) : [line(Dm('nothing said in it yet.'))])] };
}

// ---------- notices (storage, Nov 22): what logOS tells you; your runpod credit as it last saw it; models it holds ----------
async function noticesPage(ctx) {
  if (!ctx.user()) return signIn(ctx, 'notices');
  let n, s; try { [n, s] = await Promise.all([get('/me/notices?limit=60'), get('/me/storage')]); } catch (e) { return { render: () => [line(Am(ascii(e.message)))] }; }
  if (n.unseen) post('/me/notices/seen').then(() => { ctx.st.unseen = 0; }).catch(() => {});
  const a = s.account;
  const credit = !a ? 'not checked yet (link a runpod key: runpod setup)'
    : a.state === 'no_permission' ? "logOS can't see it: your key needs billing:read (runpod setup)"
      : a.state === 'postpaid' ? 'invoiced (postpaid): nothing to watch'
        : `$${Number(a.balance || 0).toFixed(2)}${a.hours_left != null ? `, about ${(a.hours_left / 24).toFixed(1)} days at $${Number(a.burn_per_hour || 0).toFixed(3)}/h` : ''} (${ascii(a.state)})`;
  return { path: '/notices', sugg: n.notices.some((x) => x.kind === 'app_update' && !x.seen_at) ? ['get soma', 'my models'] : ['my models', 'runpod setup'],
    render: (c) => [...H(c, 'notices', 'what logOS tells you'),
      kv('credit', credit, 10),
      ...(s.holds.length ? [blank(), line(Hi('held on logOS')), ...s.holds.map((h) => line(Dm('  '), K(ascii(h.name), `where ${h.name}`),
        Dm(h.state === 'held' ? `  until ${fmtDate(h.until)}; move it, or it is deleted then` : '  on its way here from its volume')))] : []),
      blank(),
      ...(n.notices.length ? n.notices.flatMap((x) => wrap([x.seen_at ? Dm(`${fmtDate(x.created_at)}  `) : Am(`${fmtDate(x.created_at)}  `), S(x.text), ...(x.kind === 'app_update' ? [S(' '), K('get soma')] : [])], Math.min(c, 100), '', '            '))
        : [line(Dm('nothing yet.'))]),
      blank(), ...P(c, Dm(`if your runpod credit is about to run out, logOS brings your models here and holds them ${s.hold_days} days. `), X('top up', s.topup), Dm('.'))] };
}

// ---------- downloads (Nov 22, James): take your chats with you; a store comes from the machine that holds it ----------
async function saveFile(path) {
  const r = await fetch(`${net.base}${path}`, auth());
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `logOS answered ${r.status}`);
  const name = (r.headers.get('Content-Disposition') || '').match(/filename="([^"]+)"/)?.[1] || 'chats.zip';
  const url = URL.createObjectURL(await r.blob());
  const a = document.createElement('a'); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
  return name;
}

async function findChat(ctx, ref) {
  // the current chat when talking, a number from the last list, a chat id, or a name
  const s = String(ref || '').trim();
  if (!s) return ctx.st.chat?.s?.chat || null;
  if (/^\d+$/.test(s) && ctx.st.list[+s - 1]?.kind === 'chat') return ctx.st.list[+s - 1].id;
  if (/^ch_[0-9a-f]+$/.test(s)) return s;
  const d = await get('/me/chats');
  const hit = d.chats.filter((x) => (x.name || '').toLowerCase() === s.toLowerCase());
  return hit.length ? hit[0].chat_id : null;
}

function storeWhenReady(ctx, chat) {
  (async () => {
    let r; try { r = await post(`/me/chats/${encodeURIComponent(chat.chat_id)}/store`); } catch (e) { return ctx.print(() => [line(Am(`${ascii(chat.name)}'s store: ${ascii(e.message)}`))]); }
    if (r.status === 'waiting') return ctx.print(() => [line(Dm(`${ascii(chat.name)}'s store: ${ascii(r.note)}. ask again then: `), K(`download chat ${chat.chat_id}`))]);
    for (let i = 0; i < 180 && r.status !== 'ready'; i += 1) {
      await new Promise((z) => setTimeout(z, 5000));
      r = await get(`/me/chats/${encodeURIComponent(chat.chat_id)}/store`).catch(() => r);
    }
    if (r.status !== 'ready') return ctx.print(() => [line(Dm(`${ascii(chat.name)}'s store hasn't come up yet; it will when its machine can send it: `), K(`download chat ${chat.chat_id}`))]);
    return ctx.print(() => [line(Dm(`${ascii(chat.name)}'s store is ready${r.bytes ? ` (${fmtBytes(r.bytes)})` : ''}: `), X('download it', r.download_url), Dm('. the link lasts a day.'))]);
  })();
}

async function downloadChats(ctx, ref, all) {
  if (!ctx.user()) return signIn(ctx, all ? 'download chats' : 'download chat');
  try {
    if (all) {
      const d = await get('/me/chats');
      if (!d.chats.length) return { render: () => [line(Dm('you have no chats yet.'))] };
      const name = await saveFile('/me/chats/download');
      for (const c of d.chats) storeWhenReady(ctx, c);
      return { render: (c) => [line(Dm('saved '), Hi(ascii(name)), Dm(`: the histories of your ${d.chats.length} chat${d.chats.length === 1 ? '' : 's'}.`)),
        ...P(c, Dm('stores come from the machines holding them. links print here; each lasts a day.'))] };
    }
    const id = await findChat(ctx, ref);
    if (!id) return { render: () => [line(Dm(ref ? `no chat of yours called ${ascii(ref)}. ` : 'which chat? '), K('chats'))] };
    const h = await get(`/me/chats/${encodeURIComponent(id)}/history?limit=1`);
    const name = await saveFile(`/me/chats/${encodeURIComponent(id)}/download`);
    storeWhenReady(ctx, { chat_id: id, name: h.chat.name });
    return { render: (c) => [line(Dm('saved '), Hi(ascii(name)), Dm(': its history.')),
      ...P(c, Dm(`its store comes from the machine holding ${ascii(h.chat.model)}. the link prints here.`))] };
  } catch (e) { return { render: () => [line(Am(ascii(e.message)))] }; }
}

// ---------- closing an account (privacy, level 1): everything logOS holds about you goes ----------
function closeAccount(ctx) {
  if (!ctx.user()) return signIn(ctx, 'close account');
  const me = ctx.user().username;
  ctx.ask('delete your published models too? [y/n]> ', (pub) => {
    const removePublished = /^(y|yes)$/i.test(pub);
    ctx.ask(`type your username (${ascii(me)}) to close your account for good> `, async (a) => {
      if ((a || '').trim() !== me) return ctx.print(() => [line(Dm('kept. nothing changed.'))]);
      try {
        const r = await post('/me/close', { confirm: me, remove_published: removePublished });
        await ctx.logout();
        return ctx.print(() => [line(Dm(`closed. ${r.models_deleted} model${r.models_deleted === 1 ? '' : 's'} deleted, with your chats, histories, keys and links. goodbye.`))]);
      } catch (e) { return ctx.print(() => [line(Am(ascii(e.message)))]); }
    });
  }, ['no', 'yes']);
  return { path: null, sugg: [],
    render: (c) => [...H(c, 'close your account', 'everything logOS holds about you goes'),
      ...P(c, Dm('deleted: your private models and files, chats and histories (stores go when their machines next report), runpod key, tokens, terminal stream, shares, library, linked apps, sign-ins, comments, follows.')),
      ...P(c, Dm('kept: published models, unless you ask (others may rely on them; they show a closed account as maker). your runpod account, its volumes and your machines are yours.')),
      ...P(c, Am('this can\'t be undone. '), Dm('take your chats first: '), K('download chats'), Dm('; your models: '), K('my models'), Dm('.'))] };
}

// ---------- new model (Nov 22, James): a shape, where it runs (compute only, shapes that fit), where it's saved ----------
const gb = (x) => `${Number(x).toFixed(x < 10 ? 1 : 0)} GB`;
const fmtParams = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 2)}B` : `${Math.round(n / 1e6)}M`);
async function newModel(ctx) {
  if (!ctx.user()) return signIn(ctx, 'new model');
  const d = await get('/shapes');
  const step2 = async (shape) => {
    // where it can run: computers that can create one and hold it, and gpus on your runpod account
    const apps = ((await get('/me/apps').catch(() => ({})))?.apps || []).filter((a) => a.online && (a.state?.app?.capabilities || []).includes('create.v1'));
    const fits = (a) => !a.state.app.memory_bytes || shape.computer_gb * 1e9 <= a.state.app.memory_bytes * 0.75;
    const cloud = await get('/me/cloud').catch(() => null);
    let offers = [];
    if (cloud?.runpod?.linked) {
      const q = shape.name === 'custom' ? `width=${shape.width}&depth=${shape.depth}&heads=${shape.heads}` : `shape=${shape.name}`;
      const o = await get(`/me/cloud/new-offers?${q}`).catch(() => ({ offers: [] }));
      const seen = new Set();
      for (const x of o.offers || []) for (const dc of x.data_centers || []) if (!seen.has(dc)) { seen.add(dc); offers.push({ ...x, dc, storage: o.storage }); }
      offers = offers.slice(0, 6);
    }
    const mine = here() ? await thisComputerName() : null;
    const options = [...apps.filter(fits).map((a) => ({ kind: 'computer', app: a, label: `${macName(a.name)}${a.name === mine ? ' (this machine)' : ''}` })),
      ...offers.map((o) => ({ kind: 'gpu', offer: o, label: `a gpu in ${o.dc}: ${o.name.replace('NVIDIA ', '')}, $${o.price.toFixed(2)}/h` }))];
    const too = apps.filter((a) => !fits(a));
    if (!options.length) {
      return ctx.print((c) => [line(Am(`nowhere to start a ${ascii(shape.name)} model yet.`)),
        ...(too.length ? [line(Dm(`${too.map((a) => macName(a.name)).join(', ')} hasn't the memory for it (${gb(shape.computer_gb)}).`))] : []),
        ...(cloud?.runpod?.linked ? [line(Dm(`no gpu with ${shape.gpu_gb} GB is free right now.`))] : [line(Dm('a gpu needs a runpod account: '), K('runpod setup'))]),
        line(Dm('soma 2.0.4 or later can create one on a machine: '), K('get soma'))]);
    }
    await ctx.print((c) => [blank(), line(Hi('where it runs')), ...options.map((o, i) => line(Dm(`  ${i + 1}. `), S(ascii(o.label)))),
      ...(too.length ? [line(Dm(`  (${too.map((a) => macName(a.name)).join(', ')}: not enough memory for this shape)`))] : [])]);
    ctx.ask('where> ', async (a) => {
      const o = options[+a - 1]; if (!o) return ctx.print(() => [line(Dm('cancelled.'))]);
      const saves = o.kind === 'computer'
        ? [{ label: `${macName(o.app.name)} only`, logos: false }, { label: `${macName(o.app.name)}, and a backup on logOS after its first save`, logos: true }]
        : [{ label: `a network volume in ${o.offer.dc} (a gpu only loads from its own volume; about $${(o.offer.storage?.monthly_usd || 0).toFixed(2)}/month while it's there)`, logos: false }];
      await ctx.print(() => [blank(), line(Hi('where it is saved')), ...saves.map((s, i) => line(Dm(`  ${i + 1}. `), S(ascii(s.label))))]);
      ctx.ask('saved> ', async (b) => {
        const save = saves[+b - 1] || (saves.length === 1 && /^(y|yes|1)?$/i.test(b) ? saves[0] : null);
        if (!save) return ctx.print(() => [line(Dm('cancelled.'))]);
        ctx.ask('name> ', async (raw) => {
          const name = (raw || '').trim(); if (!name) return ctx.print(() => [line(Dm('cancelled.'))]);
          const cost = o.kind === 'gpu' ? ` on ${ascii(o.offer.name)} at $${o.offer.price.toFixed(2)}/h, stopping itself after 20 idle minutes` : '';
          ctx.ask(`create ${ascii(name)}: ${ascii(shape.name)}, ${fmtParams(shape.parameters)} parameters${cost}? [y/n]> `, async (y) => {
            if (!/^(y|yes)$/i.test(y)) return ctx.print(() => [line(Dm('nothing created.'))]);
            try {
              if (o.kind === 'gpu') {
                const r = await post('/me/cloud/machines', { purpose: 'new', ...(shape.name === 'custom' ? { width: shape.width, depth: shape.depth, heads: shape.heads } : { shape: shape.name }),
                  name, data_center: o.offer.dc, max_rate: o.offer.price });
                await ctx.print(() => [line(Dm('renting '), Hi(ascii(r.machine.gpu || 'a gpu')), Dm(` in ${ascii(r.machine.data_center || '')}; it creates ${ascii(name)} on its volume, then you can talk to it.`))]);
                return ctx.run(`cloud chat ${r.machine.machine_id}`);
              }
              const args = { name, settings: shape.settings };
              const isHere = here() && (await thisComputerName()) === o.app.name;
              if (isHere) await here().command('model.create', args); else await post(`/me/apps/${o.app.token_id}/commands`, { kind: 'model.create', args });
              await ctx.print(() => [line(Dm('creating '), Hi(ascii(name)), Dm(` on ${ascii(macName(o.app.name))}. it knows nothing yet. talk to it, or /pick files.`))]);
              if (save.logos) backupWhenSaved(ctx, o.app.token_id, name, isHere);
              return ctx.run(`talk to ${name}`);
            } catch (e) { return ctx.print(() => [line(Am(ascii(e.message)))]); }
          }, ['yes', 'no']);
        });
      }, saves.map((_, i) => String(i + 1)));
    }, options.map((_, i) => String(i + 1)));
    return true;
  };
  ctx.ask('shape> ', async (a) => {
    const s = (a || '').trim().toLowerCase();
    if (s === 'custom' || s === String(d.shapes.length + 1)) {
      return ctx.ask('width depth heads> ', async (w) => {
        const [width, depth, heads] = (w || '').trim().split(/\s+/).map(Number);
        try { return step2(await get(`/shapes/custom?width=${width}&depth=${depth}&heads=${heads}`)); } catch (e) { return ctx.print(() => [line(Am(ascii(e.message)))]); }
      });
    }
    const shape = d.shapes.find((x) => x.name === s) || d.shapes[+s - 1];
    if (!shape) return ctx.print(() => [line(Dm('cancelled.'))]);
    return step2(shape);
  }, [...d.shapes.map((x) => x.name), 'custom']);
  return { path: null, sugg: [],
    render: (c) => [...H(c, 'new model', 'a shape, where it runs, where it is saved'),
      ...d.shapes.map((x, i) => line(Dm(`  ${pad(String(i + 1), 3)}`), S(pad(x.name, 8)), Dm(pad(`${x.width} x ${x.depth}`, 12)), S(pad(fmtParams(x.parameters), 7)),
        Dm(`  gpu ${pad(gb(x.gpu_gb), 7)} computer ${pad(gb(x.computer_gb), 8)} ${c >= 100 ? ascii(x.known) : ''}`))),
      line(Dm(`  ${pad(String(d.shapes.length + 1), 3)}`), S('custom'), Dm('  your own width, depth and heads'))] };
}

async function thisComputerName() {
  try { return `soma on ${(await window.soma.local.presence()).app?.machine}`; } catch (_) { return null; }
}

function backupWhenSaved(ctx, tokenId, name, isHere) {
  // logOS too: once the new model has loaded (its first save), back it up, so it's on logOS and its chats' histories have a home
  (async () => {
    for (let i = 0; i < 180; i += 1) {
      await new Promise((z) => setTimeout(z, 5000));
      const s = isHere ? await here().session(-1, 0).then((r) => r?.session || r).catch(() => null)
        : (await get(`/me/apps/${tokenId}/session?rev=-1&wait=0`).catch(() => null))?.session;
      if (s?.loaded && s.model === name) {
        try {
          if (isHere) await here().command('model.backup', { checkpoint: name }); else await post(`/me/apps/${tokenId}/commands`, { kind: 'model.backup', args: { checkpoint: name } });
          return ctx.print(() => [line(Dm('backing '), Hi(ascii(name)), Dm(' up to logOS.'))]);
        } catch (e) { return ctx.print(() => [line(Am(`the backup didn't start: ${ascii(e.message)}`))]); }
      }
    }
  })();
}

// ---------- a model's budget and usage (Nov 21): what its machines may spend in a month, and who spent it ----------
const usd = (x) => (x == null ? 'none' : `$${Number(x).toFixed(2)}`);
async function budgetPage(ctx, ref, change) {
  if (!ctx.user()) return signIn(ctx, `budget ${ref}`);
  const url = `/me/models/${encodeURIComponent(ref)}/budget`;
  let d;
  try { d = change ? await post(url, change) : await get(url); } catch (e) { return { render: () => [line(Am(ascii(e.message)))] }; }
  return { path: null, sugg: [`usage ${ref}`, `history ${ref}`],
    render: (c) => [...H(c, `${d.model}: budget`, 'gpu time, this month (sydney)'),
      line(Dm('this month   '), Hi(usd(d.spent)), Dm(d.monthly_usd != null ? ` of ${usd(d.monthly_usd)}; ${usd(d.left)} left` : ' spent; no budget set')),
      line(Dm('for guests   '), S(d.monthly_usd == null ? 'only you can start it until it has a budget'
        : d.guest_max_rate != null ? `they can start it on machines up to ${usd(d.guest_max_rate)}/h` : 'they can start it on the cheapest machine that fits')),
      blank(),
      ...(d.owner ? [line(K(`budget ${d.model} $20/month`, `budget ${ref} $`), Dm('  a monthly cap; when it runs out, its machine saves and stops')),
        line(K(`budget ${d.model} $1.50/h`, `budget ${ref} $`), Dm('  the most a machine a guest starts may cost an hour')),
        line(K(`budget ${d.model} off`, `budget ${ref} off`), Dm('  no cap (and guests can no longer start it)'))] : [line(Dm('its owner sets the budget.'))])] };
}

async function usagePage(ctx, ref) {
  if (!ctx.user()) return signIn(ctx, `usage ${ref}`);
  let d; try { d = await get(`/me/models/${encodeURIComponent(ref)}/usage`); } catch (e) { return { render: () => [line(Am(ascii(e.message)))] }; }
  return { path: null, sugg: [`budget ${ref}`, `history ${ref}`],
    render: (c) => [...H(c, `${d.model}: usage`, 'this month (sydney)'),
      line(Dm('spent '), Hi(usd(d.spent)), Dm(d.monthly_usd != null ? ` of a ${usd(d.monthly_usd)} budget` : '')), blank(),
      line(Hi('machines, by who started them')),
      ...(d.machines.length ? d.machines.map((x) => line(Dm(`  ${pad('@' + ascii(x.who), 18)} `), S(`${x.machines} machine${x.machines === 1 ? '' : 's'}, ${x.hours} h, ${usd(x.dollars)}`))) : [line(Dm('  none this month.'))]),
      blank(), line(Hi('reading and teaching, by person')),
      ...(d.reads.length ? d.reads.map((x) => line(Dm(`  ${pad('@' + ascii(x.who), 18)} `), S(`taught ${fmtBytes(x.taught_bytes)}, read ${fmtBytes(x.read_bytes)}, ${x.scores} score${x.scores === 1 ? '' : 's'}`))) : [line(Dm('  nothing recorded this month.'))])] };
}

function parseBudget(rest) {
  const s = rest.trim().toLowerCase();
  if (/^(off|none|no cap|clear)$/.test(s)) return { monthly_usd: null };
  let m;
  if ((m = s.match(/^(?:guests?\s+(?:up\s+to\s+)?)?\$?\s*(\d+(?:\.\d+)?)\s*(?:\/\s*h|\/\s*hr|\/\s*hour|an hour|per hour)$/))) return { guest_max_rate: Number(m[1]) };
  if ((m = s.match(/^(?:guests?\s+)?(?:any|no ceiling|no limit)$/))) return { guest_max_rate: null };
  if ((m = s.match(/^\$?\s*(\d+(?:\.\d+)?)\s*(?:\/\s*(?:month|mo|m)|a month|per month|monthly)?$/))) return { monthly_usd: Number(m[1]) };
  return undefined;
}

// ---------- a model's learning record (Nov 21): its curve (scores over time) and its history ----------
async function curvePage(ctx, ref, file) {
  if (!ctx.user()) return signIn(ctx, `curve ${ref}`);
  const d = await get(`/me/models/${encodeURIComponent(ref)}/reads${file ? `?file=${encodeURIComponent(file)}` : ''}`);
  const scores = d.reads.filter((r) => r.score && r.bpb != null).reverse();
  const files = [...new Set(scores.map((r) => r.file))];
  const on = file || (files.length === 1 ? files[0] : null);
  const pts0 = scores.filter((r) => !on || r.file === on);
  if (!on && files.length > 1) {
    return { render: (c) => [...H(c, `${d.model}: scores`), line(Dm('scores on more than one file; choose one:')),
      ...files.map((f) => line('  ', K(ascii(f), `curve ${ref} on ${f}`), Dm(`  ${scores.filter((r) => r.file === f).length} scores`)))] };
  }
  if (pts0.length < 2) {
    return { render: (c) => [...H(c, `${d.model}: curve`), ...pts0.map((r) => line(Dm(`${fmtDate(r.at)}  `), S(Number(r.bpb).toFixed(4)), Dm(` bpb on ${ascii(r.file)}`))),
      line(Dm(pts0.length ? 'one score so far. ' : 'no scores yet. '), Dm('while talking to it, '), K('/score <file>', '/score '), Dm(' reads a file with learning off and records how it does.'))] };
  }
  const t0 = new Date(pts0[0].at).getTime();
  const pts = pts0.map((r) => [(new Date(r.at).getTime() - t0) / 864e5, Number(r.bpb)]);
  const ys = pts.map((p) => p[1]); const lo = Math.min(...ys); const hi = Math.max(...ys); const pad = Math.max(0.01, (hi - lo) * 0.15);
  const x1 = Math.max(pts[pts.length - 1][0], 1);
  const last = pts0[pts0.length - 1]; const first = pts0[0];
  return { path: null, sugg: [`history ${ref}`, `where ${ref}`],
    render: (c) => [...H(c, `${d.model}: ${on}`, 'bits per byte, lower is better'),
      ...plot(c, pts, { y0: lo - pad, y1: hi + pad, x1, ylabels: [hi, (hi + lo) / 2, lo], xlabels: [0, Math.round(x1 / 2), Math.round(x1)], xunit: 'd' }),
      line(Dm(`${pts0.length} scores from ${fmtDate(first.at)}: `), S(Number(first.bpb).toFixed(4)), Dm(' to '), Hi(Number(last.bpb).toFixed(4)))] };
}

async function historyPage(ctx, ref) {
  if (!ctx.user()) return signIn(ctx, `history ${ref}`);
  const d = await get(`/me/models/${encodeURIComponent(ref)}/history`);
  const when = (e) => ascii(new Date(e.at).toLocaleString('en-AU', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }));
  const say = (e) => {
    if (e.kind === 'revision') return [Dm('saved     '), S(`revision ${String(e.revision_id).slice(0, 8)}`), Dm(e.pinned ? '  pinned' : '')];
    if (e.kind === 'read') return [Dm(e.learned ? 'learned   ' : 'read      '), S(ascii(e.file)), Dm(`  ${fmtBytes(e.bytes)}${e.bpb != null ? `  ${Number(e.bpb).toFixed(3)} bpb` : ''}${e.who ? `  @${ascii(e.who)}` : ''}`)];
    if (e.kind === 'score') return [Dm('scored    '), S(ascii(e.file)), Dm(`  ${e.bpb != null ? Number(e.bpb).toFixed(4) : '?'} bpb${e.who ? `  @${ascii(e.who)}` : ''}`)];
    if (e.kind === 'machine') return [Dm('machine   '), S(`${ascii(e.purpose)} on ${ascii(e.gpu || '')} ${ascii(e.data_center || '')}`), Dm(`  ${ascii(e.status)}${e.hours != null ? `  ${e.hours} h` : ''}${e.note ? `  ${ascii(e.note)}` : ''}`)];
    if (e.kind === 'shared') return [Dm('shared    '), S(`with @${ascii(e.who)}`)];
    return [Dm(ascii(e.kind))];
  };
  return { path: null, sugg: [`curve ${ref}`, `where ${ref}`],
    render: (c) => [...H(c, `${d.model}: history`), ...(d.events.length ? d.events.map((e) => line(Dm(`${pad(when(e), 16)} `), ...say(e))) : [line(Dm('nothing yet.'))])] };
}

// ---------- shared models (Nov 21): models people share with you, and the ones you share ----------
async function sharedPage(ctx) {
  if (!ctx.user()) return signIn(ctx, 'shared models');
  const d = await get('/me/shared');
  const status = (x) => (x.machine ? `${x.machine.status === 'running' ? 'loaded' : ascii(x.machine.status)} on ${ascii(x.machine.gpu || 'a gpu')} $${Number(x.machine.rate || 0).toFixed(2)}/h` : 'not running');
  return { path: '/shared', sugg: d.with_me.length ? [`talk to ${d.with_me[0].name}`, 'my models'] : ['my models'],
    render: (c) => [...H(c, 'shared models'),
      line(Hi('shared with you')),
      ...(d.with_me.length ? d.with_me.map((x) => line('  ', K(ascii(x.name), `talk to ${x.checkpoint_id}`), Dm(`  @${ascii(x.owner)}'s  ${status(x)}  ${x.teach ? 'learning on: you teach it' : 'learning off: reading only'}`))) : [line(Dm('  none yet.'))]),
      blank(), line(Hi('yours, shared')),
      ...(d.by_me.length ? d.by_me.map((x) => line('  ', K(ascii(x.name), `where ${x.checkpoint_id}`), Dm('  with '),
        ...x.people.flatMap((u, i) => { const on = (x.teachers || []).includes(u);
          return [S(`@${ascii(u)} `), Dm('learning '), K(on ? 'on' : 'off', `share ${x.checkpoint_id} with @${u} learning ${on ? 'off' : 'on'}`), S(' '), K('unshare', `unshare ${x.checkpoint_id} from @${u}`), S(i < x.people.length - 1 ? ', ' : '')]; }),
        S('  '), K('budget', `budget ${x.checkpoint_id}`), S(' '), K('usage', `usage ${x.checkpoint_id}`)))
        : [line(Dm('  none. '), K('share <model> with @name', 'share '))])] };
}

async function sharedTalk(ctx, item) {
  const cid = item.checkpoint_id;
  const open = (s) => {
    if (ctx.st.chat) ctx.st.chat.stopped = true;
    const ch = ctx.st.chat = { mac: SHARED + cid, name: `@${item.owner}'s`, blocks: new Map(), s: s.session || {}, shared: true,
      token: s.machine?.token_id, learn: item.teach === false ? false : undefined, teach: item.teach !== false };
    setBaseline(ch, 0); ctx.setModel(item.name); chatPoll(ctx);
    return { path: '/talk', sugg: ['/pick', '/learn on', '/exit'],
      render: () => [line(Dm('talking to '), Hi(ascii(item.name)), Dm(`, @${ascii(item.owner)}'s, shared. `),
        Dm(ch.teach === false ? 'you can read it, not teach it: it keeps nothing of you for anyone else.'
          : learnOf(ch) ? 'learning is on: what you say teaches it, for everyone.' : 'learning is off: it reads only, and keeps nothing of you for anyone else.'))] };
  };
  const s = await get(`/shared/${cid}/session?rev=-1&wait=0`);
  if (s.machine && s.session?.loaded) return open(s);
  if (s.machine) {
    const b = ctx.printLive(() => [line(Dm(`${ascii(item.name)} is starting on ${ascii(s.machine.gpu || 'a gpu')} ...`))]);
    for (let i = 0; i < 360; i += 1) {
      await new Promise((z) => setTimeout(z, 5000));
      const n = await get(`/shared/${cid}/session?rev=-1&wait=0`).catch(() => null);
      if (n?.session?.loaded) { ctx.update(b, () => [line(Dm('loaded.'))]); return open(n); }
      if (n && !n.machine) { ctx.update(b, () => [line(Am('it stopped before it loaded. '), K('try again', `talk to ${cid}`))]); return null; }
    }
    return null;
  }
  const o = await get(`/shared/${cid}/offers`);
  const offers = (o.offers || []).slice(0, 4);
  if (!offers.length) return { render: () => [line(Am(`no suitable gpu is free in ${ascii(o.data_center || 'its data center')} right now.`))] };
  ctx.ask(`start ${ascii(item.name)} on which? [1-${offers.length}, or cancel]> `, (a) => {
    const pick = offers[(parseInt(a, 10) || 0) - 1];
    if (!pick) return ctx.print(() => [line(Dm('nothing started.'))]);
    ctx.ask(`start ${ascii(pick.name)} at $${Number(pick.price).toFixed(2)}/h on @${ascii(item.owner)}'s account? [y/n]> `, async (y) => {
      if (!/^(y|yes)$/i.test(y)) return ctx.print(() => [line(Dm('nothing started.'))]);
      try { await post(`/shared/${cid}/start`, { gpu: pick.id, max_rate: pick.price }); }
      catch (e) { return ctx.print(() => [line(Am(ascii(e.message)))]); }
      const r = await sharedTalk(ctx, item);
      if (r) await ctx.print(r.render || r);
    }, ['yes', 'no']);
  }, offers.map((_, i) => String(i + 1)).concat(['cancel']));
  return { render: () => [line(Dm(`${ascii(item.name)} isn't running. it runs on @${ascii(item.owner)}'s runpod account and stops after ${o.idle_minutes || 20} idle minutes:`)),
    ...offers.map((x, i) => line(`  ${i + 1}. `, S(ascii(x.name)), Dm(`  ${x.memory} gb  $${Number(x.price).toFixed(2)}/h`)))] };
}

// learning (Nov 23, James: on by default): the model's own setting until you change it here
function learnOf(ch) {
  if (!ch) return true;
  if (ch.teach === false) return false;
  return ch.learn ?? ch.s?.settings?.learn ?? true;
}

// a prompt or read the machine refuses (or that fails there) says so here, instead of vanishing (Nov 22)
function failedLater(ctx, r) {
  const ch = ctx.st.chat; const id = r?.command_id;
  if (!id || ch?.mac === LOCAL) return;
  const statusOf = ch?.mac?.startsWith(SHARED) ? () => get(`/shared/${ch.mac.slice(SHARED.length)}/commands/${id}`) : () => get(`/me/commands/${id}`);
  (async () => {
    for (let i = 0; i < 400; i += 1) {
      const s = await statusOf().catch(() => null);
      if (s?.status === 'failed') return ctx.print(() => [line(Am(`that didn't go through: ${ascii(s.result?.error || 'it failed')}`))]);
      if (s?.status === 'done' || s?.status === 'expired') return s?.status === 'expired' ? ctx.print(() => [line(Am('the machine never picked that up; is it still running?'))]) : undefined;
      await new Promise((z) => setTimeout(z, i < 30 ? 1000 : 5000));
    }
  })();
}

async function commandResult(ctx, kind, args = {}) {
  const ch = ctx.st.chat;
  try {
    const r = await command(ctx, kind, args);
    if (ch?.mac === LOCAL) return { ok: true, result: r?.result || r || {} };
    const id = r?.command_id; if (!id) return { ok: true, result: r || {} };
    const statusOf = ch?.mac?.startsWith(SHARED) ? () => get(`/shared/${ch.mac.slice(SHARED.length)}/commands/${id}`) : () => get(`/me/commands/${id}`);
    for (let i = 0; i < 60; i += 1) {
      const s = await statusOf().catch(() => null);
      if (s?.status === 'done') return { ok: true, result: s.result || {} };
      if (s?.status === 'failed') return { ok: false, error: s.result?.error || 'it failed' };
      await new Promise((z) => setTimeout(z, 1000));
    }
    return { ok: false, error: 'no answer yet; /chats shows how it stands' };
  } catch (e) { return { ok: false, error: e.message }; }
}

async function command(ctx, kind, args = {}) {
  const ch = ctx.st.chat; if (!ch) throw new Error('no model open: talk to one');
  if (kind === 'session.prompt' && ch.prefs) args = { ...ch.prefs, ...args };   // /force read, /search, /run (1.5.0)
  if (ch.mac === LOCAL) return here().command(kind, args);
  if (ch.mac.startsWith(SHARED)) return post(`/shared/${ch.mac.slice(SHARED.length)}/commands`, { kind, args });
  return post(`/me/apps/${ch.mac}/commands`, { kind, args });
}

async function chat(ctx, macHint) {
  if (!ctx.user()) return signIn(ctx, 'my models');
  if(!macHint) return libraryPage(ctx,{get,post});        // no chat page: you talk to a model, from your library
  const apps = (await get('/me/apps')).apps || [];
  ctx.st.apps = apps;
  if (!apps.length) {
    return (c) => [...H(c, 'your machines'), line(Dm('none linked.')), blank(),
      line('  ', K('get soma'), Dm('        the app, for mac and windows; it links itself when you sign in')),
      blank(), line(K('cloud'))];
  }
  const n = norm(macHint || '');
  const mac = apps.find((a) => a.token_id === macHint) || apps.find((a) => n && norm(macName(a.name)).includes(n)) || apps.find((a) => a.token_id === ctx.st.chat?.mac) || apps.find((a) => a.online) || apps[0];
  if (!ctx.st.chat || ctx.st.chat.mac !== mac.token_id) { if (ctx.st.chat) ctx.st.chat.stopped = true; ctx.st.chat = { mac: mac.token_id, name: macName(mac.name), blocks: new Map(), s: {} }; }
  const ch = ctx.st.chat;
  const first = await fetch(`${net.base}/me/apps/${mac.token_id}/session?rev=-1&wait=0`, auth()).then((r) => (r.ok ? r.json() : {})).catch(() => ({}));
  ch.s = first.session || {}; ch.baseline = null;
  const s = ch.s; const ms = s.models || [];
  ctx.st.list = ms.map((m) => ({ kind: 'app-model', id: m.name }));
  return {
    path: '/talk', sugg: s.loaded ? ['/files', '/save', '/exit'] : ms.slice(0, 2).map((m) => `talk to ${m.name}`).concat(['machines']),
    after: () => { if (s.loaded) { ctx.setModel(s.model || 'soma'); showTurns(ctx, ch); chatPoll(ctx); } },
    render: (c) => [...H(c, ascii(ch.name), mac.online ? 'online' : 'offline'),
      ...(apps.length > 1 ? [line(Dm('machines: '), ...apps.flatMap((a, i) => [S(macName(a.name)), Dm(a.online ? '' : ' (offline)'), S(i < apps.length - 1 ? '  ' : '')]))] : []),
      ...(!mac.online ? P(c, Am('soma on ' + ch.name + " isn't online: open it, and check its logOS page allows chat from logossoma.com.")) : []),
      ...(s.loaded ? [line(Dm('talking to '), Hi(ascii(s.model || '')), Dm(`  ${s.where ? ascii(s.where) : ''}${s.dirty ? '  unsaved' : ''}`))]
        : [...P(c, 'models on ' + ch.name + ':'), ...ms.map((m, i) => line(Dm(pad(String(i + 1), 4)), K(ascii(m.name), `talk to ${m.name}`), Dm(m.lives_on ? `  lives on ${ascii(m.lives_on)}` : ''))),
          ...(ms.length ? [] : [line(Dm('  none yet: make one in soma.'))])]),
    ],
  };
}

async function talk(ctx, q) {
  if (/^(the )?demo$/.test(q.trim())) {             // no hosted demo (Nov 20): a model runs where you put it
    return (c) => [...H(c, 'talk to a model'), line(K('public models'), S('   '), K('get soma'), S('   '), K('sign in'))];
  }
  if (here()) {                                            // in the app: this machine's own models answer directly
    const n1 = norm(q); const mine = (await localModels()).find((c) => norm(c.name) === n1 || c.checkpoint_id === q.trim());
    if (mine) return localTalk(ctx, mine.name);
    if (!ctx.user()) return localModelsPage(ctx);
  }
  if (!ctx.user()) return signIn(ctx, `talk to ${q}`);
  const all=(await get('/me/library')).models;
  const n0=norm(q);
  const selected=all.find((m)=>[m.checkpoint_id,m.library_entry_id].includes(q.trim())) || all.find((m)=>norm(m.name)===n0);
  if(!selected) {
    const sh = ((await get('/me/shared').catch(() => ({})))?.with_me || []).find((x) => x.checkpoint_id === q.trim() || norm(x.name) === n0);
    if (sh) return sharedTalk(ctx, sh);
    const apps = ((await get('/me/apps').catch(() => ({})))?.apps || []);
    if (q.trim() && apps.some((a) => a.token_id === q.trim() || norm(macName(a.name)).includes(norm(q)))) return chat(ctx, q.trim());
    return libraryPage(ctx,{get,post},q.trim()?`no '${cut(ascii(q),30)}' in your library`:'talk to which?');
  }
  // places (Nov 20): talking needs it loaded; loaded, it answers; not, it asks where to load it
  return loadCmd(ctx, { get, post }, selected.library_entry_id || selected.checkpoint_id, '', placeHandlers(ctx));
}

// what to do once a model is where it can run
export function placeHandlers(ctx) {
  const api = { get, post, request, chat };
  const onComputer = async (ps, place) => {
    const me = await thisComputer(ps);
    const name = place?.local_name || ps.model.name;
    if (me && place && me.id === place.id) return localTalk(ctx, name);
    return openOnComputer(ctx, place.token_id, name);
  };
  return {
    computer: onComputer,
    gpu: (ps,dc,approved) => runCloud(ctx, `cloud start ${ps.model.ref}`, {...api,data_center:dc,approved}),
    talk: (ps) => {
      const l = ps.loaded;
      if (l.kind === 'gpu') return runCloud(ctx, `cloud chat ${l.machine_id}`, api);
      return onComputer(ps, ps.places.find((x) => x.token_id === l.token_id));
    },
  };
}

async function openOnComputer(ctx, token, name) {
  const opened = await chat(ctx, token);
  const ch = ctx.st.chat; if (!ch) return typeof opened === 'function' ? opened : opened.render;
  setBaseline(ch, 0);
  if (!(ch.s.loaded && ch.s.model === name)) await command(ctx, 'session.load', { checkpoint: name, where: 'mac' });
  ctx.setModel(name); chatPoll(ctx);
  return { path: '/talk', sugg: ['/pick', '/learn on', '/files', '/exit'],
    render: () => [line(Dm(ch.s.loaded && ch.s.model === name ? 'talking to ' : 'loading '), Hi(ascii(name)), Dm(` on ${ascii(ch.name)} ...`))] };
}

export async function unloadOn(ctx, ps) {
  const l = ps.loaded; if (!l) return { render: () => [line(Dm(`${ascii(ps.model.name)} isn't loaded.`))] };
  if (l.kind === 'gpu') return runCloud(ctx, `cloud stop ${l.machine_id}`, { get, post, request, chat });
  const me = await thisComputer(ps);
  if (me && me.token_id === l.token_id) await here().command('session.unload', {});
  else await post(`/me/apps/${l.token_id}/commands`, { kind: 'session.unload', args: {} });
  if (ctx.st.chat) { ctx.st.chat.stopped = true; ctx.st.chat = null; ctx.setModel(null); }
  return { render: () => [line(Dm(`unloading ${ascii(ps.model.name)}; it saves where it's stored first.`))] };
}

export async function resumeChat(ctx, saved) {
  if (saved.mac === LOCAL && !here()) return false;
  const r = await sessionOf(saved.mac, -1, 0).catch(() => null);
  const s = r?.session || {};
  if (!s.loaded) return false;
  ctx.st.chat = { mac: saved.mac, name: saved.name, blocks: new Map(), s };
  setBaseline(ctx.st.chat, 0);
  ctx.setModel(s.model || saved.model); chatPoll(ctx);
  return true;
}

export async function chatInput(ctx, raw, docs = []) {
  let t = raw.trim(); const ch = ctx.st.chat;
  let m;
  // /exit saves and stops it (Nov 22, James); on a shared model it leaves, since the others are still talking to it
  // one chat per account (Nov 24, James): the same chat in every tab, so with lanes /exit leaves and the model stays
  if (/^\/(chats|new\s+chat.*|chat\s+.+|discard)$/i.test(t)) return ctx.print(() => [line(Dm('one chat with each model: this is it. '), K('/history'))]);
  if ((m = t.match(/^\/download\s+(chats|chat)(?:\s+(.+))?$/i))) {
    const r = await downloadChats(ctx, (m[2] || '').trim(), m[1].toLowerCase() === 'chats');
    return ctx.print(r.render);
  }
  if (/^\/(exit|quit|bye)$/i.test(t) && ch?.shared) await command(ctx, 'session.leave').catch(() => {});   // your session ends, saved
  if (/^\/(exit|quit|bye|leave)$/i.test(t)) { if (ch) ch.stopped = true; ctx.st.chat = null; ctx.setModel(null); const name = ch?.s?.model || ''; return ctx.print((c) => [line(Dm(ch?.shared ? 'left the conversation; it keeps running for the others, and stops itself when nobody talks to it. ' : 'left the conversation; it stays loaded. '), K(name ? `talk to ${ascii(name)}` : 'talk to <model>', name ? `talk to ${name}` : 'talk to '), Dm(' comes back.'))]); }
  if (/^\/unload$/i.test(t) && ch?.shared) {
    const cid = ch.mac.slice(SHARED.length);
    ctx.ask('this stops it for everyone it is shared with. stop it? [y/n]> ', async (a) => {
      if (!/^(y|yes)$/i.test(a)) return ctx.print(() => [line(Dm('left running.'))]);
      try { await post(`/shared/${cid}/stop`); } catch (e) { return ctx.print(() => [line(Am(ascii(e.message)))]); }
      ch.stopped = true; ctx.st.chat = null; ctx.setModel(null);
      return ctx.print(() => [line(Dm('stopping: it saves on its volume first.'))]);
    }, ['yes', 'no']);
    return true;
  }
  if (/^\/unload$/i.test(t)) {
    await command(ctx, 'session.unload'); if (ch) ch.stopped = true; ctx.st.chat = null; ctx.setModel(null);
    const name = ascii(ch?.s?.model || 'the model'); const cloud = ch?.s?.cloud;
    const b = ctx.printLive(() => [line(Dm(`unloading ${name}; it saves first ...`))]);
    // follow it out: the session closes; a cloud machine then saves onto its volume and stops
    (async () => {
      for (let i = 0; i < 360; i += 1) {
        await new Promise((z) => setTimeout(z, 5000));
        if (cloud) {
          const m = ((await get('/me/cloud').catch(() => null))?.machines || []).find((x) => x.token_id === ch.mac);
          if (m?.ended_at) return ctx.update(b, () => [line(Hi(`${name} is unloaded`), Dm(m.status === 'ended' ? `: saved on its volume in ${ascii(m.data_center || '')}, and the machine has stopped.` : `; the machine ${ascii(m.status)}: ${ascii(m.note || '')}`))]);
        } else {
          const s = (await sessionOf(ch.mac, -1, 0).catch(() => null))?.session;
          if (s && !s.loaded && !s.opening) return ctx.update(b, () => [line(Hi(`${name} is unloaded`), Dm(`; saved where it's stored, on ${ascii(ch.name)}.`))]);
        }
      }
      ctx.update(b, () => [line(Am(`${name} is still unloading after half an hour; `), K('cloud'), Dm(' shows how it stands.'))]);
    })();
    return true;
  }
  if (/^\/backup$/i.test(t)) {
    if (ch?.shared) return ctx.print(() => [line(Dm("backing up is its owner's to do."))]);
    await command(ctx, 'model.backup', { checkpoint: ch?.s?.model || '' });
    return ctx.print(() => [line(Dm('backing it up to logOS as it is now; it keeps running where it is. '), K('history', `history ${ch?.s?.checkpoint_id || ''}`), Dm(' shows the new save when it lands.'))]);
  }
  if (/^\/save$/i.test(t)) { await command(ctx, 'session.save'); return ctx.print(() => [line(Dm('saving. soma marks it when it has.'))]); }
  // /stop (Nov 23, James): stops what it's doing now, never the session. a read pauses where it is, and /resume or the
  // next message carries it on; a reply stops. /exit ends the session.
  if (/^\/stop$/i.test(t)) {
    if (!ch?.s?.busy) return ctx.print(() => [line(Dm('model not busy.'))]);
    const r = await commandResult(ctx, 'session.stop');
    if (!r.ok && /doesn't do|isn't something/.test(r.error || '')) { await command(ctx, 'session.cancel'); return ctx.print(() => [line(Dm('stopped.'))]); }
    if (!r.ok) return ctx.print(() => [line(Am(ascii(r.error)))]);
    const s = r.result || {};
    if (!s.stopped) return ctx.print(() => [line(Dm('model not busy.'))]);
    if (s.stopped === 'read') return ctx.print(() => [line(Dm('stopped reading '), Hi(ascii(s.file || '')), Dm(s.total ? ` at ${Math.floor(100 * s.position / s.total)}%. ` : '. '), K('/resume'), Dm(', or your next message, carries on from there.'))]);
    return ctx.print(() => [line(Dm('stopped the reply.'))]);
  }
  if (/^\/cancel$/i.test(t)) { await command(ctx, 'session.cancel'); return ctx.print(() => [line(Dm('cancelled: the rest of that prompt is dropped.'))]); }
  if (/^\/resume$/i.test(t)) { await command(ctx, 'session.resume'); return ctx.print(() => [line(Dm('carrying on.'))]); }
  if ((m = t.match(/^\/(stop|resume|skip)\s+(\d{1,2})$/i))) {     // one read, by its number (Nov 25): /stop 2 stops read 2 where it is
    const which = m[1].toLowerCase() === 'stop' ? 'pause' : m[1].toLowerCase(); const n = Number(m[2]);
    const r = liveReads(ch).find((x) => (x.n || 1) === n);
    if (!r) return ctx.print(() => [line(Am(`no read ${n} running. `), K('/lanes'))]);
    await command(ctx, `session.${which}`, { read: n });
    const done = { pause: 'stopping ', resume: 'carrying on with ', skip: 'ending ' }[which];
    return ctx.print(() => [line(Dm(done), Hi(ascii(r.name)), Dm(which === 'skip' ? ' there.' : '.'))]);
  }
  if (/^\/lanes$/i.test(t)) return ctx.print(lanesView(ch));
  // /auto read (Nov 25, James): lanes that read the model's whole folder on a loop, evenly spread, yielding to seats
  if ((m = t.match(/^\/auto\s*read\s+(on|off|\d{1,2})$/i))) {
    const lanes = /^on$/i.test(m[1]) ? 4 : /^off$/i.test(m[1]) ? 0 : Number(m[1]);
    const ref = ch?.s?.checkpoint_id || ch?.s?.model;
    try { if (ref) await post(`/me/models/${encodeURIComponent(ref)}/auto-read`, { lanes }); } catch (e) { return ctx.print(() => [line(Am(ascii(e.message)))]); }
    const r = await commandResult(ctx, 'session.auto_read', { lanes });
    if (!r.ok) return ctx.print(() => [line(Am(ascii(r.error || 'it could not be set')))]);
    return ctx.print(() => [line(Dm(lanes ? `auto read: ${lanes} lane${lanes === 1 ? '' : 's'} reading its files. ` : 'auto read off. '), K('/lanes'))]);
  }
  if (/^\/skip$/i.test(t)) { await command(ctx, 'session.skip'); return ctx.print(() => [line(Dm('ending the read there; the rest of the prompt goes on.'))]); }
  if (/^\/(files|delete file|remove file|rename file|take file)\b/i.test(t)) {       // the model's own files (files.js)
    const done = await sessionFiles(ctx, t, ch, { commandResult });
    if (done) return done;
  }
  if ((m = t.match(/^\/score\s+(.+)$/i))) {
    const name = m[1].trim();
    failedLater(ctx, await command(ctx, 'session.prompt', { parts: [{ file: name }], learn: false, score: true }));
    return ctx.print(() => [line(Dm('scoring '), Hi(ascii(name)), Dm(' with learning off; it goes on the model\'s '), K('curve', `curve ${ch?.s?.checkpoint_id || ch?.s?.model || ''}`))]);
  }
  if (/^\/who$/i.test(t)) {
    const s = ch?.s || {};
    return ctx.print(() => [line(Dm('here: '), S((s.present || []).map((x) => `@${ascii(x)}`).join(', ') || 'just you')),
      line(Dm(s.serving ? `answering ${ascii(s.serving)}` : 'free'), Dm((s.queue || []).length ? `; waiting: ${(s.queue || []).map((x) => ascii(x.who)).join(', ')}` : ''))]);
  }
  // somatic core 1.5.0 (Nov 22, James): force read off (the default) lets a model take its turn while it reads, where it
  // predicts it; search and run are its two skills. Each is yours, per model, sent with what you say next.
  if ((m = t.match(/^\/(images|videos)(?:\s+(on|off))?$/i))) {
    const key = m[1].toLowerCase(); const chatId = ch?.s?.chat || '-';
    const now = mediaPrefs(chatId)[key];
    const want = m[2] ? m[2].toLowerCase() === 'on' : !now;
    setMediaPref(chatId, key, want);
    return ctx.print(() => [line(Dm(key === 'images' ? (want ? 'images show where they appear, in this chat.' : 'images are a click away, in this chat.')
      : (want ? 'videos play where they appear, in this chat.' : 'videos are a click away, in this chat (the default).'))),
      line(Dm('images and videos load from their own sites. they see your address, not the page.'))]);
  }
  if ((m = t.match(/^\/(force\s+read|search|run)(?:\s+(on|off))?$/i))) {
    const key = { search: 'search', run: 'run' }[m[1].toLowerCase()] || 'force_read';
    const now = ch?.prefs?.[key] ?? ch?.s?.settings?.[key] ?? true;
    const want = m[2] ? m[2].toLowerCase() === 'on' : !now;
    if (ch) ch.prefs = { ...(ch.prefs || {}), [key]: want };
    const say = {
      force_read: want ? 'force read on: it reads straight through and waits for a real turn.' : 'force read off: while it reads, it takes its turn wherever it predicts it, then goes back to the file.',
      search: want ? 'search on: it can look things up and read links it has seen (3 a turn, 30 an hour). each one shows here.' : 'search off: it can\'t look anything up.',
      run: want ? (ch?.shared ? 'run on, for when its owner allows it: it runs on their machine.' : 'run on: it can run python in a sandbox with no network. each run shows here.') : 'run off: it can\'t run code.',
    }[key];
    return ctx.print(() => [line(Dm(say)), line(Dm('from what you say or read next.'))]);
  }
  if ((m = t.match(/^\/learn(?:\s+(on|off))?$/i))) {
    const want = m[1] ? m[1].toLowerCase() === 'on' : !learnOf(ch);
    if (ch?.shared && want && ch.teach === false) {
      return ctx.print(() => [line(Dm('its owner has turned learning off for you on this model: you read it, and nothing of yours changes it.'))]);
    }
    if (ch) ch.learn = want;
    if (ch?.shared) return ctx.print(() => [line(Dm('learning: '), Hi(ch.learn ? 'on' : 'off'),
      Dm(ch.learn ? '. what you say and read now teaches the shared model, for everyone it is shared with.' : '. it reads only: nothing of yours reaches anyone else.'))]);
    return ctx.print(() => [line(Dm('learning from what it reads and hears: '), Hi(ch?.learn ? 'on' : 'off'), Dm(ch?.learn ? '' : '  (it reads with no weight updates)'))]);
  }
  // /paste <file> (Nov 25, James): one of the model's files, read into this chat's store, so it's in working memory
  if ((m = t.match(/^\/paste(?:\s+(.+))?$/i))) {
    const names = [...(ch?.s?.data || []), ...(ch?.s?.shared || [])].map((f) => f.name);
    const go = async (name) => {
      if (names.length && !names.includes(name)) return ctx.print(() => [line(Am(`no file '${ascii(name)}'. `), K('/files'))]);
      failedLater(ctx, await command(ctx, 'session.prompt', { parts: [{ file: name, into: 'chat' }], learn: learnOf(ch) }));
      return ctx.print(() => [line(Dm('into this chat: '), Hi(ascii(name)))]);
    };
    if (m[1]) return go(m[1].trim());
    ctx.ask('paste which file? > ', (a) => (a.trim() ? go(a.trim()) : null), names.slice(0, 8));
    return true;
  }
  if (/^\/(pick|choose)$/i.test(t)) {
    if (ch && ch.mac !== LOCAL && ch.s?.cloud && ctx.pick) {
      // a model on a cloud gpu reads from its machine's own data folder: the file goes there first (transfer.js)
      ctx.pick(async (files) => {
        for (const f of files) {
          const token = ch.mac.startsWith(SHARED) ? ctx.st.chat?.token : ch.mac;
          const statusOf = ch.mac.startsWith(SHARED) ? (id) => get(`/shared/${ch.mac.slice(SHARED.length)}/commands/${id}`) : null;
          let name; try { name = await sendToMachine(ctx, { get, post }, token, f, statusOf); } catch (_) { continue; }
          for (let i = 0; i < 30 && !(ctx.st.chat?.s?.data || []).some((x) => x.name === name); i += 1) await new Promise((z) => setTimeout(z, 1000));
          if (!(ctx.st.chat?.s?.data || []).some((x) => x.name === name)) { await ctx.print(() => [line(Am(`${ascii(name)} hasn't reached the machine yet; /files shows when it has.`))]); continue; }
          failedLater(ctx, await command(ctx, 'session.prompt', { parts: [{ file: name }], learn: learnOf(ctx.st.chat) }));
          await ctx.print(() => [line(Dm('reading '), Hi(ascii(name)), Dm(learnOf(ctx.st.chat) ? ' on the machine' : ' on the machine (no weight updates)'))]);
        }
      });
      return true;
    }
    if (!window.soma?.addFiles || !ctx.pick) return ctx.print(() => [line(Dm('choosing files works in the soma app, or while talking to a model on a cloud gpu; here, '), K('/files'))]);
    ctx.pick(async (files) => {
      const added = await window.soma.addFiles(files);
      for (const f of added) {
        // the app puts it in soma's data folder; it belongs to the loaded model (Nov 23: each model has its own folder)
        const took = await commandResult(ctx, 'data.take', { name: f.name, replace: true });
        if (!took.ok) { await ctx.print(() => [line(Am(`${ascii(f.name)}: ${ascii(took.error || 'it could not be moved in')}`))]); continue; }
        failedLater(ctx, await command(ctx, 'session.prompt', { parts: [{ file: f.name }], learn: learnOf(ctx.st.chat) }));
        await ctx.print(() => [line(Dm('reading '), Hi(ascii(f.name)), Dm(` (${fmtBytes(f.size)})${learnOf(ctx.st.chat) ? '' : ' (no weight updates)'}`))]);
      }
    });
    return true;
  }
  if ((m = t.match(/^\/read\s+(.+)$/i))) {
    const name = m[1].trim(); const data = [...(ch?.s?.data || []), ...(ch?.s?.shared || [])];
    if (data.length && !data.some((f) => f.name === name)) return ctx.print(() => [line(Am(`no file '${ascii(name)}' among ${ascii(ch?.s?.model || 'its')} files. `), K('/files'))]);
    failedLater(ctx, await command(ctx, 'session.prompt', { parts: [{ file: name }], learn: learnOf(ch) }));
    const traces = /\.traces\.jsonl$/i.test(name);
    return ctx.print(() => [line(Dm('reading '), Hi(ascii(name)), Dm(learnOf(ch) ? '' : ' (no weight updates)'), Dm(traces ? '  (recorded turns: read as conversations)' : '')),
      ...(traces && !learnOf(ch) ? [line(Am('learning is off: read only. to learn it: /skip, /learn on, /read it again.'))] : [])]);
  }
  if (t.startsWith('/')) return null;                                            // a site command: the caller runs it
  // long text (Nov 25, James): a paste, or words past what a message carries, goes in as a document: put in the
  // model's files, then read into this chat's store before the words
  const long = [...docs]; if (t.length > MESSAGE_MAX) { long.push(t); t = ''; }
  if (!t && !long.length) return;
  const mac = (ctx.st.apps || []).find((a) => a.token_id === ch?.mac);
  const pasted = [];
  for (const [i, doc] of long.entries()) {
    const stamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14);
    const name = await putOnMachine(ctx, ch, new File([doc], `pasted-${stamp}-${i + 1}.txt`, { type: 'text/plain' }));
    if (!name) return true;
    pasted.push({ file: name, into: 'chat' });
  }
  failedLater(ctx, await command(ctx, 'session.prompt', { parts: [...pasted, ...(t ? [{ text: t }] : [])], learn: learnOf(ch) }));
  // logOS takes the message even when soma isn't there to read it: say so, rather than wait in silence
  if (mac && !mac.online) await ctx.print(() => [line(Am(`soma on ${ch.name} looks offline: this waits until it's back.`))]);
  return true;
}

// ---------- the table ----------
export async function run(ctx, raw) {
  // a command can be typed with a slash too ("/profile", "/developments"): outside a conversation the slash is dropped
  const t = raw.trim().replace(/^\/(?=[a-z])/i, ''); const low = t.toLowerCase(); let m;
  if (/^runpod setup$/.test(low)) return runpodSetup(ctx);      // before the cloud commands: it's for people with no account yet
  const fl = await filesCommand(ctx, t, { get, post });         // a model's files, before library's delete/rename of models
  if (fl) return fl;
  const cloud = await runCloud(ctx, t, { get, post, request, chat });
  if (cloud) return cloud;
  const lib=await libraryCommand(ctx,t,{get,post,request,chat,findModel,go:ctx.go,handlers:placeHandlers(ctx),unloadOn});
  if(lib) return lib;
  const R = (render, path, sugg) => ({ render, path, sugg });
  if (/^(home|welcome|start|hello|hi|logos|~)$/.test(low)) {
    const own = ctx.user()?.prefs?.home;                    // your home: a command of your choosing, or the home page
    if (own && !/^(home|welcome|start|~)$/i.test(own)) return run(ctx, own);
    return R(home(ctx), '/', ctx.user() ? ['my models', 'public models', 'developments'] : ['public models', 'developments', 'get soma', 'sign in']);
  }
  if (/^home (page|default)$/.test(low)) return R(home(ctx), '/', ['settings', 'my models']);
  if (/^(help|\?|commands|man)$/.test(low)) return R(help(ctx), '/help', ['models', 'developments', 'get soma']);
  if (/^settings$/.test(low)) return settingsPage(ctx);
  // personal access tokens (Nov 25, the open platform): for programs, with only the powers you give them
  if (/^tokens$/i.test(t)) {
    if (!ctx.user()) return signIn(ctx, 'tokens');
    const d = await get('/me/tokens');
    return R((c) => [...H(c, 'tokens'), ...(d.tokens.length ? d.tokens.map((x) => line(Hi(ascii(x.name)), Dm(`  ...${x.hint}  ${x.scopes.join(', ')}${x.spend_cap_usd != null ? ` ($${x.spend_cap_usd}/month)` : ''}  ${x.last_used_at ? `used ${fmtDate(x.last_used_at)}` : 'unused'}${x.expires_at ? `  until ${fmtDate(x.expires_at)}` : ''}   `), K('revoke', `token revoke ${x.name}`)))
      : [line(Dm('none.'))]), blank(), line(K('token new <name>', 'token new '))], null);
  }
  if ((m = t.match(/^token\s+revoke\s+(.+)$/i))) {
    try { await request(`/me/tokens/${encodeURIComponent(m[1].trim())}`, 'DELETE'); return R(() => [line(Dm('revoked: '), Hi(ascii(m[1].trim())))], null); }
    catch (e) { return R(() => [line(Am(ascii(e.message)))], null); }
  }
  if ((m = t.match(/^token\s+new(?:\s+(.+))?$/i))) {
    if (!ctx.user()) return signIn(ctx, 'token new');
    const named = (name) => {
      ctx.ask('which scopes? read, talk, manage, spend > ', (sc) => {
        const scopes = sc.toLowerCase().split(/[\s,]+/).filter(Boolean);
        const make = async (cap, expires) => {
          try {
            const r = await post('/me/tokens', { name, scopes, spend_cap_usd: cap, expires });
            return ctx.printPrivate(() => [line(Hi(ascii(r.name)), Dm(`  ${r.scopes.join(', ')}`)), line(S(r.secret)),
              line(Am('shown once: logOS keeps only its hash, and this line is in no history.'))]);
          } catch (e) { return ctx.print(() => [line(Am(ascii(e.message)))]); }
        };
        const when = (cap) => ctx.ask('expires in? 30, 90, 365 days, or never > ', (ex) => make(cap, ex.trim() || '90'), ['90', '30', '365', 'never']);
        if (scopes.includes('spend')) return ctx.ask('most it may spend a month, in dollars > ', (d) => when(Number(d.replace(/[$\s]/g, ''))), ['5', '20']);
        return when(null);
      }, ['read', 'read, talk', 'read, talk, manage']);
    };
    if (m[1]) { named(m[1].trim()); return R(() => [], null); }
    ctx.ask('name it (what program is it for?) > ', (n) => (n.trim() ? named(n.trim()) : null), []);
    return R(() => [], null);
  }
  if ((m = t.match(/^auto\s*read\s+(.+?)\s+(on|off|\d{1,2})$/i))) {
    if (!ctx.user()) return signIn(ctx, t);
    const lanes = /^on$/i.test(m[2]) ? 4 : /^off$/i.test(m[2]) ? 0 : Number(m[2]);
    try { const r = await post(`/me/models/${encodeURIComponent(m[1].trim())}/auto-read`, { lanes }); return R(() => [line(Hi(ascii(r.model)), Dm(r.auto_read ? `  auto read: ${r.auto_read} lanes, from its next load` : '  auto read off'))], null); }
    catch (e) { return R(() => [line(Am(ascii(e.message)))], null); }
  }
  if ((m = t.match(/^set\s+username(?:\s+(?:to\s+)?(.*))?$/i))) {
    if (!ctx.user()) return signIn(ctx, 'set username');
    const name = (m[1] || '').trim();
    if (!name) { chooseName(ctx); return R(() => [], null); }
    try { const r = await request('/users/me', 'PATCH', { username: name }); Object.assign(ctx.user(), { username: r.username, username_chosen: true }); return R(() => [line(Hi(`@${ascii(r.username)}`))], null); }
    catch (e) { return R(() => [line(Am(ascii(e.message)))], null); }
  }
  if ((m = t.match(/^set\s+(home|start)(?:\s+(?:to\s+)?(.*))?$/i))) return setPref(ctx, m[1].toLowerCase(), (m[2] || '').trim());
  if (/^runpod setup$/.test(low)) return runpodSetup(ctx);
  if ((m = low.match(/^post(?:\s+(paste|file))?$/))) return postDev(ctx, m[1]);
  if (/^(new model|create model|new)$/.test(low)) return newModel(ctx);
  if ((m = t.match(/^read(?:\s+(.*))?$/i))) {           // read is for a conversation: /read <file> while you talk to a model
    const what = (m[1] || '').trim();
    return R(() => [line(Dm('read is for reading a file into a model while you talk to it: '), K('/read <file>', 'talk to '), Dm('. ')),
      line(Dm('the lab notebook is '), K(what ? `developments ${what}` : 'developments'), Dm('.'))], null, [what ? `developments ${what}` : 'developments']);
  }
  // suggestions name what they point at: a number would mean whatever list came last by the time it's clicked
  const firstModel = () => { const it = ctx.st.list.find((x) => x.kind === 'model'); return it ? `open ${it.name || it.id}` : null; };
  if (/^(models|my models|library|ls|list|ls models)$/.test(low) && ctx.user()) return libraryPage(ctx,{get,post});
  if (/^(models|ls|list|browse|public models|ls models)$/.test(low)) { const r = await models(ctx, ''); return R(r, '/models', [firstModel(), 'search ', 'read'].filter(Boolean)); }
  if ((m = t.match(/^(search|find|grep|ls)\s+(.+)$/i))) { const r = await searchAll(ctx, m[2]); return R(r, `/models?q=${encodeURIComponent(m[2])}`, [firstModel(), 'models'].filter(Boolean)); }
  if ((m = t.match(/^(open|show|view|cd)\s+(.+)$/i)) || /^\d+$/.test(low)) {
    const q = m ? m[2] : low;
    if (/^\d+$/.test(q)) { const it = ctx.st.list[+q - 1]; if (it?.kind === 'library-model') return run(ctx, `library ${it.id}`); if (it?.kind === 'dev') return run(ctx, `developments ${it.id}`); if (it?.kind === 'app-model') return run(ctx, `talk to ${it.id}`); }
    if (/^(the )?(chat|models|developments|download|runtime|help)$/.test(q.toLowerCase())) return run(ctx, q.replace(/^the /, ''));
    return open(ctx, q);
  }
  if (/^(developments|reading|notes|news|blog|ls developments)$/.test(low)) { const r = await devList(ctx); const d = ctx.st.list[0]; return R(r, '/developments', [d ? `developments ${d.id}` : 'models', 'models']); }
  if ((m = t.match(/^(developments|reading|cat|less|more)\s+(.+)$/i))) return devPage(ctx, m[2]);
  if (/^(get soma|get|download soma|install|soma|app|downloads)$/.test(low)) return R(await download(ctx), '/download', ['public models', 'runtime']);
  if (/^(runtime|spec|somatic core|tech specs|specs)$/.test(low)) return runtime(ctx);
  if ((m = low.match(/^(runtime|spec)\s+(.+)$/))) return runtime(ctx, m[2].replace(/\s+/g, '-'));
  if ((m = t.match(/^(whois|who is|profile|finger)\s+@?(\S+)$/i))) return whois(ctx, m[2]);
  if (/^about$/.test(low)) return R(await page(ctx, 'about', 'about'), '/about', ['get soma', 'models']);
  if (/^(privacy|safeguards|what logos holds)$/.test(low)) return R(await page(ctx, 'privacy', 'what logOS holds'), '/privacy', ['download chats', 'about']);
  if (/^info$/.test(low)) return R(info(), '/info', ['help']);
  if (/^(sign in|signin|login|log in)$/.test(low)) return R(ctx.user() ? (c) => [line(Dm('signed in as '), K('@' + ctx.user().username, `whois ${ctx.user().username}`))] : signIn(ctx), null);
  if (/^(sign out|signout|logout|log out)$/.test(low)) { await ctx.logout(); return R(() => [line(Dm('signed out.'))], null, ['sign in', 'models']); }
  if (/^(whoami|me|id|profile)$/.test(low)) return ctx.user() ? whois(ctx, ctx.user().username) : R(() => [line(Dm('a visitor. '), K('sign in'))], null);
  if (/^(shared|shared models)$/.test(low)) return sharedPage(ctx);
  if ((m = t.match(/^curve\s+(.+?)(?:\s+on\s+(.+))?$/i))) return curvePage(ctx, m[1].trim(), (m[2] || '').trim());
  // /history (Nov 25, James): your terminal stream, from every device, filled in up to now
  if (/^history$/.test(low)) {
    if (!ctx.user()) return signIn(ctx, 'history');
    const d = await get('/me/terminal?limit=300'); const n = ctx.backfill ? ctx.backfill(d.blocks || []) : 0;
    return R(() => (n ? [] : [line(Dm('nothing earlier.'))]), null);
  }
  if (/^history\s+clear$/.test(low)) {
    if (!ctx.user()) return signIn(ctx, 'history clear');
    ctx.ask('delete your whole terminal stream, on every device? [y/n]> ', async (a) => {
      if (!/^(y|yes)$/i.test(a)) return ctx.print(() => [line(Dm('kept.'))]);
      const r = await request('/me/terminal', 'DELETE'); return ctx.print(() => [line(Dm(`deleted ${r.forgotten}.`))]);
    }, ['no', 'yes']);
    return R(() => [], null);
  }
  if ((m = t.match(/^history\s+(.+)$/i))) return historyPage(ctx, m[1].trim());
  if ((m = t.match(/^usage\s+(.+)$/i))) return usagePage(ctx, m[1].trim());
  if ((m = t.match(/^budget\s+(.+)$/i))) {
    // "budget <model>" shows it; "budget <model> $20/month", "... $1.50/h", "... off" change it. The model name can have spaces,
    // so the change is whatever parses at the end.
    const words = m[1].trim().split(/\s+/);
    for (let i = 1; i < words.length; i += 1) {
      const change = parseBudget(words.slice(i).join(' '));
      if (change) return budgetPage(ctx, words.slice(0, i).join(' '), change);
    }
    return budgetPage(ctx, m[1].trim());
  }
  if (/^(machines|macs|devices)$/.test(low)) return R(await machines(ctx), '/machines', ['my models', 'cloud']);
  if ((m = low.match(/^link\s*(.*)$/))) return R(await link(ctx, m[1].trim()), m[1].trim() ? null : '/machines');
  if ((m = low.match(/^unlink\s+(.+)$/))) {
    const id = m[1].trim();
    ctx.ask('unlink it? it will need to be linked again. [y/n] ', async (a) => {
      if (!/^(y|yes)$/i.test(a)) return ctx.print(() => [line(Dm('kept.'))]);
      await fetch(`${net.base}/auth/app-tokens/${encodeURIComponent(id)}`, auth({ method: 'DELETE' }));
      return ctx.print(() => [line(Dm('unlinked.'))]);
    }, ['yes', 'no']);
    return R(() => [line(Dm('unlinking an app signs it out of logOS.'))], null);
  }
  if ((m = t.match(/^chats(?:\s+(?:with\s+)?(.+))?$/i))) return chatsPage(ctx, (m[1] || '').trim());
  if ((m = t.match(/^chat\s+history\s+(.+)$/i))) return chatHistory(ctx, m[1]);
  if ((m = t.match(/^rename\s+chat\s+(\S+)\s+to\s+(.+)$/i))) {
    try { const r = await post(`/me/chats/${encodeURIComponent(chatRef(ctx, m[1]))}/rename`, { name: m[2].trim() }); return R(() => [line(Dm('renamed: '), Hi(ascii(r.name)))], null, ['chats']); }
    catch (e) { return R(() => [line(Am(ascii(e.message)))], null); }
  }
  if ((m = t.match(/^delete\s+chat\s+(\S+)$/i))) {
    const id = chatRef(ctx, m[1]);
    ctx.ask('delete this chat? its store and its whole history go, for good. [y/n]> ', async (a) => {
      if (!/^(y|yes)$/i.test(a)) return ctx.print(() => [line(Dm('kept.'))]);
      try { const r = await request(`/me/chats/${encodeURIComponent(id)}`, 'DELETE'); return ctx.print(() => [line(Dm(`deleted. its store: ${ascii(r.store)}.`))]); }
      catch (e) { return ctx.print(() => [line(Am(ascii(e.message)))]); }
    }, ['no', 'yes']);
    return R(() => [line(Am('a deleted chat can\'t be brought back.'))], null);
  }
  if (/^(chat|talk)(\s+(to|with))?$/i.test(t)) {
    if (ctx.st.chat?.s?.loaded) { const ch = ctx.st.chat; return R(() => [line(Dm('talking to '), Hi(ascii(ch.s.model || ch.name || 'a model')), Dm('. type to it, or '), K('/exit'), Dm(' to save and stop.'))], '/talk', ['/files', '/exit']); }
    return chat(ctx, '');
  }
  if ((m = t.match(/^(talk to|talk with|talk|chat to|chat with|chat|load)\s+(.+)$/i))) return talk(ctx, m[2]);
  if ((m = low.match(/^(un)?star\s*(.*)$/))) {
    const id = await findModel(ctx, m[2]);
    if (!id) return R(() => [line(Dm('star which? open a model first, or say '), K('star 1'))], null);
    if (!ctx.user()) return R(signIn(ctx, `${m[1] ? 'un' : ''}star ${id}`), null);
    const r = await post(`/checkpoints/${id}/star`);
    return R(() => [line(r.starred ? Hi('starred.') : Dm('unstarred.'), Dm(`  ${r.star_count} ${r.star_count === 1 ? 'star' : 'stars'}`))], null, [`open ${id}`]);
  }
  if ((m = t.match(/^download\s+chats$/i))) return downloadChats(ctx, '', true);
  if ((m = t.match(/^download\s+chat(?:\s+(.+))?$/i))) return downloadChats(ctx, (m[1] || '').trim(), false);
  if (/^(close|delete)\s+(my\s+)?account$/i.test(t)) return closeAccount(ctx);
  if (/^(notices|notifications|alerts|storage|credit)$/i.test(t)) return noticesPage(ctx);
  if ((m = low.match(/^(download|get)\s+(.+)$/)) || low === 'download') {
    const id = await findModel(ctx, m ? m[2] : '');
    if (!id) return R(() => [line(Dm('download which? open a model first. the app: '), K('get soma'))], null);
    const r = await post(`/checkpoints/${id}/download`);
    ctx.go(r.download_url);
    return R(() => [line(Dm('downloading '), Hi(fmtBytes(r.file_size_bytes || 0))), ...P(80, Dm('open the file in soma to load it, or save a fork of it there.'))], null);
  }
  if ((m = low.match(/^fork\s*(.*)$/))) {
    const id = await findModel(ctx, m[1]);
    if (!id) return R(() => [line(Dm('fork which? open a model first, or say '), K('fork 1'))], null);
    return R((c) => [...P(c, 'a fork starts where its parent got to and grows on its own. today it happens in soma: '), line('  1. ', K('download', `download ${id}`), Dm(' the model')),
      line(Dm('  2. open it in soma, and save a fork')), blank(), ...P(c, Dm('forking straight onto your machine or the cloud from here is coming.'))], null);
  }
  if (/^upload$/.test(low)) return ctx.user() ? uploadPage(ctx, { get, post }) : signIn(ctx, 'upload');
  if (/^admin(\s|$)/.test(low)) return ctx.user() ? adminCommand(ctx, t, { get, post, request }) : signIn(ctx, 'admin');
  if ((m = t.match(/^goto\s+(.+)$/i))) return R(() => [line(Dm(`nothing at ${cut(ascii(m[1]), 40)}. `), K('home'), Dm(' or '), K('help'))], null);
  // nothing matched: a search, so nothing typed is just an error
  const known = ctx.known();
  if (!known.length) await get('/checkpoints?limit=60').then((l) => ctx.remember(l)).catch(() => {});
  if (!ctx.knownDevs().length) await get('/developments').then((d) => ctx.rememberDevs(d.developments || [])).catch(() => {});
  return R(notFound(ctx, t), null, ['help', 'models']);
}

export const COMPLETE = ['models', 'open ', 'read ', 'search ', 'talk to ', 'star', 'download', 'fork', 'help', 'get soma', 'runtime', 'sign in', 'sign out', 'whoami',
  'my models','public models','library ','where ','move ','add ','publish ',
  'delete model ', 'move ', 'warm ', 'set username', 'history clear', 'tokens', 'token new ', 'rename ', 'cloud', 'cloud history', 'runpod key', 'runpod forget', 'cloud offers ', 'cloud start ', 'cloud status ', 'cloud stop ',
  'whois ', 'machines', 'link ', 'unlink ', 'about', 'info', 'upload', 'history', 'clear', 'back', 'home', 'profile', 'settings', 'set home ', 'set start ',
  'developments', 'developments ', 'reading', 'runpod setup', 'chat to ', '/leave', '/post', 'new model', 'chats', 'chat history ', 'delete chat ',
  '/history', '/download chat', '/download chats', 'download chat ', 'download chats', 'close account', 'privacy', 'notices',
  'load ', 'unload ', 'share ', 'unshare ', 'shared models', 'budget ', 'usage ', 'curve ', 'history ', 'cloud adopt ',
  'files ', 'delete file ', 'rename file ', '/delete file ', '/rename file ', '/take file ',
  '/files', '/read ', '/save', '/backup', '/unload', '/exit', '/stop', '/pick', '/learn on', '/learn off', '/score ', '/who', '/resume', '/skip', '/lanes', '/history', '/paste ', '/auto read on', '/auto read off', '/force read on', '/force read off', '/search on', '/search off', '/run on', '/run off', '/images on', '/images off', '/videos on', '/videos off'];

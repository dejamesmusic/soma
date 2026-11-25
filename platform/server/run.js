// the language over http (Nov 26, spec Part 2): logOS runs the terminal's own JavaScript headless, one short process
// per call, so /api/run, the command-line tool and agents get exactly what the terminal would draw.
//
// stdin:  { line, state, cols, confirm, wait, base, headers, session }
//   line     what was typed: a /command, words for an open model, or the answer to a pending question
//   state    what this session remembers (from the last call's `state`); null for a fresh one
//   wait     seconds to stay for a model's reply to words sent now (0: return as soon as they're sent)
//   base     logOS's /api, reached from inside the server; headers: the caller's own credentials
// stdout: { blocks, text, question, status, data, links, state }
//
// Nothing here is trusted more than the caller: every request goes back through logOS's front door with the caller's
// credentials, so a command can do what its caller could and nothing else.
/* eslint-disable no-console */

// ---- a browser's worth of globals, empty: the commands that need a real browser say so and stop ----
const memory = () => { const m = new Map(); return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, String(v)), removeItem: (k) => m.delete(k) }; };
globalThis.window = {
  location: { href: '', pathname: '/', search: '', origin: '' },
  history: { pushState() {}, back() {} },
  addEventListener() {}, removeEventListener() {},
};
globalThis.document = {
  visibilityState: 'visible', title: '', addEventListener() {}, removeEventListener() {},
  createElement: () => ({ click() {}, remove() {}, style: {} }), body: { appendChild() {} },
};
globalThis.localStorage = memory();
globalThis.sessionStorage = memory();
// the commands' own logging goes to stderr: stdout is the answer
console.log = console.info = console.warn = console.debug = (...a) => process.stderr.write(`${a.join(' ')}\n`);

const { net } = require('../src/terminal/net');
const { execute } = require('../src/terminal/shell');
const { resumeChat } = require('../src/terminal/commands');
const { line, Dm, Am, K } = require('../src/terminal/text');

const KEEP = ['list', 'here', 'model', 'apps', 'filesModel', 'models', 'devs'];
const DATA_MAX = 256 * 1024;
const WAIT_MAX = 120;
const TERMINAL = new Set(['done', 'stopped', 'failed']);

const plain = (lines) => lines.map((l) => l.map((g) => g.t).join('')).join('\n');
const segs = (lines) => lines.map((l) => l.map((g) => { const o = { t: g.t }; if (g.cls) o.cls = g.cls; if (g.cmd !== undefined) o.cmd = g.cmd; if (g.href) o.href = g.href; if (g.media) o.media = g.media; return o; }));

async function main(input) {
  const cols = Math.max(40, Math.min(200, Number(input.cols) || 100));
  const saved = input.state || {};
  const session = String(input.session || 'api');
  let n = Number(saved.n) || 0;
  net.base = input.base;
  net.headers = () => ({ ...(input.headers || {}) });

  // what the commands fetched: their JSON, for programs that want structure as well as the page
  const data = []; let dataSize = 0; const quiet = { on: false };
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (url, init = {}) => {
    const r = await realFetch(url, init);
    const u = String(url);
    if (!quiet.on && (!init.method || init.method === 'GET') && u.startsWith(net.base) && !/\/session\?rev=|\/auth\/me$/.test(u) && r.ok) {
      r.clone().json().then((body) => {
        const size = JSON.stringify(body).length;
        if (dataSize + size <= DATA_MAX) { dataSize += size; data.push({ path: u.slice(net.base.length), body }); }
      }).catch(() => {});
    }
    return r;
  };

  const me = await realFetch(`${net.base}/auth/me`, { headers: net.headers() }).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const user = me && (me.id || me.user_id || me.username) ? me : null;

  const st = { list: [], here: null, model: null, ask: null, chat: null, apps: [], models: new Map(), devs: new Map() };
  for (const k of KEEP) if (saved[k] !== undefined) st[k] = k === 'models' || k === 'devs' ? new Map(saved[k]) : saved[k];

  const blocks = []; const links = []; const tasks = [];
  let status = null; let rows = null; let lastSugg = [];
  const block = (render, echo, opts = {}) => {
    const b = { id: `api-${session}-${(n += 1)}`, render, echo: echo || null, private: !!opts.private, kind: opts.kind || (echo ? 'command' : 'note'), model: opts.model || st.model || null };
    if (!quiet.on) blocks.push(b);
    return b;
  };
  const ctx = {
    st,
    user: () => user,
    login: () => Promise.reject(new Error('signing in happens in a terminal: logossoma.com')),
    logout: async () => { throw new Error("a program can't sign you out: revoke its key instead"); },
    run: (cmd) => { const p = execute(ctx, cmd.startsWith('/') ? cmd : `/${cmd}`, ui); tasks.push(p); return p; },
    navigate: () => {},
    go: (url) => { if (!quiet.on) links.push(String(url)); },
    print: async (render, echo) => block(render, echo),
    printLive: (render, meta = {}) => block(render, null, meta),
    printPrivate: async (render) => block(render, null, { private: true }),
    update: (b, render) => { b.render = render; },
    suggest: (items) => { lastSugg = items || []; },
    setStatus: (s) => { status = s; },
    setRows: (r) => { rows = r; },
    backfill: () => 0,
    ask: (q, answer, sugg, options = {}) => { st.ask = { ps: q, answer, secret: !!options.secret }; lastSugg = sugg || []; },
    active: () => true,
    setModel: (name) => { st.model = name || null; if (!name) { status = null; rows = null; } },
    remember: (list) => { for (const m of list || []) if (m?.checkpoint_id) st.models.set(m.checkpoint_id, { checkpoint_id: m.checkpoint_id, name: m.name }); },
    known: () => [...st.models.values()],
    rememberDevs: (list) => { for (const d of list || []) if (d?.slug) st.devs.set(d.slug, { slug: d.slug, title: d.title }); },
    knownDevs: () => [...st.devs.values()],
  };
  const ps = () => (st.ask ? st.ask.ps : st.model ? `${st.model.slice(0, 24)}> ` : '> ');
  const ui = { ps, echo: true };
  // a pasted document (the MCP server's paste): read into the chat's store before the words, as a long paste is
  if (input.paste) ui.pasted = (text) => ({ words: text.replace(/\[pasted document[^\]]*\]/g, '').trim(), docs: [String(input.paste)] });

  // an open conversation carries on from the model's own session on logOS
  if (saved.chat && user) {
    const ok = await resumeChat(ctx, saved.chat).catch(() => false);
    if (!ok) { st.model = null; st.chat = null; }
  }

  // a question is answered by replaying what led to it, quietly, then the answer. Only questions reached without a
  // change on logOS are kept, so the replay changes nothing (see `answerable` below).
  const lineIn = String(input.line ?? '');
  let lines = [lineIn];
  // a new /command instead of an answer drops the question, as a click does in the terminal
  if (saved.pending?.length && !lineIn.trim().startsWith('/')) {
    quiet.on = true; net.writes = 0;
    for (const l of saved.pending) {
      if (st.ask === null && l !== saved.pending[0]) break;
      await execute(ctx, l, ui);
    }
    quiet.on = false;
    if (!st.ask || net.writes) {
      st.ask = null;
      block(() => [line(Am('that question has gone; run the command again.'))], null);
      lines = null;
    } else lines = [...saved.pending, lineIn];
  }
  net.writes = 0;
  const wasChat = !!(st.model && st.chat);
  if (input.paste && !wasChat) { block(() => [line(Dm('a document goes to the model you are talking to: '), K('/talk to <model>', 'talk to '), Dm(' first.'))], null); lines = null; }
  if (lines) await execute(ctx, lineIn, ui);
  // "confirm": true answers the terminal's own [y/n] confirmations (spec, decided Nov 25)
  for (let i = 0; i < 4 && input.confirm && st.ask && /\[y\/n\]/i.test(st.ask.ps); i += 1) { lines.push('y'); await execute(ctx, 'y', ui); }
  await Promise.allSettled(tasks);

  // words to a model: stay for the reply, as long as asked
  const wait = Math.max(0, Math.min(WAIT_MAX, Number(input.wait) || 0));
  if (wait && wasChat && !lineIn.trim().startsWith('/') && st.chat) {
    const until = Date.now() + wait * 1000;
    const settled = () => {
      const ch = st.chat; if (!ch?.s) return true;
      const fresh = (ch.s.turns || []).filter((t, i) => t.kind === 'user' && !ch.baseline?.has(t.id !== undefined && t.id !== null ? `${t.id}` : `${t.kind}:${t.at || i}`));
      return fresh.length > 0 && fresh.every((t) => TERMINAL.has(t.state)) && !ch.s.busy;
    };
    while (Date.now() < until && !settled()) await new Promise((z) => setTimeout(z, 750));
    if (!settled()) block(() => [line(Dm('still working; '), K('/history'), Dm(' or the live stream has the rest.'))], null);
  }

  // what this call drew goes to the account's stream, like any terminal's: logOS writes it (`stream` below)
  const drawn = blocks.map((b) => { let ls = []; try { ls = b.render(cols); } catch (e) { ls = [line(Am(`that page failed to draw: ${e.message}`))]; } return { b, ls }; });
  const kept = drawn.filter(({ b }) => !b.private).map(({ b, ls }) => ({ block_id: b.id, echo: b.echo, kind: b.kind, model: b.model, lines: segs(ls) }));

  const ask = st.ask;
  const answerable = !!(ask && lines && !ask.secret && net.writes === 0);
  const out = {
    blocks: drawn.map(({ b, ls }) => ({ id: b.id, echo: b.echo ? b.echo.text : null, lines: segs(ls), ...(b.private ? { private: true } : {}) })),
    text: drawn.map(({ b, ls }) => [b.echo ? `${b.echo.ps}${b.echo.text}` : null, plain(ls)].filter((x) => x).join('\n')).filter(Boolean).join('\n'),
    question: ask ? { prompt: ask.ps, choices: lastSugg, answerable, ...(ask.secret ? { secret: true } : {}) } : null,
    suggestions: ask ? [] : lastSugg,
    status: { model: st.model, line: status ? status.segs.map((g) => g.t).join('') : null, rows: rows ? rows.map((l) => (Array.isArray(l) ? l.map((g) => g.t).join('') : String(l))) : [] },
    data,
    links,
    stream: kept,
    state: {
      n,
      ...Object.fromEntries(KEEP.map((k) => [k, k === 'models' || k === 'devs' ? [...st[k].entries()].slice(-200) : st[k]])),
      chat: st.chat ? { mac: st.chat.mac, name: st.chat.name, model: st.model } : null,
      pending: answerable ? lines : null,
    },
  };
  return JSON.stringify(out);
}

// exit only once the answer is written: a pipe is asynchronous on some systems, and exiting early cuts it short
const say = (text, code) => process.stdout.write(text, () => process.exit(code));

let raw = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (c) => { raw += c; });
process.stdin.on('end', () => {
  let input; try { input = JSON.parse(raw); } catch (e) { say(JSON.stringify({ error: 'bad input' }), 2); return; }
  main(input).then((text) => say(text, 0)).catch((e) => say(JSON.stringify({ error: String(e?.message || e) }), 1));
});

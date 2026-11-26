#!/usr/bin/env node
// logos: the logOS terminal in your own (Nov 26, the open platform, Part 5). The language itself runs on logOS
// (POST /api/run), the same code the browser runs, so this is a window onto it and never a second copy.
//
//   logos                      the terminal, interactive
//   logos run "/models"        one line; --json for the whole answer, --confirm, --wait <s>, --session <id>
//   logos stream               follow your account's stream live: every terminal, program and agent
//                              --kind reply,read  --model soma-1b: only those
//   logos login                sign in through your browser (or: logos login --token lgp_...)
//   logos logout               forget the sign-in on this computer
//
// LOGOS_URL picks the site (https://logossoma.com); LOGOS_TOKEN overrides the saved sign-in.
import { readFileSync, writeFileSync, mkdirSync, existsSync, unlinkSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { join, basename } from 'node:path';
import { createInterface } from 'node:readline';

process.stdout.on('error', (e) => { if (e.code === 'EPIPE') process.exit(0); throw e; });   // a pipe closed early (| head)

const DIR = join(process.env.XDG_CONFIG_HOME || join(homedir(), '.config'), 'logos');
const CONFIG = join(DIR, 'config.json');
const SITE = (process.env.LOGOS_URL || 'https://logossoma.com').replace(/\/$/, '');
const PASTE_MAX = 4_000_000;

const saved = () => { try { return JSON.parse(readFileSync(CONFIG, 'utf8')); } catch (_) { return {}; } };
const token = () => process.env.LOGOS_TOKEN || saved().token || null;
const headers = () => ({ 'content-type': 'application/json', 'user-agent': 'logos-cli/0.1', ...(token() ? { authorization: `Bearer ${token()}` } : {}) });

// ---- drawing: the terminal's segments, in colour where the terminal allows it ----
const COLOUR = process.stdout.isTTY && !process.env.NO_COLOR;
const paint = (seg) => {
  const t = seg.t || '';
  if (!COLOUR || !t) return t;
  if (seg.cmd !== undefined) return `\x1b[1m${t}\x1b[22m`;
  if (seg.href) return `\x1b[4m${t}\x1b[24m`;
  if (seg.cls === 'hi') return `\x1b[1;97m${t}\x1b[0m`;
  if (seg.cls === 'dm') return `\x1b[2m${t}\x1b[22m`;
  if (seg.cls === 'am') return `\x1b[33m${t}\x1b[39m`;
  return t;
};
const draw = (blocks) => {
  for (const b of blocks || []) for (const ln of b.lines || []) process.stdout.write(`${ln.map(paint).join('')}\n`);
};

async function api(path, body, method = body === undefined ? 'GET' : 'POST') {
  const r = await fetch(`${SITE}/api${path}`, { method, headers: headers(), body: body === undefined ? undefined : JSON.stringify(body) });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(d.detail || `logOS answered ${r.status}`), { status: r.status });
  return d;
}

const cols = () => Math.max(40, Math.min(200, process.stdout.columns || 100));

// a local file given to /paste is read here and sent as a document; anything else /paste means goes to logOS
function localPaste(line) {
  const m = line.match(/^\/paste\s+(.+)$/i); if (!m) return null;
  const path = m[1].trim().replace(/^~(?=\/)/, homedir());
  if (!existsSync(path) || !statSync(path).isFile()) return null;
  const text = readFileSync(path, 'utf8');
  if (text.length > PASTE_MAX) throw new Error(`${basename(path)} is more than ${PASTE_MAX.toLocaleString('en')} characters: upload it with /upload in the browser`);
  return { text, name: basename(path) };
}

async function run(line, opts = {}) {
  const paste = localPaste(line);
  const body = { cols: cols(), session: opts.session, confirm: !!opts.confirm, wait: opts.wait ?? 0 };
  if (paste) Object.assign(body, { command: `[pasted document: ${paste.name}]`, paste: paste.text, wait: opts.wait ?? 60 });
  else if (opts.answer) body.answer = line;
  else body.command = line;
  return api('/run', body);
}

// ---- logos login: the device-code link the soma app uses, as a terminal ----
async function login(args) {
  const i = args.indexOf('--token');
  let tok = i >= 0 ? args[i + 1] : null;
  let who = null;
  if (!tok) {
    const d = await api('/auth/device', { client_name: 'logos (command line)', kind: 'shell' });
    process.stdout.write(`open ${d.verification_uri_complete}\nand approve the code ${d.user_code}\n`);
    const until = Date.now() + d.expires_in * 1000;
    while (Date.now() < until) {
      await new Promise((z) => setTimeout(z, (d.interval || 3) * 1000));
      const r = await fetch(`${SITE}/api/auth/device/token`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ device_code: d.device_code }) });
      const b = await r.json().catch(() => ({}));
      if (r.ok) { tok = b.access_token; who = b.username; break; }
      if (b.detail !== 'authorization_pending') throw new Error(b.detail || `logOS answered ${r.status}`);
    }
    if (!tok) throw new Error('the code expired: run logos login again');
  }
  mkdirSync(DIR, { recursive: true, mode: 0o700 });
  writeFileSync(CONFIG, JSON.stringify({ token: tok }), { mode: 0o600 });
  if (!who) { process.env.LOGOS_TOKEN = tok; who = (await api('/auth/me').catch(() => ({}))).username; }
  process.stdout.write(who ? `signed in as @${who}\n` : 'saved; logOS did not recognise that token\n');
}

// ---- logos stream: the live feed ----
async function stream(args) {
  const flag = (f) => { const i = args.indexOf(f); return i >= 0 ? args[i + 1] : ''; };
  const only = new URLSearchParams({ ...(flag('--kind') ? { kind: flag('--kind') } : {}), ...(flag('--model') ? { model: flag('--model') } : {}) }).toString();
  let { rev } = await api('/me/terminal?limit=1');
  process.stderr.write('following your stream (ctrl+c stops)\n');
  for (;;) {
    let d;
    try { d = await api(`/me/terminal?after=${rev}&wait=25${only ? `&${only}` : ''}`); } catch (e) { process.stderr.write(`${e.message}; again in 5s\n`); await new Promise((z) => setTimeout(z, 5000)); continue; }
    for (const b of d.blocks) {
      // the stored prompt is Φ, in the site's own face; a terminal we can't set the font of shows > (Nov 26, James)
      if (b.echo?.text) process.stdout.write(`${paint({ t: `${(b.echo.ps || '').replace('Φ', '>')}${b.echo.text}`, cls: 'dm' })}\n`);
      draw([b]);
    }
    rev = d.rev;
  }
}

// ---- logos: the terminal ----
async function terminal() {
  let session = null; let question = null; let model = null;
  const rl = createInterface({ input: process.stdin, output: process.stdout, terminal: !!process.stdin.isTTY, historySize: 500 });
  const prompt = () => { rl.setPrompt(question ? question.prompt : model ? `${model.slice(0, 24)}> ` : '> '); rl.prompt(); };
  rl.on('close', () => { process.stdout.write('\n'); process.exit(0); });
  if (!token()) process.stdout.write(paint({ t: 'not signed in: logos login. public models and /help work without.\n', cls: 'dm' }));
  prompt();
  for await (const line of rl) {
    const text = line.trim();
    if (!text && !question) { prompt(); continue; }
    rl.pause();
    try {
      const chatting = model && !text.startsWith('/') && !question;
      if (chatting) process.stdout.write(paint({ t: '...\n', cls: 'dm' }));
      const out = await run(text, { session, answer: !!question && !text.startsWith('/'), wait: chatting ? 120 : 0 });
      session = out.session;
      draw(out.blocks);                      // what you typed is already on screen: only what came back
      for (const l of out.links || []) process.stdout.write(`${paint({ t: l, href: l })}\n`);
      question = out.question?.answerable ? out.question : null;
      if (out.question && !out.question.answerable) process.stdout.write(paint({ t: `${out.question.prompt.trim()} (answer it in the browser terminal, or give the whole command)\n`, cls: 'am' }));
      model = out.status?.model || null;
      if (out.status?.line) process.stdout.write(`${paint({ t: out.status.line, cls: 'dm' })}\n`);
    } catch (e) {
      process.stdout.write(`${paint({ t: e.message, cls: 'am' })}\n`);
    }
    rl.resume();
    prompt();
  }
}

async function main() {
  const [cmd, ...args] = process.argv.slice(2);
  if (!cmd) return terminal();
  if (cmd === 'login') return login(args);
  if (cmd === 'logout') { try { unlinkSync(CONFIG); } catch (_) { /* */ } process.stdout.write('signed out here. the sign-in stays listed in /settings until you end it there.\n'); return; }
  if (cmd === 'stream') return stream(args);
  if (cmd === 'run') {
    const flag = (f) => { const i = args.indexOf(f); if (i < 0) return undefined; const v = args[i + 1]; args.splice(i, 2); return v; };
    const json = args.includes('--json'); const confirm = args.includes('--confirm');
    for (const f of ['--json', '--confirm']) { const i = args.indexOf(f); if (i >= 0) args.splice(i, 1); }
    const wait = Number(flag('--wait') || 0); const session = flag('--session');
    const out = await run(args.join(' '), { wait, session, confirm });
    if (json) process.stdout.write(`${JSON.stringify(out, null, 2)}\n`);
    else {
      draw(out.blocks);
      if (out.question) process.stdout.write(`${paint({ t: `${out.question.prompt.trim()}  (answer: logos run --session ${out.session} "<answer>")`, cls: 'am' })}\n`);
    }
    return;
  }
  if (cmd === '--version' || cmd === '-v') { process.stdout.write('logos 0.1.0\n'); return; }
  process.stdout.write('logos            the terminal\nlogos run "/models" [--json] [--confirm] [--wait s] [--session id]\nlogos stream     follow your stream [--kind reply,read] [--model name]\nlogos login      sign in (--token lgp_... for a key)\nlogos logout\n');
}

main().catch((e) => { process.stderr.write(`${e.message}\n`); process.exit(1); });

// logOS is one terminal (Nov 19, James): a page is printed ASCII, a bold byte is a link, and a link is a command: a
// click types it into the prompt and runs it, so clicking and typing are one act and every click teaches its command.
// Every place is an address: the address bar follows the terminal, and opening an address prints its page.
import { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { line, Dm, Am, S, K, ascii, wrap } from './text';
import { startRain } from './rain';
import { pathToCommand, commandToPath, resumeChat, COMPLETE, keepStream, chooseName } from './commands';
import { execute } from './shell';
import './terminal.css';

const PRINT_MS = 250;
const SHELL = 'logos:term:shell';            // this tab's shell: sessionStorage belongs to the tab alone

export default function Terminal() {
  const auth = useAuth();
  const navigate = useNavigate();
  const authRef = useRef(auth); authRef.current = auth;
  const navRef = useRef(navigate); navRef.current = navigate;
  const root = useRef(null);
  const engine = useRef(null);

  useEffect(() => {
    const el = root.current;
    const scroll = el.querySelector('.tm-scroll'); const input = el.querySelector('.tm-input');
    const psEl = el.querySelector('.tm-ps'); const typedEl = el.querySelector('.tm-typed'); const bkEl = el.querySelector('.tm-bk');
    const afterEl = el.querySelector('.tm-after'); const suggEl = el.querySelector('.tm-sugg');
    const statusEl = el.querySelector('.tm-status'); const strow = el.querySelector('.tm-strow'); let status = null; let statusTimer = null;
    // the suggestions and the status line scroll sideways; '...' at either end says there's more that way, and the
    // text fades into it, as the scrollback fades at the top and bottom
    const edges = (box, [l, r]) => () => {
      const room = box.scrollWidth - box.clientWidth;
      l.hidden = box.scrollLeft < 2; r.hidden = room - box.scrollLeft < 2;
      box.classList.toggle('tm-fade-l', !l.hidden); box.classList.toggle('tm-fade-r', !r.hidden);
    };
    const overflow = edges(suggEl, el.querySelector('.tm-srow').querySelectorAll('.tm-more'));
    const overflowStatus = edges(statusEl, strow.querySelectorAll('.tm-more')); const meas = el.querySelector('.tm-measure');
    const pline = el.querySelector('.tm-pline');
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const fine = window.matchMedia('(pointer: fine)').matches;
    const TAB = Math.random().toString(36).slice(2, 8); let nblk = 0;     // this terminal, and its blocks' ids
    const BLOCKS_SHOWN = 400; const STORE_COLS = 100; const PASTE_AS_DOCUMENT = 2000;
    let cols = 80; let printing = null; let busy = false; let alive = true;
    const blocks = []; const hist = []; let hi = 0;
    const st = { list: [], here: null, model: null, ask: null, chat: null, apps: [], models: new Map(), devs: new Map() };

    const measure = () => {
      const chW = meas.offsetWidth / meas.textContent.length || 8.4;      // layout pixels, as clientWidth: a transform can't skew it
      cols = Math.max(28, Math.min(100, Math.floor((scroll.clientWidth - 10) / chW)));
    };

    // ---- drawing ----
    const host = (u) => { try { return new URL(u).hostname; } catch (_) { return ''; } };
    const mediaEl = (m) => {
      // an image loads straight from its site, without telling it which page asked (no referrer); a video plays from
      // its source, youtube through its no-cookie embed
      if (m.kind === 'image') {
        const a = document.createElement('a'); a.href = m.src; a.target = '_blank'; a.rel = 'noopener noreferrer'; a.className = 'tm-media';
        const img = document.createElement('img'); img.src = m.src; img.alt = ascii(m.alt); img.loading = 'lazy'; img.referrerPolicy = 'no-referrer'; img.className = 'tm-img';
        img.onerror = () => { a.textContent = ascii(`[image: ${m.alt || host(m.src)}]`); a.className = 'tm-lk'; };
        a.appendChild(img); return a;
      }
      const yt = m.src.match(/(?:youtube(?:-nocookie)?\.com\/(?:embed\/|watch\?v=)|youtu\.be\/)([\w-]{6,})/);
      const vm = m.src.match(/player\.vimeo\.com\/video\/(\d+)|vimeo\.com\/(\d+)/);
      if (yt || vm) {
        const f = document.createElement('iframe'); f.className = 'tm-vid'; f.allowFullscreen = true; f.referrerPolicy = 'no-referrer';
        f.src = yt ? `https://www.youtube-nocookie.com/embed/${yt[1]}` : `https://player.vimeo.com/video/${vm[1] || vm[2]}?dnt=1`;
        f.allow = 'encrypted-media; picture-in-picture; fullscreen'; return f;
      }
      const v = document.createElement('video'); v.className = 'tm-vid'; v.controls = true; v.preload = 'metadata'; v.src = m.src; return v;
    };
    const segEl = (g) => {
      if (g.media && /^https?:\/\//.test(g.media.src)) {
        const m = g.media;
        if (m.show) return mediaEl(m);
        const a = document.createElement('a'); a.href = '#'; a.className = 'tm-lk';
        a.textContent = ascii(`[ ${m.kind === 'video' ? 'play video' : 'show image'}: ${m.alt || host(m.src)} ]`) + ' ';
        const h = document.createElement('span'); h.className = 'tm-dm'; h.textContent = ascii(host(m.src));
        const wrapEl = document.createElement('span'); wrapEl.appendChild(a); wrapEl.appendChild(h);
        a.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); wrapEl.replaceWith(mediaEl(m)); });
        return wrapEl;
      }
      if (g.cmd !== undefined || g.href || g.pick) {
        const a = document.createElement('a'); a.className = 'tm-lk' + (g.cls ? ` tm-${g.cls}` : ''); a.textContent = ascii(g.t);   // strict ascii, whatever logOS sent
        if (g.pick) { a.href = '#'; a.dataset.pick = '1'; }
        else if (g.href) { a.href = g.href; if (/^https?:/.test(g.href)) { a.target = '_blank'; a.rel = 'noopener'; } }
        else { a.dataset.cmd = g.cmd; a.href = commandToPath(g.cmd) || '#'; }   // its address: cmd-click forks a shell there
        return a;
      }
      if (!g.cls) return document.createTextNode(ascii(g.t));
      const s = document.createElement('span'); s.className = `tm-${g.cls}`; s.textContent = ascii(g.t); return s;
    };
    // the status line (Nov 23, James): what a model is doing now, how long it has been at it, and that it's alive
    const SPIN = ['|', '/', '-', '\\'];
    let spinEl = null; let clockEl = null;
    const tick = () => {
      if (!status) return;
      if (spinEl) spinEl.textContent = `${SPIN[reduce ? 0 : Math.floor(Date.now() / 200) % 4]} `;
      if (clockEl) {
        const secs = Math.max(0, Math.floor((Date.now() - status.since) / 1000));
        clockEl.textContent = `   ${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, '0')}`;
      }
    };
    const drawStatus = () => {
      statusEl.textContent = ''; spinEl = clockEl = null;
      if (!status) { strow.hidden = true; return; }
      strow.hidden = false;
      if (status.spin) { spinEl = segEl(S(' ', { cls: 'am' })); statusEl.appendChild(spinEl); }
      for (const g of status.segs) statusEl.appendChild(segEl(g));
      if (!status.notimer) { clockEl = segEl(S(' ', { cls: 'dm' })); statusEl.appendChild(clockEl); }
      tick(); overflowStatus(); requestAnimationFrame(overflowStatus);
    };
    const setStatus = (s) => {
      const same = status && s && JSON.stringify(s.segs) === JSON.stringify(status.segs) && !!s.spin === !!status.spin && !!s.notimer === !!status.notimer;
      status = s ? { ...s, since: s.since || Date.now() } : null;
      if (!same) drawStatus();
      if (status && !statusTimer) statusTimer = setInterval(tick, 200);
      if (!status && statusTimer) { clearInterval(statusTimer); statusTimer = null; }
    };
    // one row per read of yours that is running (Nov 25, James): pause, skip or look without scrolling back up
    const rowsEl = el.querySelector('.tm-rows'); let rowsSig = '';
    const setRows = (rows) => {
      const sig = JSON.stringify(rows || []);
      if (sig === rowsSig) return; rowsSig = sig;
      rowsEl.textContent = ''; rowsEl.hidden = !(rows && rows.length);
      for (const segs of rows || []) { const d = document.createElement('div'); d.className = 'tm-rline'; for (const g of segs) d.appendChild(segEl(g)); rowsEl.appendChild(d); }
    };
    const lineEl = (segs) => { const d = document.createElement('div'); d.className = 'tm-ln'; for (const g of segs) d.appendChild(segEl(g)); if (!segs.length) d.textContent = ' '; return d; };
    const atBottom = () => scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 40;
    const draw = (b, animate) => {
      const lines = [];
      if (b.echo) lines.push([S(b.echo.ps, { cls: 'dm' }), S(b.echo.text, { cls: 'echo' })]);
      try {
        for (const l of b.render(cols)) {
          // a line wider than the screen wraps rather than run off the glass (phones)
          const len = l.reduce((a, g) => a + g.t.length, 0);
          if (len > cols) lines.push(...wrap(l, cols)); else lines.push(l);
        }
      } catch (e) { lines.push(line(Am(`that page failed to print: ${ascii(e.message)}`))); }
      const stick = atBottom();
      b.el.textContent = '';
      if (!animate || reduce || document.hidden) { for (const l of lines) b.el.appendChild(lineEl(l)); if (stick || animate) scroll.scrollTop = scroll.scrollHeight; return Promise.resolve(b); }
      // a quarter of a second for a page, line by line, on timers (a hidden tab gets no animation frames); any key or
      // click finishes it, and a print only ever clears its own flag
      return new Promise((done) => {
        const t0 = performance.now(); let n = 0;
        const skip = () => { for (const l of lines.slice(n)) b.el.appendChild(lineEl(l)); n = lines.length; };
        printing = skip; el.classList.add('tm-printing');
        const step = () => {
          const want = Math.min(lines.length, Math.ceil((lines.length * (performance.now() - t0)) / PRINT_MS));
          while (n < want) b.el.appendChild(lineEl(lines[n++]));
          scroll.scrollTop = scroll.scrollHeight;
          if (n < lines.length) setTimeout(step, 16);
          else { if (printing === skip) { printing = null; el.classList.remove('tm-printing'); } done(b); }
        };
        setTimeout(step, 0);
      });
    };
    const newBlock = (render, echo, opts = {}) => {
      const div = document.createElement('div'); div.className = 'tm-blk'; scroll.appendChild(div);
      const b = { el: div, render, echo, id: opts.id || `${TAB}-${(nblk += 1)}`, nostore: !!opts.nostore, kind: opts.kind || (echo ? 'command' : 'note'), model: opts.model || st.model || null }; blocks.push(b);
      while (blocks.length > BLOCKS_SHOWN) blocks.shift().el.remove();
      return b;
    };
    // the stream (Nov 25, James): each block this terminal draws is sent to logOS, a few seconds after it changes
    const segsOf = (ls) => ls.map((l) => l.map((g) => { const o = { t: g.t }; if (g.cls) o.cls = g.cls; if (g.cmd !== undefined) o.cmd = g.cmd; if (g.href) o.href = g.href; if (g.media) o.media = g.media; return o; }));
    const dirty = new Set(); let flushTimer = null;
    const flush = () => {
      flushTimer = null; if (!ctx.user()) { dirty.clear(); return; }
      const out = [];
      for (const b of dirty) { try { out.push({ block_id: b.id, echo: b.echo || null, kind: b.kind, model: b.model, lines: segsOf(b.render(STORE_COLS)) }); } catch (_) { /* */ } }
      dirty.clear(); if (out.length) keepStream(out);
    };
    const mark = (b) => { if (b.nostore) return; dirty.add(b); if (!flushTimer) flushTimer = setTimeout(flush, 3000); };
    const print = (render, echo) => { const b = newBlock(render, echo); mark(b); return draw(b, true); };
    // a secret shown once (a new token): never sent to the stream, never kept for a reload
    const printPrivate = (render) => { const b = newBlock(render, null, { nostore: true }); b.private = true; return draw(b, true); };
    const printLive = (render, meta = {}) => { const b = newBlock(render, null, meta); mark(b); draw(b, false); scroll.scrollTop = scroll.scrollHeight; setTimeout(() => { scroll.scrollTop = scroll.scrollHeight; }, 30); return b; };
    const update = (b, render) => { b.render = render; mark(b); draw(b, false); keep(); };
    // /history: the account's stream, filled in above what this terminal already shows
    const backfill = (stored) => {
      const have = new Set(blocks.map((b) => b.id)); const top = blocks[0]?.el || null; const added = [];
      for (const x of stored) {
        if (have.has(x.block_id)) continue;
        const div = document.createElement('div'); div.className = 'tm-blk'; scroll.insertBefore(div, top);
        const b = { el: div, render: () => x.lines, echo: x.echo, id: x.block_id, nostore: true }; added.push(b); draw(b, false);
      }
      blocks.unshift(...added);
      while (blocks.length > BLOCKS_SHOWN) blocks.shift().el.remove();
      scroll.scrollTop = scroll.scrollHeight; return added.length;
    };
    const redrawAll = () => { for (const b of blocks) draw(b, false); scroll.scrollTop = scroll.scrollHeight; };

    // ---- the prompt ----
    // the prompt bar keeps > (Nov 26, James: "slightly more intuitive and familiar"); the mark is Φ elsewhere
    const ps = () => (st.ask ? st.ask.ps : st.model ? `${ascii(st.model).slice(0, 24)}> ` : '> ');
    const syncPrompt = () => {
      // at rest the prompt reads >_ ; the space comes back as you type
      const p = ps(); psEl.textContent = !input.value && !st.ask && p.endsWith('> ') ? p.slice(0, -1) : p;
      const v = st.ask?.secret ? '*'.repeat(input.value.length) : input.value; const c = input.selectionStart ?? v.length;
      const ch = v.slice(c, c + 1);
      typedEl.textContent = v.slice(0, c);
      // the block sits on the character after the caret; at a line break or the end it's a space before it
      if (!ch || ch === '\n') { bkEl.textContent = ' '; afterEl.textContent = v.slice(c); }
      else { bkEl.textContent = ch; afterEl.textContent = v.slice(c + 1); }
      // a long command grows the prompt up to half the screen, then scrolls with the caret in view
      if (c >= v.length - 1) pline.scrollTop = pline.scrollHeight;
    };
    const pasteKey = () => {                       // touch screens only: a link that pastes at the prompt
      const a = document.createElement('a'); a.href = '#'; a.className = 'tm-lk tm-paste'; a.dataset.paste = '1'; a.textContent = 'paste';
      return a;
    };
    const suggest = (items) => {
      lastSugg = items || [];
      suggEl.textContent = '';
      if (!fine && navigator.clipboard?.readText) suggEl.appendChild(pasteKey());     // first: the row scrolls sideways
      if (items && items.length) {
                // a question's answers are words; anything else is a command, and shows its slash
        items.forEach((cmd, i) => { suggEl.appendChild(segEl(st.ask ? S(cmd, { cmd }) : cmd.startsWith('/') ? K(cmd) : K(`/${cmd}`, cmd))); if (i < items.length - 1) suggEl.appendChild(document.createTextNode('   ')); });
      }
      suggEl.scrollLeft = 0; overflow(); requestAnimationFrame(overflow);
    };
    // a long paste (Nov 25, James) shows in the prompt as [pasted document n: x chars]; sent to a model, it goes to
    // the model's files and is read into this chat's store before your words
    const pastes = new Map(); let pasteN = 0;
    const insertText = (text) => {
      if (text.length >= PASTE_AS_DOCUMENT || text.split('\n').length > 40) {
        pasteN += 1; pastes.set(pasteN, text);
        text = `[pasted document ${pasteN}: ${text.length.toLocaleString('en')} chars]`;
      }
      const v = input.value; const a = input.selectionStart ?? v.length; const b = input.selectionEnd ?? a;
      input.value = v.slice(0, a) + text + v.slice(b);
      input.focus({ preventScroll: true }); input.setSelectionRange(a + text.length, a + text.length);
      syncPrompt();
    };
    const onPaste = (e) => {
      if (st.ask?.secret) return;
      const text = e.clipboardData?.getData('text') || '';
      if (text.length < PASTE_AS_DOCUMENT && text.split('\n').length <= 40) return;      // short: pasted as it is
      e.preventDefault(); insertText(text);
    };
    const pasteHere = async () => {
      let text = '';
      try { text = await navigator.clipboard.readText(); } catch (_) { return; }       // declined, or nothing to paste
      if (text) insertText(text);
    };
    const setPath = (path) => { if (path && path !== window.location.pathname + window.location.search) window.history.pushState({ tm: true }, '', path); };
    let place = 'home';
    const title = () => { document.title = st.model ? `logOS: talk to ${ascii(st.model)}` : place === 'home' ? 'logOS' : `logOS: ${place}`; };

    const ctx = {
      st,
      user: () => authRef.current.user,
      login: (opts) => authRef.current.login(opts),
      run: (cmd) => typeAndRun(cmd),
      pick: (handler, accept = '') => { st.pick = handler; picker.accept = accept; picker.click(); },
      logout: () => authRef.current.logout(),
      navigate: (p) => navRef.current(p),
      go: (url) => { window.location.href = url; },
      print, printLive, printPrivate, update, suggest, setStatus, setRows, backfill,
      ask: (q, answer, sugg, options = {}) => { st.ask = { ps: q, answer, secret: !!options.secret }; syncPrompt(); suggest(sugg || []); },
      active: () => alive,
      setModel: (name) => { st.model = name || null; syncPrompt(); title(); keep(); if (name) suggest(['/files', '/save', '/exit']); else { setStatus(null); setRows(null); } },
      remember: (list) => { for (const m of list || []) if (m?.checkpoint_id) st.models.set(m.checkpoint_id, { checkpoint_id: m.checkpoint_id, name: m.name }); },
      known: () => [...st.models.values()],
      rememberDevs: (list) => { for (const d of list || []) if (d?.slug) st.devs.set(d.slug, { slug: d.slug, title: d.title }); },
      knownDevs: () => [...st.devs.values()],
    };

    let lastSugg = [];
    let keepTimer = null;
    const keep = () => {
      clearTimeout(keepTimer);
      keepTimer = setTimeout(() => {
        try {
          let budget = 4000; const saved = [];
          for (const b of blocks.slice(-40).reverse()) {
            if (b.private) continue;
            let ls; try { ls = b.render(cols); } catch (_) { continue; }
            if (ls.length > budget) break; budget -= ls.length;
            saved.unshift({ e: b.echo, l: ls.map((l) => l.map((g) => { const o = { t: g.t }; if (g.cls) o.cls = g.cls; if (g.cmd !== undefined) o.cmd = g.cmd; if (g.href) o.href = g.href; if (g.media) o.media = g.media; return o; })) });
          }
          const at = window.location.pathname + window.location.search;
          const chat = st.model && st.chat ? { mac: st.chat.mac, name: st.chat.name, model: st.model } : null;
          sessionStorage.setItem(SHELL, JSON.stringify({ v: 1, at, place, hist: hist.slice(-200), sugg: lastSugg, blocks: saved, chat }));
        } catch (_) { /* storage full or blocked: the shell just doesn't survive a reload */ }
      }, 400);
    };
    const restore = () => {
      let s = null; try { s = JSON.parse(sessionStorage.getItem(SHELL) || 'null'); } catch (_) { /* */ }
      if (!s || s.v !== 1) return null;
      for (const b of s.blocks || []) { const nb = newBlock(() => b.l, b.e, { nostore: true }); draw(nb, false); }
      hist.push(...(s.hist || [])); hi = hist.length; place = s.place || 'home';
      suggest(s.sugg || []); scroll.scrollTop = scroll.scrollHeight; title();
      return s;
    };

    // a new entry snaps both to the bottom: the scrollback, and the prompt (which shrinks back as it clears)
    const snap = () => {
      const go = () => { scroll.scrollTop = scroll.scrollHeight; pline.scrollTop = pline.scrollHeight; };
      go(); requestAnimationFrame(go); setTimeout(go, 30);
    };

    // ---- running a line ----
    const exec = async (raw, { echo = true, push = true } = {}) => {
      if (printing) printing();
      snap();
      await execute(ctx, raw, {
        echo, ps,
        typed: (text) => { hist.push(text); hi = hist.length; },
        asked: () => { syncPrompt(); suggest([]); },
        busy: (on) => { busy = on; el.classList.toggle('tm-busy', on); },
        pasted: (text) => {
          const docs = []; const words = text.replace(/\[pasted document (\d+): [^\]]*\]/g, (all, n) => { if (pastes.has(+n)) { docs.push(pastes.get(+n)); return ''; } return all; }).trim();
          return { words, docs };
        },
        result: (r, cmd) => {
          if (push && r.path) setPath(r.path);
          if (r.path) { place = st.here?.kind === 'model' && /^open /i.test(cmd) ? `open ${ascii(st.here.name || '')}` : st.here?.kind === 'dev' && /^(open|developments|reading) /i.test(cmd) ? `developments ${st.here.id}` : cmd.slice(0, 48); title(); }
        },
        clear: () => { for (const b of blocks.splice(0)) b.el.remove(); suggest([]); },
        back: () => window.history.back(),
      });
      keep();
    };
    const typeAndRun = (cmd) => {
      if (busy) return;
      if (st.ask && !/^(yes|no|y|n)$/i.test(cmd) && !lastSugg.includes(cmd)) { st.ask = null; input.value=''; }
      if (!st.ask && !cmd.startsWith('/')) cmd = '/' + cmd;          // a click is a command: it types its slash
      if (printing) printing();
      // a command that wants more ("open ", "search ") is typed and left for you to finish, like a shell
      if (/\s$/.test(cmd)) { input.value = cmd; syncPrompt(); input.focus(); input.setSelectionRange(cmd.length, cmd.length); syncPrompt(); return; }
      if (/^(yes|no)$/.test(cmd) && st.ask) cmd = cmd === 'yes' ? 'y' : 'n';
      input.value = ''; let i = 0;
      const per = Math.max(2, Math.min(9, 140 / Math.max(1, cmd.length)));
      const step = () => {
        i=reduce?cmd.length:i+1;
        input.value = cmd.slice(0, i); syncPrompt();
        if (i < cmd.length && !reduce) setTimeout(step, per);
        else { const v = input.value; input.value = ''; syncPrompt(); exec(v); }
      };
      step();
    };

    // ---- keys and clicks ----
    const onInput = () => syncPrompt();
    const onKey = (e) => {
      if (printing && e.key.length === 1) printing();
      const multi = input.value.includes('\n');
      if (st.ask?.secret && ['ArrowUp', 'ArrowDown', 'Tab'].includes(e.key)) { e.preventDefault(); return; }
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (busy) return; const v = input.value; input.value = ''; syncPrompt(); exec(v); }
      else if (e.key === 'Enter') { /* shift+enter: a line break, for longer messages */ }
      else if (e.key === 'ArrowUp' && !(multi && input.selectionStart > input.value.indexOf('\n'))) { e.preventDefault(); if (hi > 0) { input.value = hist[--hi]; syncPrompt(); } }
      else if (e.key === 'ArrowDown' && !(multi && input.selectionStart <= input.value.lastIndexOf('\n'))) { e.preventDefault(); if (hi < hist.length - 1) input.value = hist[++hi]; else { hi = hist.length; input.value = ''; } syncPrompt(); }
      else if (e.key === 'Tab') {
        e.preventDefault(); const v = input.value.toLowerCase(); if (!v || (!v.startsWith('/') && !st.ask)) return;
        const pool = [...COMPLETE, ...ctx.known().flatMap((m) => [`open ${m.name}`, `talk to ${m.name}`]), ...ctx.knownDevs().map((d) => `developments ${d.slug}`)]
          .map((c) => (c.startsWith('/') ? c : `/${c}`));
        const opts = [...new Set(pool.filter((c) => c.toLowerCase().startsWith(v)))];
        if (opts.length === 1) input.value = opts[0];
        else if (opts.length > 1) {
          let p = opts[0]; for (const o of opts) while (!o.toLowerCase().startsWith(p.toLowerCase())) p = p.slice(0, -1);
          if (p.length > v.length) input.value = p; else print((c) => wrap(opts.slice(0, 16).flatMap((o) => [K(o.trim()), S('   ')]), c), null);
        }
        syncPrompt();
      } else if (e.key === 'Escape') {
        if (st.ask) { st.ask = null; input.value = ''; syncPrompt(); suggest(['help']); }
        else if (st.model) exec('/exit');
      } else if (e.key === 'l' && e.ctrlKey) { e.preventDefault(); exec('/clear', { echo: false }); }
      requestAnimationFrame(syncPrompt);
    };
    const onClick = (e) => {
      if (e.target.closest('a[data-paste]')) { e.preventDefault(); pasteHere(); return; }
      if (e.target.closest('a[data-pick]')) { e.preventDefault(); picker.accept = st.pickAccept ?? '.pt,.json'; picker.click(); return; }   // the click is the gesture a file picker needs
      const a = e.target.closest('a.tm-lk');
      if (a && a.dataset.cmd !== undefined) {
        if ((e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1) && a.getAttribute('href') !== '#') return;   // a new tab, a new shell
        e.preventDefault(); typeAndRun(a.dataset.cmd); if (fine) input.focus(); return;
      }
      if (!a && !window.getSelection().toString()) input.focus({ preventScroll: true });
    };
    const onDocKey = (e) => { if (printing && e.target !== input) printing(); };
    const onPop = () => { exec(`/${pathToCommand(window.location.pathname, window.location.search)}`, { push: false }); };
    // one hidden file input; a command that wants files (upload) leaves st.pick to receive them
    const picker = document.createElement('input'); picker.type = 'file'; picker.multiple = true; picker.accept = '.pt,.json'; picker.hidden = true;
    picker.addEventListener('change', () => { const fs = [...picker.files]; picker.value = ''; if (fs.length && st.pick) Promise.resolve(st.pick(fs)).catch((err) => print(() => [line(Am(ascii(err.message || 'those files could not be read')))])); });
    el.appendChild(picker);
    for (const ev of ['input', 'keyup', 'click', 'select', 'focus']) input.addEventListener(ev, onInput);
    input.addEventListener('keydown', onKey); input.addEventListener('paste', onPaste);
    el.addEventListener('click', onClick);
    // in the soma app, a file dropped on the glass goes into soma's data folder on this computer; /read takes it from there
    const onDragOver = (e) => { if (window.soma?.addFiles && e.dataTransfer?.types?.includes('Files')) e.preventDefault(); };
    const onDrop = async (e) => {
      if (!window.soma?.addFiles || !e.dataTransfer?.files?.length) return;
      e.preventDefault();
      try {
        const added = await window.soma.addFiles(e.dataTransfer.files);
        if (!added.length) { await print(() => [line(Dm('only files go into the data folder.'))]); return; }
        await print(() => added.map((f) => line(Dm('added '), S(ascii(f.name)), Dm(' to the data folder  '), K(`/read ${f.name}`))));
        suggest(added.map((f) => `/read ${f.name}`));
      } catch (err) { await print(() => [line(Am(ascii(err.message || 'the file could not be added')))]); }
    };
    el.addEventListener('dragover', onDragOver); el.addEventListener('drop', onDrop);
    // the engine on this computer says what it's doing (setup, failures) in every shell
    const engineNote = (n) => n && print(() => [line(n.kind === 'error' ? Am(ascii(n.text)) : Dm(ascii(n.text)))]);
    window.soma?.engineNote?.().then(engineNote).catch(() => {});
    const offEngine = window.soma?.onEngine?.(engineNote);
    suggEl.addEventListener('scroll', overflow, { passive: true });
    statusEl.addEventListener('scroll', overflowStatus, { passive: true });
    document.addEventListener('keydown', onDocKey);
    window.addEventListener('popstate', onPop);
    measure(); syncPrompt();
    const ro = new ResizeObserver(() => { const c = cols; measure(); if (c !== cols) redrawAll(); overflow(); overflowStatus(); });
    ro.observe(scroll);
    // the spare width: the terminal's column sits in the middle of the glass, and past a margin on either side the
    // runtime's source rolls (rain.js). layout pixels throughout: the monitor may scale the screen with a transform
    const rainEls = [...el.querySelectorAll('.tm-rain')];
    const calm = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    const still = () => calm || !el.closest('.mon')?.classList.contains('mon-motion');
    const rains = rainEls.map((c) => startRain(c, { still }));
    let rainBox = '';
    const placeRain = () => {
      const cs = getComputedStyle(el);
      const chW = meas.offsetWidth / meas.textContent.length || 8.4;
      const padL = parseFloat(cs.paddingLeft) || 28; const padR = parseFloat(cs.paddingRight) || 28;
      const gap = 4 * chW;
      const width = Math.floor(Math.min(scroll.offsetLeft - padL, el.clientWidth - padR - scroll.offsetLeft - scroll.offsetWidth) - gap);
      const top = (parseFloat(cs.paddingTop) || 4) + 28; const height = Math.floor(el.clientHeight - top - (parseFloat(cs.paddingBottom) || 14));
      if (width < 12 * chW || height < 100) { for (const c of rainEls) c.hidden = true; rainBox = ''; return; }
      const box = `${width},${height},${el.clientWidth}`;
      for (const c of rainEls) c.hidden = false;
      if (box === rainBox) return;
      rainBox = box;
      const font = `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`; const lh = parseFloat(cs.lineHeight) || 21;
      rainEls[0].style.left = `${padL}px`; rainEls[1].style.left = `${el.clientWidth - padR - width}px`;
      rains.forEach((rn, i) => { rainEls[i].style.top = `${top}px`; rn.resize(width, height, font, chW, lh); if (still()) rn.settle(); });
    };
    const rro = new ResizeObserver(placeRain); rro.observe(el); rro.observe(scroll);
    document.fonts?.ready.then(placeRain).catch(() => {}); const rainLate = setTimeout(placeRain, 900);   // after the font and the power-on settle
    engine.current = { exec, restore, ctx, keep, started: false };
    window.addEventListener('pagehide', keep);
    return () => {
      alive = false;
      ro.disconnect(); rro.disconnect(); offEngine?.(); el.removeEventListener('dragover', onDragOver); el.removeEventListener('drop', onDrop); clearTimeout(rainLate); clearInterval(statusTimer); rains.forEach((rn) => rn.stop()); el.removeEventListener('click', onClick); suggEl.removeEventListener('scroll', overflow); statusEl.removeEventListener('scroll', overflowStatus); document.removeEventListener('keydown', onDocKey);
      window.removeEventListener('popstate', onPop); input.removeEventListener('keydown', onKey); input.removeEventListener('paste', onPaste);
      window.removeEventListener('pagehide', keep);
      if (st.chat) st.chat.stopped = true;
    };
  }, []);

  // the first page waits for sign-in to settle, so "chat" or "whoami" print for who you are
  useEffect(() => {
    // the browser's sign-in changed under this tab (AuthContext 'logos-account'): its conversation was the old account's
    const onAccount = (ev) => {
      const e = engine.current;
      if (!e?.ctx) return;
      const { st } = e.ctx;
      if (st.chat) { st.chat.stopped = true; st.chat = null; e.ctx.setModel(null); }
      const to = ev.detail?.to;
      e.ctx.print(() => [line(Am(to ? `this browser is signed in as @${ascii(to)} now` : 'this browser is signed out now'))]);
    };
    window.addEventListener('logos-account', onAccount);
    return () => window.removeEventListener('logos-account', onAccount);
  }, []);

  useEffect(() => {
    const e = engine.current;
    if (!e || e.started || auth.loading) return;
    e.started = true;
    // the start is blank (Nov 25, James): nothing explained, nothing shown, until you type. an address still opens its
    // page, and a start you chose (set start) still runs
    let first = pathToCommand(window.location.pathname, window.location.search);
    if (first === 'home') first = auth.user?.prefs?.start || '';
    const saved = e.restore();
    const here = window.location.pathname + window.location.search;
    const slash = (c) => (c.startsWith('/') ? c : `/${c}`);
    const opening = (saved && saved.at === here) || !first ? Promise.resolve() : e.exec(slash(first), { push: false });
    if (saved?.chat && auth.user) resumeChat(e.ctx, saved.chat).catch(() => {});
    opening.then(() => {
      let then = null; try { then = sessionStorage.getItem('logos:term:then'); sessionStorage.removeItem('logos:term:then'); } catch (_) { /* */ }
      if (then && auth.user) e.exec(slash(then));
      if (auth.user && auth.user.username_chosen === false) chooseName(e.ctx);     // a name of your own, first (Nov 25)
    });
    if (window.matchMedia('(pointer: fine)').matches) setTimeout(() => root.current?.querySelector('.tm-input')?.focus({ preventScroll: true }), 500);
  }, [auth.loading, auth.user]);

  return (
    <div className="tm" ref={root} data-testid="terminal">
      <canvas className="tm-rain" aria-hidden="true" hidden />
      <canvas className="tm-rain" aria-hidden="true" hidden />
      <div className="tm-scroll" data-curve role="log" aria-live="polite" aria-label="logOS" />
      <div className="tm-rows" data-curve="x" aria-live="polite" hidden />
      <div className="tm-strow" hidden>
        <span className="tm-more" data-curve="x" aria-hidden="true" hidden>...</span>
        <div className="tm-status" data-curve="x" aria-live="polite" />
        <span className="tm-more" data-curve="x" aria-hidden="true" hidden>...</span>
      </div>
      <div className="tm-srow">
        <span className="tm-more" data-curve="x" aria-hidden="true" hidden>...</span>
        <div className="tm-sugg" data-curve="x" />
        <span className="tm-more" data-curve="x" aria-hidden="true" hidden>...</span>
      </div>
      <div className="tm-pline" data-curve="x">
        <span className="tm-ps">&gt; </span><span className="tm-typed" /><span className="tm-bk"> </span><span className="tm-typed tm-after" />
        <textarea className="tm-input" rows={1} aria-label="type a command" autoComplete="off" autoCapitalize="off" autoCorrect="off" spellCheck="false" enterKeyHint="go" />
      </div>
      <span className="tm-measure" aria-hidden="true">MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM</span>
    </div>
  );
}

// running a line (Nov 26): the language's own rules, in one place, shared by the browser terminal and /api/run so the
// two can't drift. The host says what it does with a page and gives its own controls through `ui`:
//   ps()              the prompt, for the echo
//   echo              show the line as typed (Nov 25: every command does)
//   typed(text)       a line was entered (history)
//   asked()           a question was answered (the browser resets its prompt)
//   busy(on)          a command is running
//   pasted(text)      { words, docs }: long pastes held aside, sent to the model as files
//   result(r, cmd)    a command's page: its path and place
//   clear(), back()   the browser's own controls; absent elsewhere
import { line, ascii, Am, Dm, K } from './text';
import { run, chatInput } from './commands';

const INLINE_KEY = /\brpa_[A-Za-z0-9_-]{12,}/;

export async function execute(ctx, raw, ui = {}) {
  const st = ctx.st;
  const text = String(raw || '').trim();
  const secret = !!st.ask?.secret;
  const fail = (e) => ctx.print(() => [line(Am(ascii(e?.message || String(e))))]);
  // refuse inline keys before echo, history, search or navigation can observe them
  if (!secret && (INLINE_KEY.test(text) || /^\/?runpod\s+key\s+\S+/i.test(text))) {
    await ctx.print(() => [line(Am('use runpod key, then enter the key at its hidden prompt.'))]); return;
  }
  let echoed = ui.echo === false ? null : { ps: ui.ps ? ui.ps() : '> ', text: secret ? '[hidden]' : text };
  if (text && !secret && ui.typed) ui.typed(text);
  if (st.ask) {
    const a = st.ask; st.ask = null; if (ui.asked) ui.asked();
    if (ui.busy) ui.busy(true);
    try { await ctx.print(() => [], echoed); await a.answer(text); }
    catch (e) { await fail(e); }
    finally { if (ui.busy) ui.busy(false); }
    return;
  }
  if (!text) return;
  // language v1 (Nov 25, James): a command starts with /. anything else is the stream: said to the model you're
  // talking to, or, outside a conversation, written to the stream (one day, the model's input too)
  if (!text.startsWith('/') && !(st.model && st.chat)) { await ctx.print(() => [], echoed); return; }
  if (text === '/clear' && ui.clear) { ui.clear(); return; }
  if (text === '/back' && ui.back) { ui.back(); return; }
  if (st.model && st.chat) {
    if (!text.startsWith('/')) {
      const { words, docs } = ui.pasted ? ui.pasted(text) : { words: text, docs: [] };
      try { await chatInput(ctx, words, docs); } catch (e) { await fail(e); }
      return;
    }
    if (echoed) { await ctx.print(() => [], echoed); echoed = null; }      // a /command in a chat is a command too
    try { const r = await chatInput(ctx, text); if (r !== null) return; } catch (e) { await fail(e); return; }
  }
  const cmd = text.slice(1);
  if (ui.busy) ui.busy(true);
  let res;
  try { res = await run(ctx, cmd); } catch (e) {
    res = { render: () => [line(Am(e.status === 404 ? `nothing found: ${ascii(cmd)}` : `that didn't work: ${ascii(e.message)}`)), line(Dm('try '), K('help'))] };
  } finally { if (ui.busy) ui.busy(false); }
  if (!res) return;
  const r = typeof res === 'function' ? { render: res } : res;
  if (ui.result) ui.result(r, cmd);
  if (r.sugg) ctx.suggest(r.sugg);
  await ctx.print(r.render, echoed);
  if (r.after) r.after();
}

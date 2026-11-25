// A model's files (Nov 23, James): "one folder per model completes the picture. a model moves around and always has access
// to whatever files people gave it, wherever they gave them." Guests see the owner's files and their own, and manage their
// own; the owner manages all. Outside a conversation the list comes from logOS (what the model's holder last reported), and
// a change waits there until the model next loads if nothing holds it now.
import { S, Dm, Hi, Am, K, line, blank, ascii, fmtBytes, P, H, cut } from './text';

const when = (iso) => (iso ? ascii(new Date(iso).toLocaleString('en-AU', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })) : '');
const enc = encodeURIComponent;

function who(f, guest) {
  if (f.sender) return f.sender === 'you' ? 'you' : `@${ascii(f.sender)}`;
  return guest ? 'the owner' : 'you';
}

// ---- outside a conversation: files <model>, delete file <name> [from <model>], rename file <a> to <b> [in <model>]
export async function filesCommand(ctx, t, { get, post }) {
  let m;
  const R = (render, path, sugg) => ({ render, path, sugg });
  if ((m = t.match(/^files(?:\s+(?:of\s+|for\s+)?(.+))?$/i))) {
    const ref = (m[1] || ctx.st.filesModel || '').trim();
    if (!ref) return R(() => [line(Dm('whose files? '), K('files <model>', 'files '))], null, ['my models']);
    if (!ctx.user()) return R(() => [line(Dm('sign in to see a model\'s files. '), K('sign in'))], null, ['sign in']);
    let d; try { d = await get(`/me/models/${enc(ref)}/files`); } catch (e) { return R(() => [line(Am(ascii(e.message)))], null, ['my models']); }
    ctx.st.filesModel = d.model;
    return R((c) => filesPage(c, d), null, d.files.length ? [`talk to ${d.model}`, 'my models'] : ['my models']);
  }
  if ((m = t.match(/^delete\s+file\s+(.+?)(?:\s+from\s+(\S+))?$/i))) {
    const name = m[1].trim(); const ref = (m[2] || ctx.st.filesModel || '').trim();
    if (!ref) return R(() => [line(Dm('from which model? '), K(`delete file ${name} from <model>`, `delete file ${name} from `))], null);
    ctx.ask(`delete ${ascii(name)} from ${ascii(ref)} for good? [y/n]> `, async (a) => {
      if (!/^(y|yes)$/i.test(a)) return ctx.print(() => [line(Dm('kept.'))]);
      try {
        const r = await post(`/me/models/${enc(ref)}/files/delete`, { name });
        return ctx.print(() => [line(r.status === 'waiting' ? Dm(`${ascii(name)} goes when ${ascii(ref)} next loads.`) : Dm(`asked ${ascii(r.where || 'its machine')} to delete ${ascii(name)}. `), K('files', `files ${ref}`))]);
      } catch (e) { return ctx.print(() => [line(Am(ascii(e.message)))]); }
    }, ['yes', 'no']);
    return R(() => [], null);
  }
  if ((m = t.match(/^rename\s+file\s+(.+?)\s+to\s+(.+?)(?:\s+in\s+(\S+))?$/i))) {
    const [name, to] = [m[1].trim(), m[2].trim()]; const ref = (m[3] || ctx.st.filesModel || '').trim();
    if (!ref) return R(() => [line(Dm('in which model? '), K(`rename file ${name} to ${to} in <model>`, `rename file ${name} to ${to} in `))], null);
    try {
      const r = await post(`/me/models/${enc(ref)}/files/rename`, { name, to });
      return R(() => [line(r.status === 'waiting' ? Dm(`${ascii(name)} becomes ${ascii(to)} when ${ascii(ref)} next loads. `) : Dm(`asked ${ascii(r.where || 'its machine')} to rename ${ascii(name)} to ${ascii(to)}. `), K('files', `files ${ref}`))], null);
    } catch (e) { return R(() => [line(Am(ascii(e.message)))], null); }
  }
  return null;
}

function filesPage(c, d) {
  const ref = d.model;
  const where = d.live ? `held now by ${ascii(d.live)}` : d.as_of ? `as ${ascii(ref)} last reported them, ${when(d.as_of)}` : '';
  const rows = d.files.map((f) => {
    const acts = f.pending === 'delete' ? [Am('  goes when it next loads')]
      : f.pending === 'rename' ? [Am(`  becomes ${ascii(f.pending_to || '')} when it next loads`)]
        : f.mine ? [Dm('  '), K('rename', `rename file ${f.name} to `), Dm(' '), K('delete', `delete file ${f.name} from ${ref}`)] : [];
    return line('  ', Hi(cut(ascii(f.name), Math.max(16, c - 44))), Dm(`  ${fmtBytes(f.bytes || 0)}  ${who(f, d.guest)}`), ...acts);
  });
  return [...H(c, `${ascii(ref)}'s files`), ...(where ? [line(Dm(where))] : []), blank(),
    ...(rows.length ? rows : [line(Dm('  none yet.'))]), blank(),
    ...P(c, Dm(d.guest ? 'you see the owner\'s files and the ones you sent, and manage yours. ' : 'they go wherever the model goes. '),
      Dm('while talking to it, '), K('/pick'), Dm(' sends more and '), K('/read <file>', '/read '), Dm(' reads one.'))];
}

// ---- in a conversation: /files (read or paste each), /take file <name>; renaming and deleting are /files <model>'s
export async function sessionFiles(ctx, t, ch, { commandResult }) {
  let m;
  const me = ctx.user()?.user_id || '';
  const guest = Boolean(ch?.shared);
  if (/^\/files$/i.test(t)) {
    const data = ch?.s?.data || []; const shared = guest ? [] : (ch?.s?.shared || []);
    const name = ch?.s?.model || ch?.name || 'its';
    return ctx.print((c) => [...H(c, `${ascii(name)}'s files`),
      // in a conversation (Nov 25, James): read it (its own store; the weights keep it) or paste it (into this chat's
      // store, working memory). renaming and deleting are /files <model>'s
      ...(data.length ? data.map((f) => line('  ', Hi(cut(ascii(f.name), Math.max(16, c - 50))), Dm(`  ${fmtBytes(f.size)}  ${f.sender_name ? '@' + ascii(f.sender_name) : guest && f.sender !== me ? 'the owner' : 'you'}   `),
        K('/read', `/read ${f.name}`), S(' '), K('/paste', `/paste ${f.name}`)))
        : [line(Dm('  none yet.'))]),
      ...(shared.length ? [blank(), line(Dm('in the older shared folder (every model here read from it before each had its own):')),
        ...shared.map((f) => line('  ', Hi(cut(ascii(f.name), Math.max(16, c - 40))), Dm(`  ${fmtBytes(f.bytes || 0)}  `), K('take', `/take file ${f.name}`))),
        line(Dm('reading one moves it into this model\'s folder too.'))] : []),
      blank(), line(K('/pick'), S('   '), K(`/files ${name}`, `files ${ch?.s?.model || ''}`))]);
  }
  const act = async (kind, args, done) => {
    const r = await commandResult(ctx, kind, args);
    return ctx.print(() => [line(r.ok ? Dm(done(r.result || {})) : Am(ascii(r.error || 'it failed')))]);
  };
  if ((m = t.match(/^\/take\s+file\s+(.+)$/i))) {
    const name = m[1].trim();
    return act('data.take', { name }, () => `${ascii(name)} is one of ${ascii(ch?.s?.model || 'its')} files now.`);
  }
  return null;
}

// admin, in the terminal (Nov 20): the numbers, the people, the models. Only an admin's session gets answers.
import { K, Dm, Hi, Am, line, blank, H, ascii, kv, table, fmtBytes, fmtCount, fmtDate } from './text';

const yes = (s) => /^(y|yes)$/i.test(s);

// a frozen publication names the model it was published from; anything else is a working model or an upload
function kindOf(x, all) {
  const from = x.details?.publication_of || x.publication_of;
  if (from) return `publication of ${ascii((all.find((y) => y.checkpoint_id === from) || {}).name || from.slice(0, 8))}`;
  return x.visibility === 'public' ? 'public' : 'private';
}

export async function adminCommand(ctx, t, api) {
  let m;
  if (!/^admin(\s|$)/i.test(t)) return null;
  if (ctx.user()?.role !== 'admin') return { render: () => [line(Dm('admin is for admins.'))] };
  if (/^admin$/i.test(t)) {
    const s = await api.get('/admin/stats');
    return { path: '/admin', sugg: ['admin users', 'admin models'], render: (c) => [...H(c, 'admin'),
      kv('people', `${fmtCount(s.total_users)}  (${s.verified_count} verified, ${s.admin_count} admin)`, 16),
      kv('models', `${fmtCount(s.total_checkpoints)}  (${s.checkpoints_this_week} this week)`, 16),
      kv('downloads', `${fmtCount(s.total_downloads)} models, ${fmtCount(s.soma_downloads || 0)} soma`, 16),
      kv('storage', fmtBytes(s.total_storage_bytes), 16), blank(),
      line(K('admin users'), '  ', K('admin models'))] };
  }
  if (/^admin users$/i.test(t)) {
    const users = await api.get('/admin/users');
    return { path: null, sugg: ['admin', 'admin models'], render: (c) => [...H(c, 'people', String(users.length)),
      ...table(c, users, [
        { head: 'name', w: 18, get: (u) => K(ascii(u.username), `whois ${u.username}`) },
        { head: 'email', flex: true, min: 70, get: (u) => Dm(ascii(u.email || '')) },
        { head: 'role', w: 7, get: (u) => (u.role === 'admin' ? Hi('admin') : Dm('user')) },
        { head: 'models', w: 8, get: (u) => Dm(String(u.checkpoint_count ?? 0)) },
        { head: '', w: 22, get: (u) => K(u.verified ? 'verified (undo)' : 'verify', `admin ${u.verified ? 'unverify' : 'verify'} ${u.username}`) },
      ])] };
  }
  if ((m = t.match(/^admin (verify|unverify|promote|demote) @?(\S+)$/i))) {
    const [, act, name] = m; const verb = act.toLowerCase();
    const change = { verify: { verified: true }, unverify: { verified: false }, promote: { role: 'admin' }, demote: { role: 'user' } }[verb];
    await api.post('/admin/verify', { username: name, ...change });
    const done = { verify: 'verified', unverify: 'unverified', promote: 'promoted', demote: 'demoted' }[verb];
    return { render: () => [line(Dm(`${done} @${ascii(name)}.`)), line(K('admin users'))] };
  }
  if (/^admin models$/i.test(t)) {
    const ms = await api.get('/admin/checkpoints');
    ctx.remember(ms);
    return { path: null, sugg: ['admin', 'admin users'], render: (c) => [...H(c, 'models', String(ms.length)),
      ...table(c, ms, [
        { head: 'model', flex: true, get: (x) => K(ascii(x.name), `open ${x.checkpoint_id}`) },
        { head: 'kind', w: 22, min: 66, get: (x) => Dm(kindOf(x, ms)) },
        { head: 'by', w: 14, min: 60, get: (x) => Dm(ascii(x.creator_username || '')) },
        { head: 'size', w: 9, get: (x) => Dm(fmtBytes(x.file_size_bytes || 0)) },
        { head: 'dl', w: 6, min: 50, get: (x) => Dm(String(x.download_count || 0)) },
        { head: 'made', w: 13, min: 74, get: (x) => Dm(fmtDate(x.created_at)) },
        { head: '', w: 7, get: (x) => K('delete', `admin delete ${x.checkpoint_id}`) },
      ])] };
  }
  if ((m = t.match(/^admin delete ([0-9a-f]{32})$/i))) {
    const id = m[1];
    ctx.ask(`delete ${id.slice(0, 8)}... and its files for good? this cannot be undone. [y/n]> `, async (a) => {
      if (!yes(a)) return ctx.print(() => [line(Dm('kept.'))]);
      await api.request(`/checkpoints/${id}`, 'DELETE');
      return ctx.print(() => [line(Dm('deleted.')), line(K('admin models'))]);
    }, ['no', 'yes']);
    return { render: () => [line(Am('a deletion removes the model and its stored files.'))] };
  }
  return { render: () => [line(Dm('admin, admin users, admin models, admin verify|unverify|promote|demote <name>, admin delete <model id>.'))] };
}

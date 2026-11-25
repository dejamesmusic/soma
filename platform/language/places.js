// Places (Nov 20, James): a model is stored on a machine's disk, on logOS, or on a RunPod volume, and runs only where
// it is stored: on that computer, or on a cloud GPU mounted on that volume. One place holds its newest revision; a
// move sends it somewhere else and deletes the copy it left once the new one is verified. Every page here shows the
// same places and the one action that fits each.
import { S, K, Dm, Hi, Am, H, line, blank, ascii, fmtBytes, pad } from './text';

const yes = (a) => /^(y|yes)$/i.test(a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const enc = encodeURIComponent;
const short = (r) => (r ? String(r).slice(0, 8) : '');
export const where = (p) => (p.kind === 'computer' ? ascii((p.name||'computer').replace(/^soma (harness )?on /, '')) : p.kind === 'logos' ? 'logOS' : `runpod ${p.data_center}`);
export const newestOf = (ps) => ps.places.find((x) => x.newest) || null;

// this machine, when the terminal is the soma app's
export async function thisComputer(ps) {
  if (!window.soma?.local) return null;
  try {
    const me = (await window.soma.local.presence()).app?.machine;
    return ps.places.find((x) => x.kind === 'computer' && x.name === `soma on ${me}`) || null;
  } catch (_) { return null; }
}

export async function places(api, ref) { return api.get(`/me/models/${enc(ref)}/places`); }

function loadedLine(ps) {
  const l = ps.loaded;
  if (!l) return [Dm('nowhere')];
  if (l.kind === 'gpu') {
    const ending = /saving|stopping/i.test(l.note || '') || l.status === 'stopping';
    return [Hi(`a gpu in ${ascii(l.data_center || '')}`), Dm(`  ${ascii(l.gpu || '')} $${Number(l.rate || 0).toFixed(2)}/h `),
      ending ? Am(ascii(/saving|stopping/i.test(l.note || '') ? l.note : 'stopping')) : Dm(ascii(l.status || ''))];
  }
  return [Hi(where({ kind: 'computer', name: l.name }))];
}

// one model's row on `my models`
export function summary(ps) {
  const n = newestOf(ps); const older = ps.places.filter((x) => x.stored && !x.newest);
  return [
    line(Dm('   stored  '), n ? Hi(where(n)) : Am('nowhere'), ...(older.length ? [Dm(`   older: ${older.map(where).join(', ')}`)] : [])),
    line(Dm('   loaded  '), ...loadedLine(ps)),
  ];
}

// `where <model>`: every place, its state, and the action that fits it
export async function placesPage(ctx, api, ref) {
  const ps = await places(api, ref); const m = ps.model; const r = m.ref;
  const me = await thisComputer(ps); const n = newestOf(ps);
  ctx.st.here = { kind: 'library-model', id: r, name: m.name };
  const row = (p) => {
    const state = p.newest ? Hi('newest') : p.stored ? Dm('older ') : Dm('-     ');
    const rev = p.revision_id ? Dm(` ${short(p.revision_id)}`) : S('');
    const acts = [];
    if (p.kind === 'computer' && !p.engine) acts.push(...(p.legacy ? [Dm('soma 1 '), K('unlink', `unlink ${p.token_id}`)] : [K('update soma','get soma')]));
    else if (!p.newest && p.kind !== 'volume' && !ps.loaded) acts.push(K('move here', `move ${r} to ${p.kind === 'logos' ? 'logos' : p.id}`));
    if (p.stored && !p.newest && !(p.kind === 'logos' && m.reference)) acts.push(S('  '), K('remove', `remove ${r} from ${p.id}`));
    if (p.newest && !ps.loaded && (p.kind === 'computer' || p.kind === 'volume')) acts.push(K('load here', p.kind === 'volume' ? `load ${r} on gpu` : `load ${r} on ${p.id}`));
    const tag = p.kind === 'computer' ? Dm(p.online ? (me && me.id === p.id ? ' (this one)' : '') : ' (offline)') : p.kind === 'logos' && m.reference ? Dm(' (the publication)') : S('');
    return line('  ', S(pad(where(p), 18)), tag, S('  '), state, rev, S('  '), ...acts);
  };
  const storedRows = ps.places.filter((x) => x.kind !== 'volume' || x.stored).map(row);
  const loadActs = ps.loaded
    ? [K('talk', `talk to ${r}`), S('  '), K('unload', `unload ${r}`)]
    : [...(me ? [K('load on this machine', `load ${r} on this machine`), S('  ')] : []),
      K(n && n.kind === 'volume' ? `load on a gpu in ${n.data_center}` : 'load on a gpu', `load ${r} on gpu`)];
  return {
    path: `/library/${enc(r)}`, sugg: ps.loaded ? [`talk to ${r}`, `unload ${r}`] : [me ? `load ${r} on this machine` : `load ${r} on gpu`, 'my models'],
    render: (c) => [...H(c, m.name, m.reference ? 'a publication you added; its first save makes your own model' : m.not_backed_up ? 'only on a machine' : 'private model'),
      line(Dm(`${fmtBytes(m.bytes || 0)}  newest ${short(m.head_revision_id) || '?'}`)),
      ...(m.reference ? [line(Dm('from publication '),K(short(m.source_publication_id),`open ${m.source_publication_id}`))] : []), blank(),
      line(Hi('stored')), ...storedRows,
      line('  ', S(pad('a runpod volume', 18)), S('  '), Dm('-     '), S('  '), ...(ps.loaded ? [] : [K('move to runpod', `move ${r} to runpod `)])),
      blank(), line(Hi('loaded'), Dm('  '), ...loadedLine(ps)), line('  ', ...loadActs),
      ...ps.moving.map((x) => line(Am(`moving: a ${ascii(x.purpose)} machine in ${ascii(x.data_center || '')} is ${ascii(x.status || '')}`))),
      ...ps.warnings.map((x) => line(Am(ascii(x)))),
      blank(), line(K('my models'), S('  '), K('rename', `rename ${r} to `), S('  '), ...(m.reference ? [K('remove from my models', `delete ${r}`)] : [K('publish a snapshot', `publish ${r}`), S('  '), K('delete', `delete ${r}`)])),
    ],
  };
}

// place words -> the API's `to`
async function resolvePlace(ps, words) {
  const w = words.trim().toLowerCase();
  if (/^log ?os$/.test(w)) return { to: 'logos', label: 'logOS' };
  if (w.startsWith('computer:')) { const p = ps.places.find((x) => x.id === words.trim()); return p ? { to: p.id, label: where(p), place: p } : null; }
  if (['this machine','this computer','here','mac','pc'].includes(w)) { const p = await thisComputer(ps); return p ? { to: p.id, label: where(p), place: p } : null; }
  let mm = w.match(/^(?:runpod|volume)?\s*([a-z]{2,4}-[a-z0-9]+-\d+)$/i);
  if (mm) return { to: `volume:${mm[1].toUpperCase()}`, label: `runpod ${mm[1].toUpperCase()}`, dc: mm[1].toUpperCase() };
  const p = ps.places.find((x) => x.kind === 'computer' && where(x).toLowerCase() === w);
  return p ? { to: p.id, label: where(p), place: p } : null;
}

function costLine(ps, from, to, rate) {
  const size = fmtBytes(ps.model.bytes || 0);
  const del = ps.model.reference ? 'the publication stays on logOS.' : `then ${from}'s copy is deleted.`;
  if (from.startsWith('runpod') || to.startsWith('runpod')) return `a cpu machine (${rate == null ? 'quote required' : '$'+Number(rate).toFixed(2)+'/h'}) copies ${size} from ${from} to ${to}; ${del}`;
  return `${size} goes from ${from} to ${to}; ${del}`;
}

// follow a hop until the newest copy is where it was sent (or it fails), printing how it goes
async function follow(ctx, api, ref, r, target) {
  const b = ctx.printLive(() => [line(Dm('moving ...'))]);
  let sawMachine = false;
  for (let i = 0; i < 4 * 720; i += 1) {
    await sleep(5000);
    let ps; try { ps = await places(api, ref); } catch (_) { continue; }
    if (r.command?.command_id) {
      const command=await api.get(`/me/commands/${r.command.command_id}`).catch(()=>null);
      if (command && (['failed','expired'].includes(command.status) || command.args?.move_state==='failed')) {
        ctx.update(b,()=>[line(Am(ascii(command.result?.error || 'transfer failed; the source is kept. retry after checking where.')))]); return null;
      }
    }
    const n = newestOf(ps);
    const there = n && (target.to === 'logos' ? n.kind === 'logos' : target.to.startsWith('volume:') ? n.kind === 'volume' && `volume:${n.data_center}` === target.to : n.id === target.to);
    if (there && !ps.moving.length) { ctx.update(b, () => [line(Hi(`${ascii(ps.model.name)} is on ${where(n)}.`))]); return ps; }
    const c = ps.places.find((x) => x.transfer?.running || x.backup?.running || (x.transfer && /failed/.test(x.transfer.status || '')) || (x.backup && /failed/.test(x.backup.status || '')));
    const t = c && (c.transfer || c.backup);
    if (t && /failed/.test(t.status || '')) { ctx.update(b, () => [line(Am(ascii(t.status)))]); return null; }
    if (t) {
      const frac = t.total ? t.done / t.total : 0;
      ctx.update(b, () => [line(Dm(`${t.kind === 'download' ? 'downloading to' : 'uploading from'} ${where(c)}  `), S(t.total ? `${(frac * 100).toFixed(1)}%  ${fmtBytes(t.done)} of ${fmtBytes(t.total)}` : ascii(t.status || '')))]);
      continue;
    }
    if (ps.moving.length) { sawMachine = true; const x = ps.moving[0]; ctx.update(b, () => [line(Dm(`a ${x.purpose} machine in ${ascii(x.data_center || '')} is ${ascii(x.status || 'starting')}`))]); continue; }
    if (r.machine) {
      const mach = await api.get(`/me/cloud/machines/${r.machine.machine_id}`).catch(() => null);
      if (mach && /failed/.test(mach.machine?.status || mach.status || '')) { ctx.update(b, () => [line(Am(`the machine failed: ${ascii(mach.machine?.note || mach.note || '')}`))]); return null; }
    }
  }
  ctx.update(b, () => [line(Am('still moving after four hours; where shows how it stands.'))]);
  return null;
}

// move <model> to <place>: one confirmation for the whole route, then each hop in turn
export async function moveCmd(ctx, api, ref, words, after, extra=[]) {
  let ps = await places(api, ref);
  const target = await resolvePlace(ps, words);
  if (!target) {
    if (/^runpod\s*$/i.test(words.trim())) {
      ctx.ask('which data center? (for example EU-RO-1)> ', (a) => moveCmd(ctx, api, ref, `runpod ${a}`, after).then((x) => x && ctx.print(x.render)), ['EU-RO-1', 'US-GA-2']);
      return { render: () => [line(Dm('which data center?'))] };
    }
    return { render: () => [line(Dm(`no place called '${ascii(words)}'. places: this machine, logos, runpod <data center>, or a linked machine's name.`))] };
  }
  const n = newestOf(ps);
  if (n && (n.id === target.to || (target.to === 'logos' && n.kind === 'logos'))) {
    if (after) return after(ps);
    return { render: () => [line(Dm(`${ascii(ps.model.name)} is already newest on ${where(n)}.`))] };
  }
  if (ps.loaded) return { render: () => [line(Am(`${ascii(ps.model.name)} is loaded; `), K('unload', `unload ${ps.model.ref}`), Dm(' it first.'))] };
  const from = n ? where(n) : ps.model.reference ? 'the publication' : 'nowhere';
  const twoHops = n && n.kind !== 'logos' && target.to !== 'logos' && !ps.model.reference;
  const quote=await api.get(`/me/models/${enc(ps.model.ref)}/move-quote?to=${enc(target.to)}`);
  const legs=quote.legs||[];
  const plan = (twoHops ? [costLine(ps, from, 'logOS',legs[0]?.rate), costLine(ps, 'logOS', target.label,legs[1]?.rate)] : [costLine(ps, from, target.label,legs[0]?.rate)]).concat(extra);
  ctx.ask(`move ${ascii(ps.model.name)} to ${target.label}${after ? ', then load it' : ''}? [y/n]> `, async (a) => {
    if (!yes(a)) return ctx.print(() => [line(Dm('nothing moved.'))]);
    let to = target.to; let onward = null;
    for (let hop = 0; hop < 3 && to; hop += 1) {
      let r;
      try { r = await api.post(`/me/models/${enc(ps.model.ref)}/move`, { to, max_rate: legs.find(l=>l.data_center===(ps.home.kind==='volume'?ps.home.data_center:target.dc))?.rate ?? 0.2 }); }
      catch (e) { return ctx.print(() => [line(Am(ascii(e.message)))]); }
      if (r.already) break;
      const hopTarget = r.then ? { to: 'logos' } : { to };
      ps = await follow(ctx, api, ps.model.ref, r, hopTarget);
      if (!ps) return null;
      to = r.then || null; onward = r.transit || null;
    }
    // to another data center, logOS takes it the last hop itself (Nov 25): nothing to wait for here
    if (onward) return ctx.print(() => [line(Dm(`on logOS; going on to runpod ${ascii(onward)} by itself. `), K('where', `where ${ps.model.ref}`))]);
    if (after) { const x = await after(ps); if (x) await ctx.print(x.render || x); return null; }
    return ctx.print(() => [line(K('where', `where ${ps.model.ref}`))]);
  }, ['yes', 'no']);
  return { render: () => [line(Dm(`${ascii(ps.model.name)} is newest on ${from}.`)), ...plan.map((x) => line(Dm('  ' + x)))] };
}

export async function removeCmd(ctx, api, ref, placeId) {
  const ps = await places(api, ref);
  const p = ps.places.find((x) => x.id === placeId || where(x).toLowerCase() === placeId.toLowerCase() || (placeId === 'logos' && x.kind === 'logos'));
  if (!p || !p.stored) return { render: () => [line(Dm('no copy there.'))] };
  if (p.newest) return { render: () => [line(Dm(`that's the newest copy: `), K('move', `move ${ps.model.ref} to `), Dm(' it, or '), K('delete', `delete ${ps.model.ref}`), Dm(' the model.'))] };
  ctx.ask(`delete the older copy of ${ascii(ps.model.name)} on ${where(p)}? [y/n]> `, async (a) => {
    if (!yes(a)) return ctx.print(() => [line(Dm('kept.'))]);
    try { await api.post(`/me/models/${enc(ps.model.ref)}/remove`, { place: p.id, max_rate: 0.2 }); }
    catch (e) { return ctx.print(() => [line(Am(ascii(e.message)))]); }
    return ctx.print(() => [line(Dm(p.kind === 'volume' ? 'a cpu machine ($0.12/h) is deleting it.' : 'deleting it.')), line(K('where', `where ${ps.model.ref}`))]);
  }, ['yes', 'no']);
  return { render: () => [line(Dm(`older copy on ${where(p)}`), Dm(p.revision_id ? `  ${short(p.revision_id)}` : ''))] };
}

// load <model> on <this machine | a machine | gpu>: where it's stored, or moved there first (one confirmation)
export async function loadCmd(ctx, api, ref, words, h) {
  const ps = await places(api, ref); const m = ps.model; const n = newestOf(ps);
  if (ps.loaded) return h.talk(ps);
  const w = (words || '').trim().toLowerCase();
  if (!w) {
    const me = await thisComputer(ps);
    ctx.ask(`load ${ascii(m.name)} where? [${me ? 'this machine / ' : ''}gpu]> `, (a) => loadCmd(ctx, api, ref, a, h).then((x) => x && ctx.print(x.render || x)), me ? ['this machine', 'gpu'] : ['gpu']);
    return { render: () => [line(Dm(`${ascii(m.name)} is newest on ${n ? where(n) : 'nowhere'}.`))] };
  }
  if (/^(a )?(gpu|cloud|runpod)/.test(w)) {
    const dcWord = w.replace(/^(a )?(gpu|cloud|runpod)\s*(in\s*)?/, '').trim();
    if (n && n.kind === 'volume' && (!dcWord || dcWord.toUpperCase()===n.data_center)) return h.gpu(ps, n.data_center);
    if (!dcWord) {
      ctx.ask('a gpu in which data center? (for example EU-RO-1)> ', (a) => loadCmd(ctx, api, ref, `gpu in ${a}`, h).then((x) => x && ctx.print(x.render || x)), ['EU-RO-1', 'US-GA-2']);
      return { render: () => [line(Dm(`a gpu loads it from a volume; ${ascii(m.name)} is on ${n ? where(n) : 'nowhere'}, so it moves to a volume first.`))] };
    }
    const quote=await api.get(`/me/cloud/offers?${new URLSearchParams({model:m.ref,data_center:dcWord.toUpperCase(),purpose:'chat'})}`);
    const offer=quote.offers?.[0];
    if(!offer) return {render:()=>[line(Am('no suitable gpu is available there; nothing moved or rented.'))]};
    return moveCmd(ctx, api, ref, `runpod ${dcWord}`, (after) => h.gpu(after, dcWord.toUpperCase(),{quote,offer}),
      [`then load ${offer.name} at $${Number(offer.price).toFixed(2)}/h; volume storage ${quote.storage?.monthly_usd==null?'is billed separately':'$'+Number(quote.storage.monthly_usd).toFixed(2)+'/month'}.`]);
  }
  const target = await resolvePlace(ps, w);
  if (!target || !target.place) return { render: () => [line(Dm(`load it on this machine, a linked machine by name, or a gpu.`))] };
  if (n && n.id === target.to) return h.computer(ps, target.place);
  return moveCmd(ctx, api, ref, target.to, (after) => h.computer(after, after.places.find((x) => x.id === target.to)));
}

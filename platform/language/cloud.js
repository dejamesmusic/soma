// The cloud uses the app's session protocol. Only quotes allocate nothing; a confirmed start rents compute.
import { K, X, S, Dm, Hi, Am, line, blank, P, H, ascii, fmtDur, fmtDate, pad } from './text';

const yes = (s) => /^(y|yes)$/i.test(s);
const id = encodeURIComponent;
const page = (render, path = null, sugg = ['cloud', 'my models']) => ({ render, path, sugg });
const rate = (n) => `$${Number(n || 0).toFixed(2)}/h`;
const card = (m) => [
  line(Hi(ascii(m.model)), Dm(`  ${ascii(m.status)}  ${rate(m.rate)}  ${ascii(m.data_center || '')}`)),
  line(Dm(`  ${ascii(m.gpu || '')}  ${fmtDur((m.hours || 0) * 3600)}`)),
  ...(m.note ? [line(Dm(`  ${ascii(m.note)}`))] : []),
  line('  ', K('status', `cloud status ${m.machine_id}`), '  ',
    ...(!m.ended_at && m.model ? [K('talk', `talk to ${m.model}`), '  ', K('unload', `unload ${m.model}`)] : [])),
];

export async function watch(ctx, machine, api, { talking = false, since: from = 0 } = {}) {
  const b = ctx.printLive(() => card(machine));
  const until = Date.now() + 12 * 60 * 1000;
  let m = machine;
  const since = from || Date.parse(machine.created_at || '') || Date.now();
  const alive = (x) => ctx.setStatus?.({ spin: true, since, segs: [Dm(`${ascii(x.model)}: ${ascii(x.note || (x.status === 'starting' ? 'renting the machine and starting it' : x.status))}`)] });
  while (ctx.active() && Date.now() < until) {
    try {
      m = (await api.get(`/me/cloud/machines/${id(machine.machine_id)}`)).machine;
      ctx.update(b, () => card(m));
      if (m.ended_at) {                            // finished: say so, with what it did
        ctx.setStatus?.(null);
        await ctx.print(() => [line(m.status === 'failed' ? Am('finished, with a problem: ') : Hi('finished: '), Dm(ascii(m.note || m.status || 'ended')))]);
        return;
      }
      alive(m);                                    // Nov 23: a starting machine shows what it's doing, and for how long
      if (talking && m.online) {
        const s = await api.get(`/me/apps/${id(m.token_id)}/session?rev=-1&wait=0`);
        if (s.session?.loaded) {
          ctx.setStatus?.(null);
          const r = await api.chat(ctx, m.token_id);
          await ctx.print(r.render);
          if (r.sugg) ctx.suggest(r.sugg);
          if (r.after) r.after();
          return;
        }
      }
    } catch (e) {
      ctx.setStatus?.(null);
      ctx.update(b, () => [...card(m), line(Am(ascii(e.message))), line(K('check again', `cloud status ${machine.machine_id}`))]);
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

export async function runCloud(ctx, raw, api) {
  const t = raw.trim(); let m;
  if (!/^(cloud(?:\s|$)|runpod(?:\s|$))/i.test(t)) return null;
  if (!ctx.user()) return page((c) => [...H(c, 'cloud'), line(K('sign in'))], '/cloud');

  if (/^runpod key$/i.test(t)) {
    ctx.ask('runpod key (hidden; escape cancels)> ', async (key) => {
      if (!key) return ctx.print(() => [line(Dm('cancelled.'))]);
      await api.request('/me/runpod/key', 'PUT', { key });
      ctx.suggest(['cloud', 'cloud offers']);
      return ctx.print(() => [line(Hi('runpod linked.')), line(K('cloud'))]);
    }, [], { secret: true });
    return page((c) => [...H(c, 'link runpod'), line(Dm('a restricted key: pods, network volumes, catalogue. it is encrypted and never echoed.'))], null, []);
  }
  if (/^runpod forget$/i.test(t)) {
    ctx.ask('forget the runpod key? [y/n]> ', async (a) => {
      if (!yes(a)) return ctx.print(() => [line(Dm('kept.'))]);
      await api.request('/me/runpod/key', 'DELETE');
      return ctx.print(() => [line(Dm('key forgotten.'))]);
    }, ['yes', 'no']);
    return page(() => [line(Dm('stop your cloud machines first. volumes stay in your runpod account.'))], null, ['yes', 'no']);
  }
  // the cloud page (Nov 23, James): only what bills now: running machines, volumes, stopped pods' disks. the past is
  // `cloud history`
  if (/^(cloud|runpod)$/i.test(t)) {
    const [d, store] = await Promise.all([api.get('/me/cloud'), api.get('/me/storage').catch(() => null)]);
    const running = d.machines.filter((x) => !x.ended_at);
    const perHour = running.reduce((a, x) => a + Number(x.rate || 0), 0);
    const perMonth = (d.volumes || []).reduce((a, v) => a + Number(v.monthly_usd || 0), 0);
    const credit = store?.account && store.account.balance != null ? `$${Number(store.account.balance).toFixed(2)} of runpod credit` : null;
    const machineLine = (m) => [
      line(Hi(ascii(m.model || m.purpose || 'machine')), Dm(`  ${ascii(m.gpu || 'cpu')}  ${ascii(m.data_center || '')}  ${rate(m.rate)}  up ${fmtDur((m.hours || 0) * 3600)}`
        + `  (~$${(Number(m.rate || 0) * (m.hours || 0)).toFixed(2)} so far)`)),
      line(Dm(`  ${ascii(m.purpose === 'chat' ? '' : `${m.purpose}: `)}${ascii(m.note || m.status || '')}`)),
      line('  ', ...(m.purpose === 'chat' && m.model ? [K('talk', `talk to ${m.model}`), '  ', K('unload', `unload ${m.model}`), '  '] : []), K('status', `cloud status ${m.machine_id}`)),
    ];
    return page((c) => [...H(c, 'your cloud', 'what bills now'),
      line(d.runpod.linked ? Hi(`runpod linked (ending ${ascii(d.runpod.hint)})`) : Dm('no runpod account linked. '), S('  '), K(d.runpod.linked ? 'replace key' : 'link runpod', d.runpod.linked ? 'runpod key' : 'runpod setup'),
        ...(credit ? [Dm(`   ${credit}  `), K('notices')] : [])),
      ...(d.runpod.problem ? [line(Am(ascii(d.runpod.problem)))] : []),
      blank(),
      line(Hi('running now'), Dm(running.length ? `   $${perHour.toFixed(2)}/h` : '')),
      ...(running.length ? running.flatMap(machineLine) : [line(Dm('  nothing: no gpu or cpu machine is billing.'))]),
      blank(),
      line(Hi('storage'), Dm((d.volumes || []).length ? `   about $${perMonth.toFixed(2)}/month` : '')),
      ...((d.volumes || []).length ? d.volumes.map((v) => line(Dm('  '), S(pad(ascii(v.name), 22)), Dm(`  ${pad(`${v.size} GB`, 7)} ${pad(ascii(v.data_center), 9)} $${Number(v.monthly_usd || 0).toFixed(2)}/month`),
        Dm(v.models?.length ? `  holds ${v.models.map(ascii).join(', ')}` : '  holds no model of yours'))) : [line(Dm('  no network volumes.'))]),
      ...((d.stopped_pods || []).length ? [blank(), line(Am('stopped pods'), Dm('  (runpod keeps their disks, and bills for them)')),
        ...d.stopped_pods.map((x) => line(Dm('  '), S(pad(ascii(x.name), 22)), Dm(`  ${x.disk_gb} GB disk  ${ascii(x.data_center)}`))),
        line(Dm('  delete the ones you don\'t need in the '), X('runpod console', 'https://console.runpod.io/pods'), Dm('.'))] : []),
      blank(),
      line(K('cloud history'), '  ', K('cloud offers'), '  ', K('cloud start <model>', 'cloud start '), '  ', K('move <model>', 'move ')),
      line(K('forget key', 'runpod forget'), '  ', K('machines'), '  ', K('my models'))], '/cloud', d.runpod.linked ? ['cloud history', 'cloud offers', 'my models'] : ['runpod setup', 'get soma']);
  }
  if (/^cloud history$/i.test(t)) {
    const d = await api.get('/me/cloud/history');
    const what = (m) => (m.purpose === 'chat' ? 'talked' : m.purpose === 'move' ? 'moved' : m.purpose === 'home' ? 'brought home' : m.purpose === 'new' ? 'created' : ascii(m.purpose || ''));
    return page((c) => [...H(c, 'cloud history', `${d.machines.length} machines, about $${Number(d.total_usd || 0).toFixed(2)}`),
      ...(d.machines.length ? d.machines.flatMap((m) => [
        line(Dm(pad(fmtDate(m.created_at), 13)), Hi(pad(ascii(m.model || '-'), 24)), Dm(` ${pad(what(m), 13)} ${pad(ascii(m.gpu || 'cpu'), 10)} ${pad(fmtDur((m.hours || 0) * 3600), 8)} ~$${Number(m.cost_usd || 0).toFixed(2)}`),
          m.ended_at ? null : Am('  running')),
        ...(m.note ? [line(Dm(`${' '.repeat(13)}${ascii(m.note).slice(0, Math.max(20, c - 16))}`))] : []),
      ]) : [line(Dm('no cloud machines yet.'))]),
      blank(), ...P(c, Dm('costs are estimates (rate x time); runpod bills by the second. your volumes bill separately: '), K('cloud'), Dm('.'))], '/cloud/history', ['cloud', 'my models']);
  }
  if ((m = t.match(/^cloud (?:start|offers)\s*(.*?)$/i))) {                     // v1: moving is /move, ending is /unload
    const move = /^cloud move /i.test(t), only = /^cloud offers/i.test(t), home=/^cloud home /i.test(t);
    const model = m[1].trim(), dc = move ? m[2].toUpperCase() : (api.data_center||'');
    if (!model && !only) return page(() => [line(Dm('name a model backed up on logOS: cloud start <model>.'))]);
    const q = new URLSearchParams({ ...(model ? { model } : {}), ...(dc ? { data_center: dc } : {}), purpose: home?'home':move ? 'move' : 'chat' });
    const d = api.approved?.quote || await api.get(`/me/cloud/offers?${q}`);
    const choices = d.offers || [];
    const render = (c) => [...H(c, model ? `${move ? 'move' : 'run'} ${ascii(model)}` : 'runpod offers'),
      ...(d.capacity ? [line(Dm(`estimated gpu memory: ${d.capacity.gpu_gb} gb; disk: ${d.capacity.disk_gb} gb`))] : []),
      ...choices.map((o, i) => line(Hi(`${i + 1}. ${ascii(o.name)}`), Dm(`  ${o.memory} gb  ${rate(o.price)}; $${(o.price * d.cap_hours).toFixed(2)} at the hard limit`))),
      ...(choices.length ? [] : [line(Am('no suitable machine is free right now.'))]), blank(),
      ...P(c, `storage estimate: ${d.storage.gb} gb ${d.storage.kind}, $${d.storage.monthly_usd.toFixed(2)}/month while running; $${d.storage.stopped_monthly_usd.toFixed(2)}/month if retained after stopping.`),
      ...P(c, `idle limit ${d.idle_minutes} minutes; hard limit ${d.cap_hours} hours. runpod bills actual use. no resources allocated yet.`),
      ...(only ? [line(K('start a model', `cloud start ${model}`))] : [])];
    if (!only && choices.length) {
      const confirm = (o) => {
        const launch=async(a)=>{
          if (!yes(a)) return ctx.print(() => [line(Dm('cancelled; nothing rented.'))]);
          const r = await api.post('/me/cloud/machines', { model: d.model.reference||d.model.checkpoint_id, purpose: home?'home':move ? 'move' : 'chat',
            gpu: o.id, data_center: dc || d.data_center || o.data_centers[0], max_rate: o.price });
          if(r.already_home||r.already_there) {
            return ctx.print(()=>[line(Hi('already at that location; no machine, transfer or duplicate was created.'))]);
          }
          ctx.suggest([`cloud status ${r.machine.machine_id}`, `cloud stop ${r.machine.machine_id}`]);
          await ctx.print(() => [line(Hi('starting; the machine will save before it stops.'))]);
          watch(ctx, r.machine, api, { talking: !move&&!home });
        };
        if(api.approved) return launch('yes');
        ctx.ask(`start ${ascii(o.name)} at ${rate(o.price)} plus storage? [y/n]> `,launch, ['yes', 'no']);
      };
      if (api.approved) { await confirm(api.approved.offer); return page(()=>[line(Dm('loading on the quoted gpu.'))]); }
      else if (choices.length === 1) confirm(choices[0]);
      else ctx.ask(`machine [1-${choices.length}, or cancel]> `, async (a) => {
        if (!/^\d+$/.test(a) || !choices[Number(a) - 1]) return ctx.print(() => [line(Dm('cancelled; nothing rented.'))]);
        confirm(choices[Number(a) - 1]);
      }, choices.slice(0, 4).map((_, i) => String(i + 1)).concat(['cancel']));
    }
    return only ? page(render, null, ['cloud', 'cloud start ']) : { render };
  }
  if ((m = t.match(/^cloud adopt\s+(\S+)\s+(\/workspace\/\S+\.pt)\s+as\s+(.+)$/i))) {
    // a checkpoint made on one of your volumes becomes your private model, living there; nothing is copied
    const [, volume, path, name] = m;
    ctx.ask(`read ${ascii(path)} with a cpu machine beside ${ascii(volume)} (at most $0.20/h, minutes)? [y/n]> `, async (a) => {
      if (!yes(a)) return ctx.print(() => [line(Dm('cancelled; nothing rented.'))]);
      const r = await api.post('/me/cloud/adopt', { volume_id: volume, path, name: name.trim(), max_rate: 0.2 });
      ctx.suggest([`where ${name.trim()}`, 'cloud']);
      await ctx.print(() => [line(Hi(`reading it in place at ${rate(r.offer.price)}; it registers when its identity checks out.`))]);
      watch(ctx, r.machine, api);
    }, ['yes', 'no']);
    return page((c) => [...H(c, `adopt ${ascii(name)}`), line(Dm(`${ascii(volume)}  ${ascii(path)}`))], null, ['yes', 'no']);
  }
  if ((m = t.match(/^cloud (status|chat|stop)\s+(mch_[a-f0-9]+)(\s+now)?$/i))) {
    const action = m[1].toLowerCase(), machineId = m[2], force = !!m[3];
    const machine = (await api.get(`/me/cloud/machines/${id(machineId)}`)).machine;
    if (action === 'status') return page(() => card(machine));
    if (action === 'chat') {
      if (machine.ended_at) return page(() => [line(Dm('this machine has ended. '), K('start it again', `cloud start ${machine.model}`))]);
      if (machine.online) return api.chat(ctx, machine.token_id);
      return { ...page(() => card(machine)), after: () => watch(ctx, machine, api, { talking: true }) };
    }
    if (machine.ended_at) return page(() => [line(Dm('this machine has already ended.'))]);
    // unload (places, Nov 20): a gpu saves onto the volume it loaded from and stops; a move is a separate act
    const home = !machine.volume_id;                       // no volume: its disk goes with it, so the save goes to logOS
    ctx.ask(force ? 'stop without waiting for a save? [y/n]> ' : home ? 'save it to logOS and stop? [y/n]> ' : `save it on its volume in ${ascii(machine.data_center || '')} and stop? [y/n]> `, async (a) => {
      if (!yes(a)) return ctx.print(() => [line(Dm('kept running.'))]);
      const r = await api.post(`/me/cloud/machines/${id(machineId)}/stop`, { home, now: force });
      await ctx.print(() => [line(Dm(r.asked ? 'saving before it stops.' : r.disk_kept ? 'stopped; the disk is retained in runpod and still bills storage.' : 'stopped; the network volume is retained.'))]);
      if (r.asked) watch(ctx, machine, api, { since: Date.now() });
    }, ['yes', 'no']);
    return page((c) => [...card(machine), ...(force ? P(c, Am('unsaved work in memory may be lost. the disk or network volume is retained.')) : [])], null, ['yes', 'no']);
  }
  return page(() => [line(K('cloud'), S('  '), K('cloud offers'), S('  '), K('cloud start <model>', 'cloud start '), S('  '), K('cloud status <machine>', 'cloud status ')), line(K('cloud adopt <volume> <path> as <name>', 'cloud adopt '), S('  '), K('runpod key'))]);
}

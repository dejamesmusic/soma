import { S, K, Dm, Hi, Am, H, P, line, blank, ascii, fmtBytes } from './text';
import { runCloud, watch } from './cloud';
import { places, placesPage, summary, moveCmd, loadCmd, where } from './places';
const yes = (a) => /^(y|yes)$/i.test(a);
const refOf = (m) => m.library_entry_id || m.checkpoint_id;
const role = (m) => m.fork_on_save ? 'snapshot reference; first save creates a private model' : m.not_backed_up ? 'only on your machine; not backed up' : 'private working model';
export const locationLines = (w) => [
  line(Dm('working location: '), Hi(ascii(w.primary))),
  ...w.copies.map((x) => line('  ', S(ascii(x.label)), Dm(x.revision_id ? `  revision ${x.revision_id.slice(0, 12)}` : x.kind === 'mac' ? '  revision not reported' : ''),
    ...(x.path ? [Dm(`  ${ascii(x.path)}`)] : []),
    ...(x.kind === 'mac' ? [Dm(x.online ? '  online' : '  last reported, offline')] : []))),
  ...w.warnings.map((x) => line(Am(ascii(x)))),
];

export async function libraryPage(ctx, api, heading='my models') {
  if (!ctx.user()) { if (window.soma?.local) return (await import('./commands')).localModelsPage(ctx); return { render: (c) => [...H(c, 'my models'), line(K('sign in'))], path: '/library' }; }
  const { models } = await api.get('/me/library');
  const where = await Promise.all(models.map((m) => places(api, refOf(m)).catch(() => null)));
  ctx.remember(models);
  ctx.st.list = models.map((m) => ({ kind: 'library-model', id: refOf(m), name: m.name }));
  return { path: '/library', sugg: ['public models', 'cloud', 'machines'], render: (c) => [
    ...H(c, heading), blank(),
    ...models.flatMap((m, i) => [line(Hi(String(i + 1) + '. '), K(ascii(m.name), `where ${refOf(m)}`), Dm(`  ${role(m)}`)),
      ...(where[i] ? summary(where[i]) : [line(Dm('   where it is could not be read'))]),
      line('   ', K('talk', `talk to ${refOf(m)}`), '  ', K('where', `where ${refOf(m)}`)), blank()]),
    ...(!models.length ? [line(Dm('empty.'))] : []), line(K('public models')),
  ] };
}

export async function libraryCommand(ctx, raw, api) {
  let m; const t=raw.trim();
  if (/^(my models|library)$/i.test(t)) return libraryPage(ctx,api);
  if (!/^(library |where(?:\s|$)|move |add(?:\s|$)|publish(?:\s|$)|delete(?:\s|$)|remove(?:\s|$)|load(?:\s|$)|unload(?:\s|$)|rename(?:\s|$)|share(?:\s|$)|unshare(?:\s|$)|warm(?:\s|$))/i.test(t) || /^rename\s+(chat|file)\s/i.test(t)) return null;
  if (!ctx.user()) return { render: () => [line(K('sign in'))] };
  // places (Nov 20): remove an older copy, load where it's stored (moving it there first), unload
  // /warm (Nov 25, James: no waiting for start-up): hours when its machine is already up. it starts itself a little
  // before, and inside them it neither idles out nor hits the cap; outside them the usual rules
  if ((m=t.match(/^warm(?:\s+(.+?))?(?:\s+(off|\d{1,2}\s*-\s*\d{1,2}))?$/i))) {
    const hours=(m[2]||'').replace(/\s/g,'');
    const set=async(ref,h)=>{
      const tz=(()=>{try{return Intl.DateTimeFormat().resolvedOptions().timeZone||'UTC';}catch(_){return 'UTC';}})();
      const [a,b]=h.split('-').map(Number);
      let r; try { r=await api.post(`/me/models/${encodeURIComponent(ref)}/warm`, h.toLowerCase()==='off'?{off:true}:{from:a,to:b,tz}); }
      catch (e) { return {render:()=>[line(Am(ascii(e.message)))]}; }
      return {render:()=>[r.warm?line(Hi(ascii(r.model)),Dm(`  warm ${r.warm.from}-${r.warm.to} (${ascii(r.warm.tz)})`)):line(Hi(ascii(r.model)),Dm('  not kept warm'))]};
    };
    const ask=(ref)=>{ ctx.ask(`warm hours for ${ascii(ref)}? (9-17, or off) > `,async(a)=>{ const h=a.trim(); if (!h) return; const r=await set(ref,h); ctx.print(r.render); },['9-17','off']); return {render:()=>[]}; };
    if (!m[1]) return pickModel(ctx, api, 'keep warm', async (x) => ask(x));
    return hours ? set(m[1].trim(), hours) : ask(m[1].trim());
  }
  // v1 (Nov 25): /share and /rename ask for whatever they're missing, one question at a time
  if ((m=t.match(/^(share|unshare)(?:\s+(.+))?$/i)) && !/\s(with|from)\s+@?\S+/i.test(t)) {
    const verb=m[1].toLowerCase(), ref=(m[2]||'').trim();
    const whom=(x)=>{ ctx.ask(`${verb} ${ascii(x)} ${verb==='share'?'with':'from'} @`,async(a)=>{
      const who=a.trim().replace(/^@/,''); if (!who) return;
      const go=async(cmd)=>{ const r=await libraryCommand(ctx,cmd,api); if (r) ctx.print(r.render||r); };
      if (verb==='unshare') return go(`unshare ${x} from @${who}`);
      return go(`share ${x} with @${who}`);              // learning on, the point of sharing (Nov 26, James)
    },[]); return {render:()=>[]}; };
    return ref ? whom(ref) : pickModel(ctx, api, verb, async (x) => whom(x));
  }
  if ((m=t.match(/^rename(?:\s+(.+))?$/i)) && !/\sto\s/i.test(t)) {
    const named=(x)=>{ ctx.ask(`rename ${ascii(x)} to? > `,async(a)=>{ if (!a.trim()) return; const r=await libraryCommand(ctx,`rename ${x} to ${a.trim()}`,api); if (r) ctx.print(r.render||r); },[]); return {render:()=>[]}; };
    return m[1] ? named(m[1].trim()) : pickModel(ctx, api, 'rename', async (x) => named(x));
  }
  // learning is on for everyone a model is shared with (Nov 26, James: "the attractive part is the shared learning"),
  // and a toggle per person: /share <model> with @name learning off (or on). on someone already invited it only toggles
  if ((m=t.match(/^(share|unshare)\s+(.+?)\s+(?:with|from)\s+@?(\S+?)(?:\s+(learning\s+(?:on|off)|(?:to|as)\s+(?:teach|teacher)|(?:to\s+)?read(?:ing)?(?:\s+only)?))?$/i))) {
    const verb=m[1].toLowerCase(), ref=m[2].trim(), who=m[3], how=(m[4]||'').toLowerCase();
    const teach=!/off|read/.test(how);
    const toggle=(on,r)=>K(`learning ${on?'on':'off'}`,`share ${ref} with @${r.with} learning ${on?'on':'off'}`);
    const said=(r)=>[line(Hi(ascii(r.model)),Dm(verb==='share'?` is shared with @${ascii(r.with)}, learning ${r.teach?'on':'off'}. `:` is no longer shared with @${ascii(r.with)}.`)),
      ...(verb==='share'?[line(Dm(r.teach?'what they say and read teaches it, for everyone it is shared with. ':'they read it; nothing of theirs changes it. '),Dm('turn it '),toggle(!r.teach,r),Dm('.'))]:[]),
      line(K('shared models'))];
    const send=()=>api.post(`/me/models/${encodeURIComponent(ref)}/${verb}`,{username:who,teach});
    if (verb==='share' && how) {
      try { const r=await api.post(`/me/models/${encodeURIComponent(ref)}/share/learning`,{username:who,teach}); return {render:()=>said(r)}; }
      catch (e) { if (e.status!==404) return {render:()=>[line(Am(ascii(e.message)))]}; }   // not invited yet: share, below
    }
    if (verb==='unshare') {
      let r; try { r=await send(); } catch (e) { return {render:()=>[line(Am(ascii(e.message)))]}; }
      return {render:()=>said(r)};
    }
    // billing is yours, decisions are theirs (Nov 22, James): say so, and ask, before anything is shared
    ctx.ask(`share it with @${ascii(who)}? [y/n]> `,async(a)=>{
      if (!/^(y|yes)$/i.test(a)) return ctx.print(()=>[line(Dm('not shared.'))]);
      let r; try { r=await send(); } catch (e) { return ctx.print(()=>[line(Am(ascii(e.message)))]); }
      return ctx.print(()=>said(r));
    },['yes','no']);
    return {render:(c)=>[line(Am('before you share')),...P(c,Dm(`@${ascii(who)} can start and stop it on your runpod account. you pay, within its `),K('budget',`budget ${ref}`),
      Dm('. their files take space on its volume.'+(teach?' learning is on: what they say and read teaches the model, for everyone. only a backup undoes it. add learning off to the command to share it for reading.':' learning is off: they read it, and nothing of theirs changes it.'))),
      ...P(c,Dm('share with people you trust.'))]};
  }
  if ((m=t.match(/^rename\s+(.+?)\s+to\s+(.+)$/i))) {
    const ref=m[1].trim(), name=m[2].trim().replace(/^["']|["']$/g,'');
    let r;
    try { r=await api.post(`/me/models/${encodeURIComponent(ref)}/rename`,{name}); }
    catch (e) {
      // not in your library: one of your publications, by its id
      if (!/^[0-9a-f]{32}$/i.test(ref)) return {render:()=>[line(Am(ascii(e.message)))]};
      try { r=await api.post(`/me/publications/${ref}/rename`,{name}); } catch (e2) { return {render:()=>[line(Am(ascii(e2.message)))]}; }
    }
    return {render:()=>[line(Dm(r.was?`${ascii(r.was)} is now `:'renamed to '),Hi(ascii(r.name))),line(K('where',`where ${ref}`))]};
  }
  if ((m=t.match(/^unload\s*(.*)$/i))) {
    const ref=m[1].trim()||ctx.st.here?.id; if (!ref) return pickModel(ctx, api, 'unload', async (x) => api.unloadOn(ctx, await places(api, x)));
    return api.unloadOn(ctx, await places(api, ref));
  }
  if ((m=t.match(/^load\s+(.+?)(?:\s+on\s+(.+))?$/i))) return loadCmd(ctx, api, m[1].trim(), m[2] || '', api.handlers);
  // /delete (v1, Nov 25, James): one verb, said with what it deletes. a model goes from everywhere it is stored
  if (/^delete$/i.test(t)) return {render:()=>[line(K('delete model','delete model'),S('   '),K('delete file','delete file '),S('   '),K('delete chat','delete chat '),S('   '),K('history clear'),S('   '),K('close account'))]};
  if ((m=t.match(/^delete\s+model(?:\s+(.+))?$/i))) {
    const ref=(m[1]||'').trim()||(ctx.st.here?.kind==='library-model'?ctx.st.here.id:'');
    return ref ? deleteModel(ctx, api, ref) : pickModel(ctx, api, 'delete', (x) => deleteModel(ctx, api, x));
  }
  if ((m=t.match(/^delete\s+(?!file\b|chat\b|account\b|my\s+account\b)(.+)$/i))) return {render:()=>[line(K(`delete model ${m[1].trim()}`))]};
  if ((m=t.match(/^add\s+(.+)$/i))) {
    const id=await api.findModel(ctx,m[1]);
    if (!id) return {render:()=>[line(Dm('open a public model first.'))]};
    const r=await api.post(`/me/library/${id}`);
    ctx.remember([r.model]);
    return {render:(c)=>[...H(c,'added to my models'),line(Hi(ascii(r.model.name))),
      line(Dm(r.first_save_forks?'its first save makes a new private model.':'already in your library.')),
      line(K('my models'),'  ',K('locations / move',`where ${refOf(r.model)}`))]};
  }
  if ((m=t.match(/^(library|where)\s*(.*)$/i))) {
    const ref=m[2].trim() || ctx.st.here?.id;
    if (!ref) return {render:()=>[line(K('my models'),Dm(' to choose a model.'))]};
    return placesPage(ctx, api, ref);
  }
  // /move (v1): /move asks which model, /move <model> asks where; /move <model> to <place> just goes
  if (/^move$/i.test(t)) return pickModel(ctx, api, 'move', (x) => moveWhere(ctx, api, x));
  if ((m=t.match(/^move\s+(.+?)\s+to\s*(.*)$/i))) return m[2].trim() ? moveCmd(ctx, api, m[1].trim(), m[2].trim()) : moveWhere(ctx, api, m[1].trim());
  if ((m=t.match(/^move\s+(.+)$/i))) return moveWhere(ctx, api, m[1].trim());
  if ((m=t.match(/^publish\s*(.*)$/i))) {
    const ref=m[1].trim()||ctx.st.here?.id;
    if (!ref) return pickModel(ctx, api, 'publish', (x) => libraryCommand(ctx, `publish ${x}`, api));
    const {model}=await api.get(`/me/library/${encodeURIComponent(ref)}`);
    ctx.ask('licence [cc-by-4.0, mit, apache-2.0, cc0-1.0, cancel]> ',async(licence)=>{
      if (!['cc-by-4.0','mit','apache-2.0','cc0-1.0'].includes(licence)) return;
      // a publication is its own model, so it has its own name: the working model's name and today, unless you choose one
      const day=new Date().toLocaleDateString('en-CA');
      const suggested=`${model.name} ${day}`;
      ctx.ask(`name for the publication [${ascii(suggested)}]> `,async(given)=>{
      const pubName=(given||'').trim()||suggested;
      const away=model.lives_on?.volume_id&&!model.lives_on?.cached;
      ctx.ask(away?`publish from its volume in ${ascii(model.lives_on.data_center)}: a cpu machine there uploads it (at most $0.20/h)? [y/n]> `:'publish a separate frozen snapshot under that licence? [y/n]> ',async(a)=>{
        if (!yes(a)) return;
        let r;
        try { r=await api.post(`/me/models/${model.checkpoint_id}/publish`,{licence,name:pubName,...(away?{max_rate:0.2}:{})}); }
        catch (e) { return ctx.print(()=>[line(Am(ascii(e.message)))]); }
        if (r.publishing) {
          await ctx.print(()=>[line(Hi(`uploading from its home at $${Number(r.offer.price).toFixed(2)}/h; it lists once logOS has read the upload.`)),line(K('open publication',`open ${r.publication_id}`))]);
          return watch(ctx,r.machine,api);
        }
        return ctx.print(()=>[line(Hi(r.already_published?'this revision already has a publication.':`published as ${ascii(r.name||pubName)}; ${ascii(model.name)} stays private.`)),line(K('open publication',`open ${r.checkpoint_id}`))]);
      },['yes','no']);
      },[suggested]);
    },['cc-by-4.0','mit','apache-2.0','cc0-1.0','cancel']);
    return {render:()=>[line(Dm(`publish ${ascii(model.name)}`))]};
  }
  return {render:()=>[line(K('my models'))]};
}

// v1 (Nov 25, James): a command missing its model asks for one, offering yours
async function pickModel(ctx, api, verb, then) {
  let names = []; try { names = ((await api.get('/me/library')).models || []).map((x) => x.name).filter(Boolean).slice(0, 8); } catch (_) { /* */ }
  ctx.ask(`${verb} which? > `, async (a) => {
    const x = a.trim(); if (!x) return;
    try { const r = await then(x); if (r) ctx.print(r.render || r); } catch (e) { ctx.print(() => [line(Am(ascii(e.message)))]); }
  }, names);
  return { render: () => [] };
}

// /move <model>: where to, from the places there are
async function moveWhere(ctx, api, ref) {
  const ps = await places(api, ref);
  const dcs = [...new Set(ps.places.filter((x) => x.kind === 'volume').map((x) => x.data_center))];
  const options = ['logos', 'this machine', ...ps.places.filter((x) => x.kind === 'computer').map((x) => where(x)), ...dcs.map((d) => `runpod ${d}`), 'runpod '];
  ctx.ask(`move ${ascii(ps.model.name)} to? > `, async (a) => {
    const w = a.trim(); if (!w) return;
    try { const r = await moveCmd(ctx, api, ref, w); if (r) ctx.print(r.render || r); } catch (e) { ctx.print(() => [line(Am(ascii(e.message)))]); }
  }, [...new Set(options)].slice(0, 8));
  return { render: () => [] };
}

// /delete model <model>: every copy, wherever it's stored, and logOS's record. not while it's loaded or moving
async function deleteModel(ctx, api, ref) {
  const ps = await places(api, ref); const model = ps.model;
  if (ps.loaded || ps.moving) return { render: () => [line(Am(`${ascii(model.name)} is ${ps.loaded ? 'loaded' : 'moving'}. `), K('unload', `unload ${refOf(model)}`), Dm(' it first.'))] };
  if (model.reference) {
    ctx.ask(`take ${ascii(model.name)} out of your models? the publication stays. [y/n]> `, async (a) => {
      if (!yes(a)) return ctx.print(() => [line(Dm('kept.'))]);
      await api.request(`/me/library/${encodeURIComponent(refOf(model))}`, 'DELETE');
      return ctx.print(() => [line(Dm('gone. ')), K('my models')]);
    }, ['yes', 'no']);
    return { render: () => [] };
  }
  const copies = ps.places.filter((x) => x.stored && x.kind !== 'logos');
  ctx.ask(`type ${ascii(model.name)} to delete it everywhere> `, async (a) => {
    if (a.trim() !== String(model.name).trim()) return ctx.print(() => [line(Dm('kept.'))]);
    const said = [];
    for (const x of copies) {
      try { await api.post(`/me/models/${encodeURIComponent(refOf(model))}/remove`, { place: x.id, max_rate: 0.2, all: true }); said.push(line(Dm(`  ${where(x)}: deleting`))); }
      catch (e) { said.push(line(Am(`  ${where(x)}: ${ascii(e.message)}`))); }
    }
    try { await api.request(`/me/library/${encodeURIComponent(refOf(model))}`, 'DELETE'); } catch (e) { return ctx.print(() => [...said, line(Am(ascii(e.message)))]); }
    return ctx.print(() => [line(Hi(`${ascii(model.name)} deleted.`)), ...said]);
  }, []);
  return { render: (c) => [...H(c, `delete ${model.name}`), line(Am('every copy, every revision. it cannot be undone.')),
    ...copies.map((x) => line(Dm(`  ${where(x)}${x.kind === 'volume' ? ' (a $0.12/h cpu machine on your runpod deletes it)' : ''}`)))] };
}

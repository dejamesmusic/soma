import { libraryCommand, libraryPage } from './library';
const model={checkpoint_id:'a'.repeat(32),name:'mine',visibility:'private',head_revision_id:'r1',file_size_bytes:1000};
const where={primary:'runpod · EU-RO-1',copies:[{kind:'volume',label:'runpod · EU-RO-1',data_center:'EU-RO-1',revision_id:'r1'}],warnings:[],machines:[],macs:[],workspace:{head_revision_id:'r1'}};
function setup(){
 const ctx={user:()=>({username:'ada'}),st:{},remember:jest.fn(),ask:jest.fn(),print:jest.fn().mockResolvedValue(),suggest:jest.fn()};
 const api={get:jest.fn().mockResolvedValue({model,locations:where}),post:jest.fn().mockResolvedValue({checkpoint_id:'p'}),go:jest.fn()};
 return {ctx,api};
}
// places (Nov 20): one newest copy; moves ask once for the whole route; loads go where it's stored
const ps = (over = {}) => ({ model: { name: 'mine', checkpoint_id: model.checkpoint_id, ref: model.checkpoint_id, bytes: 3e9, head_revision_id: 'r1', reference: false },
  home: { kind: 'volume', data_center: 'EU-RO-1' }, loaded: null, moving: [], warnings: [],
  places: [{ kind: 'logos', id: 'logos', name: 'logOS', stored: false, newest: false },
    { kind: 'volume', id: 'volume:EU-RO-1', name: 'runpod EU-RO-1', data_center: 'EU-RO-1', stored: true, newest: true, revision_id: 'r1' },
    { kind: 'computer', id: 'computer:app_1', token_id: 'app_1', name: 'soma on ada', online: true, engine: true, stored: false, newest: false }], ...over });
test('moving a model to where its newest copy already is asks nothing and moves nothing', async () => {
  const { ctx, api } = setup(); api.get.mockResolvedValue(ps());
  await libraryCommand(ctx, `move ${model.checkpoint_id} to runpod EU-RO-1`, api);
  expect(api.post).not.toHaveBeenCalled(); expect(ctx.ask).not.toHaveBeenCalled();
});
test('a move between a volume and a computer is two hops through logOS, confirmed once', async () => {
  const { ctx, api } = setup(); api.get.mockResolvedValue(ps());
  const page = await libraryCommand(ctx, `move ${model.checkpoint_id} to ada`, api);
  const text = page.render(80).map((l) => l.map((g) => g.t).join('')).join('\n');
  expect(text).toMatch(/runpod EU-RO-1 to logOS/); expect(text).toMatch(/logOS to ada/); expect(text).toMatch(/deleted/);
  expect(ctx.ask).toHaveBeenCalledTimes(1); expect(api.post).not.toHaveBeenCalled();
  await ctx.ask.mock.calls[0][1]('no'); expect(api.post).not.toHaveBeenCalled();
});
test('a gpu loads it straight from its volume, without a move', async () => {
  const { ctx, api } = setup(); api.get.mockResolvedValue(ps()); api.handlers = { gpu: jest.fn().mockResolvedValue({ render: () => [] }) };
  await libraryCommand(ctx, `load ${model.checkpoint_id} on gpu`, api);
  expect(api.handlers.gpu).toHaveBeenCalled(); expect(api.post).not.toHaveBeenCalled();
});
test('the newest copy is never removed', async () => {
  const { ctx, api } = setup(); api.get.mockResolvedValue(ps());
  await libraryCommand(ctx, `remove ${model.checkpoint_id} from volume:EU-RO-1`, api);
  expect(ctx.ask).not.toHaveBeenCalled(); expect(api.post).not.toHaveBeenCalled();
});

test('a move-and-load chain quotes CPU and GPU before one confirmation',async()=>{
  const {ctx,api}=setup(); const {loadCmd}=require('./places');
  const state=ps({home:{kind:'logos'},places:[{kind:'logos',id:'logos',name:'logOS',stored:true,newest:true}]});
  api.get.mockImplementation(async path=>path.includes('/cloud/offers')?{offers:[{id:'gpu',name:'GPU',price:0.7}],storage:{monthly_usd:2}}:path.includes('move-quote')?{legs:[{rate:0.14,data_center:'US-GA-2'}]}:state);
  const handlers={gpu:jest.fn()}; const page=await loadCmd(ctx,api,state.model.ref,'gpu in US-GA-2',handlers);
  const text=page.render(80).flat().map(x=>x.t).join('');
  expect(text).toContain('$0.14/h'); expect(text).toContain('$0.70/h');
  expect(ctx.ask).toHaveBeenCalledTimes(1); expect(api.post).not.toHaveBeenCalled();
  await ctx.ask.mock.calls[0][1]('no'); expect(api.post).not.toHaveBeenCalled(); expect(handlers.gpu).not.toHaveBeenCalled();
});
test('publishing requires both a licence and explicit snapshot confirmation',async()=>{
 const {ctx,api}=setup(); await libraryCommand(ctx,`publish ${model.checkpoint_id}`,api);
 expect(api.post).not.toHaveBeenCalled(); await ctx.ask.mock.calls[0][1]('mit');
 expect(api.post).not.toHaveBeenCalled(); await ctx.ask.mock.calls[1][1]('no');
 expect(api.post).not.toHaveBeenCalled();
});
test('the private library is fetched independently of the public square',async()=>{
 const {ctx,api}=setup(); api.get.mockImplementation(async(path)=>path==='/me/library'?{models:[model]}:where);
 const r=await libraryPage(ctx,api);
 expect(r.path).toBe('/library'); expect(api.get).not.toHaveBeenCalledWith('/checkpoints');
 expect(ctx.st.list[0].kind).toBe('library-model');
});

test('/delete model deletes every copy, wherever it is stored, after its name is typed (v1)', async () => {
  const ctx = { user: () => ({ username: 'ada' }), st: {}, ask: jest.fn(), suggest: jest.fn(), print: jest.fn().mockResolvedValue() };
  const ps = { model: { checkpoint_id: 'c1', name: 'soma-1b' }, loaded: false, moving: false,
    places: [{ id: 'logos', kind: 'logos', stored: true }, { id: 'vol1', kind: 'volume', stored: true, data_center: 'US-GA-2' }] };
  const api = { get: jest.fn().mockResolvedValue(ps), post: jest.fn().mockResolvedValue({}), request: jest.fn().mockResolvedValue({ deleted: 'c1' }) };
  const { libraryCommand } = require('./library');
  const pointer = await libraryCommand(ctx, 'delete soma-1b', api);
  expect(pointer.render(80).map((l) => l.map((g) => g.t).join('')).join('\n')).toContain('/delete model soma-1b');
  const page = await libraryCommand(ctx, 'delete model soma-1b', api);
  expect(page.render(80).map((l) => l.map((g) => g.t).join('')).join('\n')).toMatch(/runpod US-GA-2.*cpu machine/);
  await ctx.ask.mock.calls[0][1]('soma');
  expect(api.request).not.toHaveBeenCalled();
  await ctx.ask.mock.calls[0][1]('soma-1b');
  expect(api.post).toHaveBeenCalledWith('/me/models/c1/remove', { place: 'vol1', max_rate: 0.2, all: true });
  expect(api.request).toHaveBeenCalledWith('/me/library/c1', 'DELETE');
  api.get.mockResolvedValue({ ...ps, loaded: true });
  const refused = await libraryCommand(ctx, 'delete model soma-1b', api);
  expect(refused.render(80).map((l) => l.map((g) => g.t).join('')).join('\n')).toContain('/unload');
});

test('/warm sets hours in this timezone, asks when they are missing, and turns off (Nov 25)', async () => {
  const ctx = { user: () => ({ username: 'ada' }), st: {}, ask: jest.fn(), suggest: jest.fn(), print: jest.fn().mockResolvedValue() };
  const api = { post: jest.fn().mockResolvedValue({ model: 'soma-1b', warm: { from: 9, to: 17, tz: 'UTC' } }), get: jest.fn() };
  const { libraryCommand } = require('./library');
  const page = await libraryCommand(ctx, 'warm soma-1b 9-17', api);
  expect(api.post).toHaveBeenCalledWith('/me/models/soma-1b/warm', expect.objectContaining({ from: 9, to: 17 }));
  expect(page.render(80).map((l) => l.map((g) => g.t).join('')).join('')).toContain('warm 9-17');
  await libraryCommand(ctx, 'warm soma-1b', api);
  expect(ctx.ask).toHaveBeenCalled();
  await libraryCommand(ctx, 'warm soma-1b off', api);
  expect(api.post).toHaveBeenLastCalledWith('/me/models/soma-1b/warm', { off: true });
});

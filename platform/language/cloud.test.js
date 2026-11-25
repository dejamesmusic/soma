import { runCloud } from './cloud';

function setup() {
  const ctx = { user: () => ({ username: 'ada' }), st: {}, active: () => false,
    ask: jest.fn(), suggest: jest.fn(), print: jest.fn().mockResolvedValue(),
    printLive: jest.fn(), update: jest.fn() };
  const quote = { offers: [{ id: 'gpu', name: 'gpu', memory: 24, price: 0.27, data_centers: ['EU-RO-1'] }],
    model: { checkpoint_id: 'ck_one' }, capacity: { gpu_gb: 8, disk_gb: 30 },
    storage: { gb: 30, kind: 'pod disk', monthly_usd: 3, stopped_monthly_usd: 6 }, idle_minutes: 20, cap_hours: 8 };
  const api = { get: jest.fn().mockResolvedValue(quote), request: jest.fn().mockResolvedValue({}),
    post: jest.fn().mockResolvedValue({ machine: { machine_id: 'mch_ab', model: 'one' } }) };
  return { ctx, api };
}

test('quoting a launch allocates nothing and declining keeps it that way', async () => {
  const { ctx, api } = setup();
  await runCloud(ctx, 'cloud start one', api);
  expect(api.post).not.toHaveBeenCalled();
  await ctx.ask.mock.calls[0][1]('no');
  expect(api.post).not.toHaveBeenCalled();
});

test('confirmation pins the quoted price, location and model identity', async () => {
  const { ctx, api } = setup();
  await runCloud(ctx, 'cloud start one', api);
  await ctx.ask.mock.calls[0][1]('yes');
  expect(api.post).toHaveBeenCalledWith('/me/cloud/machines', expect.objectContaining({
    model: 'ck_one', gpu: 'gpu', data_center: 'EU-RO-1', max_rate: 0.27,
  }));
});

test('provider linking requests hidden input and allocates no compute', async () => {
  const { ctx, api } = setup();
  await runCloud(ctx, 'runpod key', api);
  expect(ctx.ask.mock.calls[0][3]).toEqual({ secret: true });
  await ctx.ask.mock.calls[0][1]('rpa_test_private');
  expect(api.request).toHaveBeenCalledWith('/me/runpod/key', 'PUT', { key: 'rpa_test_private' });
  expect(api.post).not.toHaveBeenCalled();
});

test('an approved move-and-load GPU stage does not ask again',async()=>{
  const {ctx,api}=setup(); const quote=await api.get();
  api.approved={quote,offer:quote.offers[0]}; api.data_center='EU-RO-1';
  await runCloud(ctx,'cloud start one',api);
  expect(ctx.ask).not.toHaveBeenCalled();
  expect(api.post).toHaveBeenCalledWith('/me/cloud/machines',expect.objectContaining({gpu:'gpu',max_rate:0.27}));
});

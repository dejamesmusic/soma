import { uploadPage } from './upload';
import { adminCommand } from './admin';

const puts = [];
class FakeXHR {
  constructor() { this.upload = {}; }
  open(method, url) { this.url = url; }
  setRequestHeader() {}
  send(body) { puts.push({ url: this.url, size: body.size }); this.status = 200; setTimeout(() => this.onload(), 0); }
}

function setup(answers) {
  const ctx = { user: () => ({ username: 'ada' }), st: {}, ask: jest.fn(), suggest: jest.fn(), print: jest.fn().mockResolvedValue(),
    printLive: jest.fn(() => ({})), update: jest.fn(), remember: jest.fn() };
  const api = { get: jest.fn(), request: jest.fn(), post: jest.fn((path, body) => Promise.resolve(answers(path, body))) };
  return { ctx, api };
}

beforeEach(() => { puts.length = 0; global.XMLHttpRequest = FakeXHR; });

test('a large file goes up in parts, R2 lists them, and it registers as a private model under the name given', async () => {
  const MB = 1024 ** 2; const size = 4 * 1024 ** 3 + 300 * MB;        // just past one PUT: 35 parts of 128 MiB
  const { ctx, api } = setup((path, body) => ({
    '/checkpoints/upload/multipart': { r2_key: 'uploads/pending/abc/model.pt', upload_id: 'u1', part_bytes: 128 * MB, parts: Math.ceil(size / (128 * MB)) },
    '/checkpoints/upload/multipart/parts': { urls: Object.fromEntries((body?.part_numbers || []).map((n) => [n, `https://r2/part/${n}`])) },
    '/checkpoints/upload/multipart/complete': { bytes: size },
    '/checkpoints/upload/confirm': { parsed: { checkpoint_id: 'c1', revision_id: 'r1', name: 'file' }, profile: null, file_size_bytes: size, r2_key: 'uploads/pending/abc/model.pt' },
    '/checkpoints': { checkpoint_id: 'c1', name: body?.name },
  })[path]);
  uploadPage(ctx, api);
  const file = { name: 'soma.pt', size, slice: (a, b) => ({ size: b - a }) };
  await ctx.st.pick([file]);
  expect(puts).toHaveLength(35);
  expect(puts.reduce((a, p) => a + p.size, 0)).toBe(size);
  const complete = api.post.mock.calls.find(([p]) => p.endsWith('/complete'))[1];
  expect(complete.parts).toBeUndefined();
  expect(complete.file_size).toBe(size);
  const [q, answer] = ctx.ask.mock.calls[0];
  expect(q).toMatch(/^name \[file\]/);
  await answer('my soma');
  const made = api.post.mock.calls.find(([p]) => p === '/checkpoints')[1];
  expect(made).toMatchObject({ name: 'my soma', visibility: 'private', revision_id: 'r1', r2_key: 'uploads/pending/abc/model.pt' });
});

test('admin answers only an admin', async () => {
  const { ctx, api } = setup(() => ({}));
  const r = await adminCommand(ctx, 'admin', api);
  expect(api.get).not.toHaveBeenCalled();
  expect(r.render(80)[0][0].t).toMatch(/for admins/);
});

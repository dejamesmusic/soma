import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import Terminal from './Terminal';

jest.mock('react-router-dom', () => ({ useNavigate: () => jest.fn() }), { virtual: true });
jest.mock('../contexts/AuthContext', () => ({ useAuth: () => ({ loading: false, user: { username: 'ada' } }) }));
let root, host;
const key = 'rpa_this_is_a_private_key_123456789';
const wait = (ms = 20) => new Promise((resolve) => setTimeout(resolve, ms));
async function submit(text) {
  await act(async () => {
    const input = host.querySelector('textarea'); input.value = text;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await wait();
  });
}
beforeEach(async () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  window.matchMedia = () => ({ matches: true });
  HTMLCanvasElement.prototype.getContext = jest.fn(() => null);
  global.ResizeObserver = class { observe() {} disconnect() {} };
  global.fetch = jest.fn().mockResolvedValue({ ok: true, json: async () => ({ linked: true }) });
  sessionStorage.clear(); window.history.replaceState({}, '', '/');
  host = document.createElement('div'); document.body.appendChild(host); root = createRoot(host);
  await act(async () => { root.render(<Terminal />); await wait(); });
});
afterEach(async () => { await act(async () => root.unmount()); host.remove(); });

test('hidden input is masked and absent from history and persisted scrollback', async () => {
  await submit('/runpod key');
  await act(async () => {
    const input = host.querySelector('textarea'); input.value = key;
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
  expect(host.textContent).not.toContain(key);
  expect(host.querySelector('.tm-typed').textContent).toMatch(/^\*+$/);
  await submit(key);
  await act(async () => { await wait(450); });
  expect(JSON.stringify(sessionStorage)).not.toContain(key);
  expect(host.textContent).not.toContain(key);
  expect(fetch).toHaveBeenCalledWith('/api/me/runpod/key', expect.objectContaining({ method: 'PUT', body: JSON.stringify({ key }) }));
  await act(async () => host.querySelector('textarea').dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true })));
  expect(host.querySelector('textarea').value).toBe('/runpod key');
});

test('an inline key never reaches history, echo, search or the network', async () => {
  await submit(`/runpod key ${key}`);
  await act(async () => { await wait(450); });
  // Notice polling is independent of a rejected command. The key must never be sent anywhere.
  expect(fetch.mock.calls.every(([url]) => url === '/api/me/notices?limit=50')).toBe(true);
  expect(JSON.stringify(fetch.mock.calls)).not.toContain(key);
  expect(host.textContent).not.toContain(key);
  expect(JSON.stringify(sessionStorage)).not.toContain(key);
});

test('a library link executes the entire command with reduced motion enabled', async () => {
  fetch.mockResolvedValue({ ok: true, json: async () => ({ models: [] }) });
  await submit('/home');
  await act(async () => {
    host.querySelector('a[data-cmd="my models"]').dispatchEvent(new MouseEvent('click', { bubbles: true, button: 0 }));
    await wait();
  });
  expect(fetch).toHaveBeenCalledWith('/api/me/library', expect.anything());
  expect(host.textContent).toContain('MY MODELS');
});

test('a fetched page shows its images where they appear; videos play on a click (1.5.2)', async () => {
  const history = { chat: { chat_id: 'ch_1', model: 'soma-1b', name: 'moss', sessions: 1 }, events: [
    { kind: 'user', text: 'show me moss', data: {} },
    { kind: 'skill', text: 'https://en.wikipedia.org/wiki/Moss', data: { kind: 'fetch', ok: true, bytes: 900,
      excerpt: 'Moss carpets the floor.\n[image: A moss carpet](https://upload.example.org/carpet.jpg)\nAnd a film:\n[video: Moss](https://www.youtube.com/embed/abc123xyz)',
      media: [{ kind: 'image', src: 'https://upload.example.org/carpet.jpg', alt: 'A moss carpet' }, { kind: 'video', src: 'https://www.youtube.com/embed/abc123xyz', alt: 'Moss' }] } },
    { kind: 'model', text: 'here it is: https://upload.example.org/carpet.jpg', data: { media: [{ kind: 'image', src: 'https://upload.example.org/carpet.jpg', alt: 'A moss carpet' }] } },
  ] };
  global.fetch = jest.fn().mockImplementation(async (url) => ({ ok: true, json: async () => (String(url).includes('/history') ? history : {}) }));
  await submit('/chat history ch_1');
  await act(async () => { await wait(450); });
  const imgs = [...host.querySelectorAll('img.tm-img')];
  expect(imgs.length).toBe(2);
  expect(imgs[0].getAttribute('src')).toBe('https://upload.example.org/carpet.jpg');
  expect(imgs[0].referrerPolicy).toBe('no-referrer');
  expect(host.textContent).toContain('Moss carpets the floor.');
  expect(host.querySelector('iframe')).toBeNull();
  const play = [...host.querySelectorAll('a.tm-lk')].find((a) => a.textContent.includes('play video'));
  expect(play).toBeTruthy();
  await act(async () => { play.click(); await wait(); });
  expect(host.querySelector('iframe.tm-vid').getAttribute('src')).toBe('https://www.youtube-nocookie.com/embed/abc123xyz');
});

test('the start is blank, words without a slash are only written, and commands show their slash', async () => {
  expect(host.querySelector('.tm-scroll').textContent.trim()).toBe('');
  await submit('hello there');
  expect(host.querySelector('.tm-scroll').textContent).toContain('hello there');
  expect(fetch.mock.calls.some(([url]) => String(url).includes('/checkpoints'))).toBe(false);
  await submit('/help');
  expect([...host.querySelectorAll('a.tm-lk')].some((a) => a.textContent.startsWith('/'))).toBe(true);
});

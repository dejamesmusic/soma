// a file from this machine to a cloud machine's data folder (Nov 21): up to logOS's storage (one PUT, or 128 MiB parts
// four at a time), then logOS tells the machine to fetch it; the stored copy is deleted once the machine has it
import { S, Dm, Hi, Am, line, ascii, fmtBytes, progress } from './text';

function put(url, body, onProgress) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded); };
    x.onload = () => (x.status >= 200 && x.status < 300 ? resolve() : reject(new Error(`storage refused the upload (${x.status})`)));
    x.onerror = () => reject(new Error('the connection dropped during the upload'));
    x.open('PUT', url); x.send(body);
  });
}

export async function sendToMachine(ctx, api, token, file, statusOf = null) {
  const b = ctx.printLive(() => [line(Dm('sending '), Hi(ascii(file.name)), Dm(` ${fmtBytes(file.size)} ...`))]);
  let done = 0; let last = 0;
  const show = (stage) => ctx.update(b, (c) => [line(Dm(`${stage} `), Hi(ascii(file.name))),
    line(S(progress(file.size ? done / file.size : 0, Math.max(10, Math.min(30, c - 30)))), Dm(`  ${fmtBytes(done)} of ${fmtBytes(file.size)}`))]);
  const tick = (n) => { done = n; if (Date.now() - last > 300) { last = Date.now(); show('sending'); } };
  try {
    const t = await api.post('/me/transfers', { name: file.name, size: file.size });
    if (t.put_url) await put(t.put_url, file, tick);
    else {
      const sent = new Map(); const urls = {}; let next = 1;
      const worker = async () => {
        while (next <= t.parts) {
          const n = next; next += 1;
          if (!urls[n]) Object.assign(urls, (await api.post(`/me/transfers/${t.transfer_id}/parts`, { part_numbers: Array.from({ length: Math.min(64, t.parts - n + 1) }, (_, i) => n + i) })).urls);
          const blob = file.slice((n - 1) * t.part_bytes, Math.min(file.size, n * t.part_bytes));
          await put(urls[n], blob, (x) => { sent.set(n, x); tick([...sent.values()].reduce((a, v) => a + v, 0)); });
          sent.set(n, blob.size);
        }
      };
      await Promise.all([worker(), worker(), worker(), worker()]);
    }
    done = file.size; show('the machine is fetching');
    const r = await api.post(`/me/transfers/${t.transfer_id}/send`, { token_id: token });
    // the machine answers the fetch: done once the file is in its data folder, or failed with the reason
    for (let i = 0; i < 1800; i += 1) {
      await new Promise((z) => setTimeout(z, 1000));
      // a shared model's guest asks after their own command; the owner reads the machine's log
      const c = statusOf ? await statusOf(r.command.command_id).catch(() => null)
        : (await api.get(`/me/apps/${token}/commands?limit=20`).catch(() => null))?.commands?.find((x) => x.command_id === r.command.command_id);
      if (c?.status === 'done') { ctx.update(b, () => [line(Hi(ascii(r.name)), Dm(' is on the machine'))]); return r.name; }
      if (c?.status === 'failed' || c?.status === 'expired') throw new Error(c.result?.error || `the machine ${c.status} it`);
    }
    throw new Error('the machine never answered; /files shows whether it arrived');
  } catch (e) {
    ctx.update(b, () => [line(Am(`${ascii(file.name)}: ${ascii(e.message)}`))]);
    throw e;
  }
}

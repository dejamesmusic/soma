// upload, in the terminal (Nov 20): a model file from this browser becomes your private model. The bold link opens the
// file picker (a click is the gesture a browser needs); the file goes straight to R2, in one PUT up to 4 GiB and in
// 128 MiB parts above, four at a time. logOS reads it where it landed before anything is registered.
import { S, K, Dm, Hi, Am, line, blank, P, H, ascii, cut, fmtBytes, fmtDur, progress } from './text';

export const Pick = (t) => S(t, { pick: true });          // bold: opens the file picker
const SINGLE = 4 * 1024 ** 3;
const IN_FLIGHT = 4;

export function uploadPage(ctx, api) {
  ctx.st.pick = (files) => received(ctx, api, files).catch((e) => ctx.print(() => [line(Am(ascii(e.message)))]));
  ctx.st.pickAccept = '.pt,.json';
  return {
    path: '/upload', sugg: ['my models'],
    render: (c) => [...H(c, 'upload'),
      line(Pick('[ choose a file ]'), Dm('  .pt, and its .meta.json if it has one; up to 16 GB'))],
  };
}

function put(url, body, type, onProgress) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.upload.onprogress = (e) => { if (e.lengthComputable && onProgress) onProgress(e.loaded); };
    x.onload = () => (x.status >= 200 && x.status < 300 ? resolve() : reject(new Error(`storage refused the upload (${x.status})`)));
    x.onerror = () => reject(new Error('the connection dropped during the upload'));
    x.open('PUT', url);
    if (type) x.setRequestHeader('Content-Type', type);
    x.send(body);
  });
}

async function sendLarge(api, file, onProgress) {
  const up = await api.post('/checkpoints/upload/multipart', { filename: 'model.pt', file_size: file.size });
  const { r2_key: key, upload_id: uploadId, part_bytes: size, parts } = up;
  const sent = new Map(); const urls = {};
  const report = () => onProgress([...sent.values()].reduce((a, b) => a + b, 0));
  try {
    let next = 1;
    const worker = async () => {
      while (next <= parts) {
        const n = next; next += 1;
        if (!urls[n]) {
          const batch = Array.from({ length: Math.min(64, parts - n + 1) }, (_, i) => n + i).filter((k) => !urls[k]);
          Object.assign(urls, (await api.post('/checkpoints/upload/multipart/parts', { r2_key: key, upload_id: uploadId, part_numbers: batch })).urls);
        }
        const blob = file.slice((n - 1) * size, Math.min(file.size, n * size));
        for (let attempt = 1; ; attempt += 1) {
          try { await put(urls[n], blob, null, (b) => { sent.set(n, b); report(); }); break; }
          catch (e) { if (attempt >= 3) throw e; await new Promise((r) => setTimeout(r, 2000 * attempt)); }
        }
        sent.set(n, blob.size); report();
      }
    };
    await Promise.all(Array.from({ length: IN_FLIGHT }, worker));
    await api.post('/checkpoints/upload/multipart/complete', { r2_key: key, upload_id: uploadId, file_size: file.size });   // R2 lists the parts
  } catch (e) {
    api.post('/checkpoints/upload/multipart/abort', { r2_key: key, upload_id: uploadId }).catch(() => {});
    throw e;
  }
  return key;
}

async function received(ctx, api, files) {
  const pt = files.find((f) => /\.pt$/i.test(f.name));
  const meta = files.find((f) => /\.json$/i.test(f.name));
  if (!pt) return ctx.print(() => [line(Am('choose a .pt file (and, if you like, its .meta.json).'))]);
  if (pt.size > 16 * 1024 ** 3) return ctx.print(() => [line(Am(`${ascii(pt.name)} is ${fmtBytes(pt.size)}; the limit is 16 GB.`))]);
  ctx.st.pick = null;
  const began = Date.now(); let done = 0; let stage = 'uploading';
  const status = (c) => {
    const s = (Date.now() - began) / 1000; const rate = done / Math.max(1, s);
    return [line(Hi(ascii(pt.name)), Dm(`  ${fmtBytes(pt.size)}${meta ? ' + ' + ascii(meta.name) : ''}`)),
      line(S(progress(done / pt.size, Math.max(10, Math.min(30, c - 40)))), Dm(`  ${stage}`)),
      line(Dm(`${fmtBytes(done)} of ${fmtBytes(pt.size)}  |  ${fmtBytes(rate)}/s${done < pt.size && rate > 0 ? `  |  ${fmtDur((pt.size - done) / rate)} left` : ''}`))];
  };
  const b = ctx.printLive(status);
  let last = 0;
  const onProgress = (n) => { done = n; if (Date.now() - last > 250) { last = Date.now(); ctx.update(b, status); } };
  const stay = (e) => { e.preventDefault(); e.returnValue = ''; };
  window.addEventListener('beforeunload', stay);
  let key; let confirmed;
  try {
    if (pt.size <= SINGLE) {
      const slot = await api.post('/checkpoints/upload/presign', { filename: pt.name.replace(/[^\w.-]/g, '_'), file_size: pt.size });
      await put(slot.presigned_url, pt, 'application/octet-stream', onProgress);
      key = slot.r2_key;
    } else {
      key = await sendLarge(api, pt, onProgress);
    }
    done = pt.size;
    let metaKey = null;
    if (meta) {
      const slot = await api.post('/checkpoints/upload/presign_meta', { r2_key: key });
      await put(slot.presigned_url, meta, 'application/json');
      metaKey = slot.meta_r2_key;
    }
    stage = 'reading it'; ctx.update(b, status);
    confirmed = await api.post('/checkpoints/upload/confirm', { upload_id: key.split('/')[2], r2_key: key, filename: pt.name, meta_r2_key: metaKey });
    stage = 'uploaded'; ctx.update(b, status);
  } catch (e) {
    stage = 'failed'; ctx.update(b, status);
    throw e;
  } finally {
    window.removeEventListener('beforeunload', stay);
  }
  const { parsed, profile } = confirmed;
  const suggested = ascii(profile?.name || parsed.name || pt.name.replace(/\.pt$/i, '')).trim();
  ctx.ask(`name [${cut(suggested, 40)}]> `, async (answer) => {
    const name = ascii(answer).trim() || suggested;
    const made = await api.post('/checkpoints', {
      ...parsed, name, description: profile?.tagline || profile?.about || parsed.description || null,
      corpus_tags: parsed.corpus_tags || [], visibility: 'private',
      file_size_bytes: confirmed.file_size_bytes, r2_key: confirmed.r2_key, meta_r2_key: confirmed.meta_r2_key || null, profile: profile || null,
    });
    ctx.suggest([`library ${made.checkpoint_id}`, `publish ${made.checkpoint_id}`, 'my models']);
    return ctx.print((c) => [line(Hi(ascii(made.name)), Dm(' is your private model.')),
      line('  ', K('open it', `library ${made.checkpoint_id}`), '  ', K('publish a snapshot', `publish ${made.checkpoint_id}`), '  ', K('my models'))]);
  }, [suggested]);
  return null;
}

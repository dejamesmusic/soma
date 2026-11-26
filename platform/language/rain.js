// Rolling code (Nov 20, James): where the glass is wider than the terminal needs, past a margin, somatic core's own
// source falls in columns. Each drop is one line of the released runtime read top to bottom; heads are bright, trails
// fade. It stills with the monitor's motion knob or reduced motion, sleeps in a hidden tab, and draws nothing narrow.
const HEAD = '#fff6e0';
const TRAIL = '#b8a98a';

export function startRain(canvas, { still = () => false } = {}) {
  const ctx = canvas.getContext?.('2d');
  if (!ctx) return { resize: () => {}, stop: () => {}, settle: () => {} };          // no canvas (tests, old browsers)
  let lines = ['soma'];
  fetch('/rain.txt').then((r) => (r.ok ? r.text() : '')).then((t) => { const ls = t.split('\n').filter(Boolean); if (ls.length) lines = ls; }).catch(() => {});
  let cw = 8.4; let ch = 21; let cols = 0; let rows = 0; let drops = []; let timer = null; let font = '14px monospace';
  const pick = () => lines[Math.floor(Math.random() * lines.length)];
  const fresh = (x, staggered) => ({ x, y: staggered ? -Math.floor(Math.random() * rows * 1.5) : -Math.floor(Math.random() * rows * 0.5),
    text: pick(), i: 0, every: 1 + Math.floor(Math.random() * 3), t: 0 });
  const resize = (width, height, cssFont, charW, lineH) => {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(width * dpr)); canvas.height = Math.max(1, Math.floor(height * dpr));
    canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    font = cssFont; cw = charW; ch = lineH;
    cols = Math.floor(width / (cw * 1.5)); rows = Math.ceil(height / ch);          // a column every 1.5 chars: dense, still legible
    drops = Array.from({ length: cols }, (_, k) => (Math.random() < 0.7 ? fresh(k, true) : null));
    ctx.clearRect(0, 0, width, height);
  };
  const tick = () => {
    if (!cols || document.hidden) return;
    const w = canvas.width; const h = canvas.height;
    ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = 'destination-out'; ctx.fillStyle = 'rgba(0,0,0,0.05)'; ctx.fillRect(0, 0, w, h);
    ctx.restore();
    ctx.globalCompositeOperation = 'source-over'; ctx.font = font; ctx.textBaseline = 'top';
    for (let k = 0; k < cols; k += 1) {
      let d = drops[k];
      if (!d) { if (Math.random() < 0.02) drops[k] = fresh(k, false); continue; }
      d.t += 1; if (d.t % d.every) continue;
      const px = Math.round(d.x * cw * 1.5);
      if (d.y >= 0 && d.y < rows) {
        if (d.i > 0) { ctx.fillStyle = TRAIL; ctx.fillText(d.text[d.i - 1] || ' ', px, (d.y - 1) * ch); }
        const c = d.text[d.i]; if (c && c !== ' ') { ctx.fillStyle = HEAD; ctx.fillText(c, px, d.y * ch); }
      }
      d.y += 1; d.i += 1;
      if (d.i > d.text.length || d.y > rows + 2) drops[k] = Math.random() < 0.7 ? fresh(k, false) : null;
    }
  };
  const loop = () => { if (!still()) tick(); };
  timer = setInterval(loop, 70);
  return { resize, stop: () => clearInterval(timer), settle: () => { for (let n = 0; n < rows * 2; n += 1) tick(); } };
}

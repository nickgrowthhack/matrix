/* ============================================================================
 * animation.js — reconstrução (motor = reconstrutor).
 *
 * FONTE DA VERDADE = o reconstrutor:
 *   - data/interim/content_map.json  -> O QUE / ONDE / TAMANHO / FILL de cada glifo
 *   - data/interim/grid.json         -> a grade (W, H)
 *   - /animation/shapes              -> as FORMAS do molde (EFA/PCA), servidas pelo
 *                                       próprio Generator do servidor
 *
 * O "motor" (placeFit) é o porte fiel de generate.py::_place_fit. Ele gera a CENA
 * (vetorial) e ela é RASTERIZADA para um bitmap, com o ESVAECIMENTO p/ a direita já
 * "assado" (os dados se esvaem ao final). A cena é mostrada INTEIRA (sem varredura);
 * só re-renderiza quando muda (tema/esvaecimento/seed/tamanho) — custo ocioso ~0.
 * ========================================================================== */

const ANIM = (() => {
  // ---- estado de dados (a fonte da verdade) --------------------------------
  let grid = null, content = null, SHAPES = null, blobFills = null;
  let W = 5190, H = 2890;

  // ---- cena (saída do motor) + bitmap pré-rasterizado ----------------------
  let SCENE = [];
  let canvas, ctx, scale = 0.25, Wp = 0, Hp = 0;
  let sceneBmp = null, dirty = true;
  const MAXW = 1280;

  // ---- parâmetros (espelham os tunables do reconstrutor) -------------------
  const P = {
    seed: 7,
    fadeOn: true,          // esvaecimento p/ direita (como o reconstrutor)
    fadeX0: 0.50,          // generate.py: fade_x0
    fadeStrength: 1.05,
    fadeMin: 0.16,
    // paleta EXATA do original/reconstrução: fundo #323332, glifos brancos.
    bg: '#323332', fg: '#ffffff',
  };

  let needsRender = true, statusEl = null;

  // ===========================================================================
  // utilidades
  // ===========================================================================
  function mulberry32(a) {
    a >>>= 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
  function polyArea(pts) { let s = 0; for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) s += (pts[j][0] + pts[i][0]) * (pts[j][1] - pts[i][1]); return Math.abs(s) / 2; }
  function bbox(pts) { let mnx = 1e9, mxx = -1e9, mny = 1e9, mxy = -1e9; for (const p of pts) { if (p[0] < mnx) mnx = p[0]; if (p[0] > mxx) mxx = p[0]; if (p[1] < mny) mny = p[1]; if (p[1] > mxy) mxy = p[1]; } return [mnx, mxx, mny, mxy]; }
  function polyPath(p, pts) { p.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) p.lineTo(pts[i][0], pts[i][1]); p.closePath(); }
  function hexRGB(h) { h = h.replace('#', ''); return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]; }
  function bgRGBA(a) { const [r, g, b] = hexRGB(P.bg); return `rgba(${r},${g},${b},${a})`; }

  // ===========================================================================
  // O MOTOR — porte de generate.py::_place_fit (gera a CENA, 1× por seed)
  // ===========================================================================
  function placeFit(g) {
    const cx = g.cx, cy = g.cy, tw = g.bw, th = g.bh, fill = (g.fill == null ? null : g.fill);
    const path = new Path2D();

    if (g.t === '0') {
      const sh = pickShape('0', null, g._rng);
      const [mnx, mxx, mny, mxy] = bbox(sh.outer);
      const bx = (mnx + mxx) / 2, by = (mny + mxy) / 2;
      const sx = tw / Math.max(1e-6, mxx - mnx), sy = th / Math.max(1e-6, mxy - mny);
      const tf = pt => [(pt[0] - bx) * sx + cx, (pt[1] - by) * sy + cy];
      const fo = sh.outer.map(tf);
      let fi = sh.inner.map(tf);
      if (fill != null) {                          // ajuste de FILL: redimensiona o buraco
        const areaBox = Math.max(1e-6, tw * th);
        const oa = polyArea(fo), ia = polyArea(fi);
        const desI = oa - fill * areaBox;          // buraco menor => parede grossa (cheio)
        if (ia > 60) {
          let s = desI <= 0 ? 0.85 : Math.min(1.15, Math.max(0.85, Math.sqrt(desI / ia)));
          s = Math.max(s, Math.sqrt(60 / ia));     // nunca fecha o buraco (vira blob)
          fi = fi.map(pt => [(pt[0] - cx) * s + cx, (pt[1] - cy) * s + cy]);
        }
      }
      polyPath(path, fo); polyPath(path, fi);
      return { path, rule: 'evenodd' };
    } else {
      const sh = pickShape(g.t, fill, g._rng);
      const [mnx, mxx, mny, mxy] = bbox(sh.out);
      const bx = (mnx + mxx) / 2, by = (mny + mxy) / 2;
      const sx = tw / Math.max(1e-6, mxx - mnx), sy = th / Math.max(1e-6, mxy - mny);
      polyPath(path, sh.out.map(pt => [(pt[0] - bx) * sx + cx, (pt[1] - by) * sy + cy]));
      return { path, rule: 'nonzero' };
    }
  }

  // escolhe uma forma do pool do molde; blob faz fill-match (sólido esq / esparso dir)
  function pickShape(t, fill, rng) {
    const pool = SHAPES[t];
    if (t === 'blob' && fill != null) {
      let cand = [];
      for (let i = 0; i < pool.length; i++) if (Math.abs(blobFills[i] - fill) < 0.06) cand.push(i);
      if (cand.length < 8) {                       // senão, os mais próximos em fill
        cand = pool.map((_, i) => i).sort((a, b) => Math.abs(blobFills[a] - fill) - Math.abs(blobFills[b] - fill)).slice(0, 30);
      }
      return pool[cand[(rng() * cand.length) | 0]];
    }
    return pool[(rng() * pool.length) | 0];
  }

  function buildScene(seed) {
    const rng = mulberry32(seed >>> 0);
    SCENE = content.glyphs.map(g => {
      g._rng = rng;                                // generate.py amostra de um único rng
      const { path, rule } = placeFit(g);
      return { path, rule, xFrac: g.cx / W, ord: g.cx / W + (rng() * 2 - 1) * 0.045 };
    });
    dirty = true;
  }

  // ===========================================================================
  // RASTERIZAÇÃO (1× por cena/tema) — onde o motor vetorial roda
  // ===========================================================================
  function fadeAlpha(xFrac) {
    if (!P.fadeOn) return 1;
    return clamp(1 - Math.max(0, xFrac - P.fadeX0) * P.fadeStrength, P.fadeMin, 1);
  }
  function ensureScene() {
    if (!dirty && sceneBmp) return;
    const c = document.createElement('canvas');
    c.width = Wp; c.height = Hp;
    const g = c.getContext('2d');
    g.setTransform(scale, 0, 0, scale, 0, 0);
    g.fillStyle = P.fg;
    for (const s of SCENE) { g.globalAlpha = fadeAlpha(s.xFrac); g.fill(s.path, s.rule); }
    sceneBmp = c; dirty = false;
  }

  // ===========================================================================
  // render — desenha a cena INTEIRA (esvaecimento já assado no bitmap)
  // ===========================================================================
  function render() {
    ensureScene();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = P.bg; ctx.fillRect(0, 0, Wp, Hp);
    ctx.drawImage(sceneBmp, 0, 0);
    if (statusEl) statusEl.textContent = `reconstrução · ${SCENE.length} glifos`;
  }

  // só redesenha quando algo muda (tema/esvaecimento/seed/tamanho) -> ocioso ~0
  function loop() {
    requestAnimationFrame(loop);
    if (needsRender && SCENE.length) { render(); needsRender = false; }
  }

  // ===========================================================================
  // dimensionamento / bootstrap / API
  // ===========================================================================
  function resize() {
    const cssW = Math.min(canvas.parentElement.clientWidth, MAXW);
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    canvas.style.width = cssW + 'px';
    canvas.style.height = (cssW * H / W) + 'px';
    Wp = canvas.width = Math.round(cssW * dpr);
    Hp = canvas.height = Math.round(cssW * dpr * H / W);
    scale = Wp / W;
    dirty = true; needsRender = true;
  }

  async function init(opts) {
    canvas = opts.canvas; ctx = canvas.getContext('2d'); statusEl = opts.statusEl;
    try {
      [grid, content, SHAPES] = await Promise.all([
        fetch('data/interim/grid.json').then(r => r.json()),
        fetch('data/interim/content_map.json').then(r => r.json()),
        fetch('/animation/shapes').then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); }),
      ]);
    } catch (e) {
      if (statusEl) statusEl.textContent = 'erro ao carregar dados: ' + e.message +
        ' — rode o servidor FastAPI e abra via http://localhost:8000/animation.html';
      throw e;
    }
    W = grid.W; H = grid.H;
    blobFills = SHAPES.blob.map(b => b.fill);
    resize();
    buildScene(P.seed);
    window.addEventListener('resize', resize);
    requestAnimationFrame(loop);
  }

  return {
    init,
    regenerate: seed => { P.seed = (seed >>> 0); buildScene(P.seed); needsRender = true; },
    setFade: on => { P.fadeOn = on; dirty = true; needsRender = true; },
    setTheme: t => { P.bg = t.bg; P.fg = t.fg; dirty = true; needsRender = true; },
    state: () => P,
  };
})();

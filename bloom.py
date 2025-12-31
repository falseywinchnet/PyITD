<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Voronoi - Social Friction Jitter</title>
    <style>
        body { font-family: 'Segoe UI', monospace; background: #050505; color: #ccc; display: flex; flex-direction: column; align-items: center; margin: 0; padding: 10px; height: 100vh; box-sizing: border-box; overflow: hidden; }
        #canvas-wrapper { border: 2px solid #00ffff; box-shadow: 0 0 20px rgba(0, 255, 255, 0.1); background: #000; line-height: 0; margin-bottom: 10px; height: 85vh; width: auto; aspect-ratio: auto; display: flex; justify-content: center; position: relative; }
        canvas { image-rendering: pixelated; height: 100%; width: auto; max-width: 100%; object-fit: contain; }
        .controls { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; align-items: center; z-index: 10; }
        button, label { background: #1a1a1a; border: 1px solid #333; color: #00ffff; padding: 6px 12px; cursor: pointer; font-family: inherit; font-weight: bold; text-transform: uppercase; font-size: 0.75rem; border-radius: 4px; user-select: none; transition: all 0.2s; }
        button:hover, label:hover { background: #00ffff; color: #000; box-shadow: 0 0 10px rgba(0,255,255,0.4); }
        button:active { transform: translateY(1px); }
        input[type="file"] { display: none; }
        .stats { display: flex; gap: 15px; margin-bottom: 10px; font-size: 0.8rem; color: #666; background: #111; padding: 5px 15px; border-radius: 4px; border: 1px solid #222; }
        .val { color: #fff; font-weight: bold; font-size: 0.9rem; }
        #phase { color: #f0f; }
    </style>
</head>
<body>
    <div class="stats">
        <div>PHASE: <span id="phase" class="val">INIT</span></div>
        <div>ITER: <span id="iter" class="val">0</span></div>
        <div>CELLS: <span id="cells" class="val">0</span></div>
        <div>MSE: <span id="mse" class="val">0.00</span></div>
    </div>
    <div id="canvas-wrapper"><canvas id="cvs"></canvas></div>
    <div class="controls">
        <label for="file-input">📂 Load Image</label>
        <input type="file" id="file-input" accept="image/*">
        <button id="btn-reset">Reset Demo</button>
        <button id="btn-run">Pause / Run</button>
        <button id="btn-view">Mode: Synthetic</button>
    </div>

<script>
/**
 * PATCH SUMMARY (SOCIAL FRICTION):
 * 1. REMOVED: Time-based stagnation logic.
 * 2. ADDED: Comparison-based frustration. 
 * - Logic: If Cell_A.MaxError > Cell_B.MaxError, A pushes B.
 * 3. WEIGHTED PHYSICS: 
 * - Push Force = Base * (ErrorDiff) * (Mass_A / Mass_B).
 * - Larger cells resist movement from smaller cells.
 * 4. BEHAVIOR: Frustrated cells jitter themselves, push neighbors, and queue for splitting.
 */

const CFG = {
    res: 1200,
    globalLimit: 40,
    baseRadius: 50,
    minRadius: 1.0,
    decayBase: 0.05,
    lrBase: 0.2,
    views: ['Synthetic', 'Veroni', 'Error', 'Original'],
    maxCells: 6000,
    splitBatch: 150,
    jitterStrength: 5.0, // Base force for social pushing
    socialRange: 1.5 // Multiplier for neighbor check radius
};

const state = {
    running: false,
    iter: 0,
    w: 0, h: 0,
    base: null, smooth: null, err: null, owner: null, dists: null,
    cells: [],
    viewIdx: 0,
    initialTotalError: null,
    currentTotalError: 0
};

// --- MATH & UTILS ---
const s2l = x => x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
const l2s = x => x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;

function rgb2oklch(r, g, b) {
    r = s2l(r / 255); g = s2l(g / 255); b = s2l(b / 255);
    const l = 0.412221 * r + 0.536332 * g + 0.051446 * b;
    const m = 0.211903 * r + 0.680700 * g + 0.107397 * b;
    const s = 0.088302 * r + 0.281719 * g + 0.629979 * b;
    const l_ = Math.cbrt(l), m_ = Math.cbrt(m), s_ = Math.cbrt(s);
    return [
        0.210454 * l_ + 0.793618 * m_ - 0.004072 * s_,
        Math.sqrt((1.977998 * l_ - 2.428592 * m_ + 0.450594 * s_) ** 2 + (0.025904 * l_ + 0.782772 * m_ - 0.808676 * s_) ** 2),
        Math.atan2(0.025904 * l_ + 0.782772 * m_ - 0.808676 * s_, 1.977998 * l_ - 2.428592 * m_ + 0.450594 * s_)
    ];
}

function oklch2rgb(L, C, h) {
    if (isNaN(L) || L === Infinity) L = 0; if (isNaN(C) || C === Infinity) C = 0; if (isNaN(h)) h = 0;
    L = Math.max(0, Math.min(1, L)); C = Math.max(0, C);
    const a = C * Math.cos(h), b = C * Math.sin(h);
    const l_ = L + 0.396338 * a + 0.215804 * b;
    const m_ = L - 0.105561 * a - 0.063854 * b;
    const s_ = L - 0.089484 * a - 1.291486 * b;
    const lLin = l_ * l_ * l_, mLin = m_ * m_ * m_, sLin = s_ * s_ * s_;
    return [
        Math.max(0, Math.min(255, l2s(4.076742 * lLin - 3.307712 * mLin + 0.230970 * sLin) * 255)),
        Math.max(0, Math.min(255, l2s(-1.268438 * lLin + 2.609757 * mLin - 0.341319 * sLin) * 255)),
        Math.max(0, Math.min(255, l2s(-0.004196 * lLin - 0.703419 * mLin + 1.707615 * sLin) * 255))
    ];
}

function blur(d, w, h) {
    const o = new Float32Array(d.length);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            let sl = 0, sc = 0, sh = 0, n = 0;
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    const nx = x + dx, ny = y + dy;
                    if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                        const i = (ny * w + nx) * 4;
                        sl += d[i]; sc += d[i + 1]; sh += d[i + 2]; n++;
                    }
                }
            }
            const i = (y * w + x) * 4;
            o[i] = sl / n; o[i + 1] = sc / n; o[i + 2] = sh / n; o[i + 3] = d[i + 3];
        }
    }
    return o;
}

function globalBlueNoise(r, w, h) {
    const k = 30, cell = r / Math.sqrt(2);
    const cols = Math.ceil(w / cell), rows = Math.ceil(h / cell);
    const grid = new Int32Array(cols * rows).fill(-1);
    const pts = [], active = [];
    const x = Math.random() * w, y = Math.random() * h;
    pts.push({ x, y }); active.push(0);
    grid[Math.floor(y / cell) * cols + Math.floor(x / cell)] = 0;
    while (active.length) {
        const idx = Math.floor(Math.random() * active.length);
        const p = pts[active[idx]];
        let found = false;
        for (let i = 0; i < k; i++) {
            const a = Math.random() * 6.28, d = r + Math.random() * r;
            const nx = p.x + Math.cos(a) * d, ny = p.y + Math.sin(a) * d;
            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                const gx = Math.floor(nx / cell), gy = Math.floor(ny / cell);
                let ok = true;
                for (let oy = -2; oy <= 2; oy++) {
                    for (let ox = -2; ox <= 2; ox++) {
                        const cx = gx + ox, cy = gy + oy;
                        if (cx >= 0 && cx < cols && cy >= 0 && cy < rows) {
                            const nidx = grid[cy * cols + cx];
                            if (nidx !== -1) {
                                const n = pts[nidx];
                                if ((n.x - nx) ** 2 + (n.y - ny) ** 2 < r * r) { ok = false; break; }
                            }
                        }
                    }
                    if (!ok) break;
                }
                if (ok) {
                    pts.push({ x: nx, y: ny });
                    grid[gy * cols + gx] = pts.length - 1;
                    active.push(pts.length - 1);
                    found = true;
                }
            }
        }
        if (!found) active.splice(idx, 1);
    }
    return pts;
}

function localSplat(b, r) {
    const pts = [];
    const w = b.maxX - b.minX, h = b.maxY - b.minY;
    if (w <= 0 || h <= 0) return [];
    for (let i = 0; i < 6; i++) {
        const nx = b.minX + Math.random() * w, ny = b.minY + Math.random() * h;
        let ok = true;
        for (let j = 0; j < pts.length; j++) {
            if ((pts[j].x - nx) ** 2 + (pts[j].y - ny) ** 2 < r * r) { ok = false; break; }
        }
        if (ok) pts.push({ x: nx, y: ny });
    }
    return pts;
}

function updateOwners() {
    state.owner.fill(-1);
    state.dists.fill(Infinity);
    for (let i = 0; i < state.cells.length; i++) {
        const c = state.cells[i];
        if (isNaN(c.x) || isNaN(c.y)) continue;
        const r = Math.ceil(c.maxRad * 1.5) + 2;
        const minX = Math.max(0, Math.floor(c.x - r)), maxX = Math.min(state.w - 1, Math.floor(c.x + r));
        const minY = Math.max(0, Math.floor(c.y - r)), maxY = Math.min(state.h - 1, Math.floor(c.y + r));
        for (let y = minY; y <= maxY; y++) {
            const rowOffset = y * state.w;
            for (let x = minX; x <= maxX; x++) {
                const idx = rowOffset + x;
                const d2 = (x - c.x) ** 2 + (y - c.y) ** 2;
                if (d2 < state.dists[idx]) { state.dists[idx] = d2; state.owner[idx] = i; }
            }
        }
    }
}

function createCell(x, y, gen) {
    const ix = Math.floor(x), iy = Math.floor(y);
    const k = (iy * state.w + ix) * 4;
    let safeL = (state.smooth && state.smooth[k] !== undefined) ? state.smooth[k] : 0.5;
    let safeC = (state.smooth && state.smooth[k + 1] !== undefined) ? state.smooth[k + 1] : 0;
    let safeH = (state.smooth && state.smooth[k + 2] !== undefined) ? state.smooth[k + 2] : 0;
    return {
        x: x, y: y,
        L: safeL, C: safeC, H: safeH,
        dL: 0, dC: 0, dH: 0, angle: Math.random() * 6.28,
        gen: gen, maxRad: 10,
        rndColor: [Math.floor(Math.random() * 255), Math.floor(Math.random() * 255), Math.floor(Math.random() * 255)]
    };
}

function predictColor(cell, tx, ty) {
    const dx = tx - cell.x, dy = ty - cell.y;
    const proj = dx * Math.cos(cell.angle) + dy * Math.sin(cell.angle);
    return { L: cell.L + cell.dL * proj, C: cell.C + cell.dC * proj, H: cell.H + cell.dH * proj };
}

function attemptMerges(scores, stats) {
    scores.sort((a, b) => a.score - b.score);
    const budget = Math.max(10, Math.ceil(state.cells.length * 0.2));
    const cands = scores.slice(0, budget);
    const kill = new Set();
    const newCells = [];

    for (let item of cands) {
        if (kill.has(item.id)) continue;
        const cA = state.cells[item.id];
        const nA = stats[item.id * 3 + 2] || 1;
        const r = cA.maxRad + 2;
        const neighbors = new Set();

        for (let a = 0; a < 6.28; a += 1.0) {
            const sx = Math.floor(cA.x + Math.cos(a) * r), sy = Math.floor(cA.y + Math.sin(a) * r);
            if (sx >= 0 && sx < state.w && sy >= 0 && sy < state.h) {
                const oid = state.owner[sy * state.w + sx];
                if (oid !== -1 && oid !== item.id && !kill.has(oid)) neighbors.add(oid);
            }
        }
        for (let nid of neighbors) {
            const cB = state.cells[nid];
            if (!cB) continue;
            const nB = stats[nid * 3 + 2] || 1;
            const gradDiff = Math.abs(cA.dL - cB.dL) + Math.abs(cA.dC - cB.dC) + Math.abs(cA.dH - cB.dH);
            if (gradDiff > 0.015) continue;

            const midX = (cA.x + cB.x) * 0.5, midY = (cA.y + cB.y) * 0.5;
            const predA = predictColor(cA, midX, midY), predB = predictColor(cB, midX, midY);
            const diff = Math.abs(predA.L - predB.L) + Math.abs(predA.C - predB.C) + Math.abs(predA.H - predB.H) * 0.2;

            if (diff < 0.08) {
                kill.add(item.id); kill.add(nid);
                const totalN = nA + nB, rA = nA / totalN, rB = nB / totalN;
                newCells.push({
                    x: cA.x * rA + cB.x * rB, y: cA.y * rA + cB.y * rB,
                    L: cA.L * rA + cB.L * rB, C: cA.C * rA + cB.C * rB, H: cA.H * rA + cB.H * rB,
                    dL: cA.dL * rA + cB.dL * rB, dC: cA.dC * rA + cB.dC * rB, dH: cA.dH * rA + cB.dH * rB,
                    angle: cA.angle * rA + cB.angle * rB, gen: Math.max(cA.gen, cB.gen),
                    maxRad: Math.max(cA.maxRad, cB.maxRad),
                    rndColor: nA > nB ? cA.rndColor : cB.rndColor
                });
                break;
            }
        }
    }
    if (kill.size > 0) {
        state.cells = state.cells.filter((_, i) => !kill.has(i));
        state.cells.push(...newCells);
        return true;
    }
    return false;
}

function tick() {
    if (!state.running) return;

    const isGlobal = state.iter < CFG.globalLimit;
    document.getElementById('phase').innerText = (isGlobal ? "GLOBAL" : "LOCAL") + " (RUNNING)";

    // Stats Layout: [TotalError, MaxPixelError, PixelCount]
    const stats = new Float32Array(state.cells.length * 3);
    const grads = state.cells.map(() => ({ dl: 0, dc: 0, dh: 0, da: 0, n: 0 }));
    const bounds = state.cells.map(() => ({ minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity }));

    state.currentTotalError = 0;
    const wL = 1.0, wC = 2.5, wH = 0.5;

    // --- 1. PIXEL SCAN (Error & Gradients) ---
    for (let y = 0; y < state.h; y++) {
        const rowOff = y * state.w;
        for (let x = 0; x < state.w; x++) {
            const idx = rowOff + x;
            const id = state.owner[idx];
            if (id === -1 || !state.cells[id]) continue;
            const c = state.cells[id];
            const dx = x - c.x, dy = y - c.y;
            const proj = dx * Math.cos(c.angle) + dy * Math.sin(c.angle);
            const pL = c.L + c.dL * proj, pC = c.C + c.dC * proj, pH = c.H + c.dH * proj;
            const k = idx * 4;
            const tL = state.base[k], tC = state.base[k + 1], tH = state.base[k + 2];
            const diffL = tL - pL, diffC = tC - pC;
            let diffH = Math.abs(tH - pH); if (diffH > Math.PI) diffH = 2 * Math.PI - diffH;
            
            const errVal = Math.log(1.0 + Math.sqrt((diffL * wL) ** 2 + (diffC * wC) ** 2 + (diffH * wH) ** 2) * 15.0);
            state.err[idx] = errVal;
            state.currentTotalError += errVal;
            
            grads[id].dl += diffL * proj; grads[id].dc += diffC * proj;
            let hErr = tH - pH;
            if (hErr > Math.PI) hErr -= 2 * Math.PI; if (hErr < -Math.PI) hErr += 2 * Math.PI;
            grads[id].dh += hErr * proj;
            
            const dproj = -dx * Math.sin(c.angle) + dy * Math.cos(c.angle);
            grads[id].da += (diffL * c.dL + diffC * c.dC + hErr * c.dH) * dproj;
            grads[id].n++;
            
            const sid = id * 3;
            stats[sid] += errVal;
            if (errVal > stats[sid + 1]) stats[sid + 1] = errVal; // Max Error tracking
            stats[sid + 2]++; // Mass (Pixel Count)
            
            if (x < bounds[id].minX) bounds[id].minX = x; if (x > bounds[id].maxX) bounds[id].maxX = x;
            if (y < bounds[id].minY) bounds[id].minY = y; if (y > bounds[id].maxY) bounds[id].maxY = y;
            const d2 = dx * dx + dy * dy; if (d2 > c.maxRad * c.maxRad) c.maxRad = Math.sqrt(d2);
        }
    }

    if (state.initialTotalError === null && state.iter > 1) state.initialTotalError = state.currentTotalError;
    let errRatio = state.initialTotalError > 0 ? state.currentTotalError / state.initialTotalError : 1.0;

    // --- 2. SOCIAL FRICTION & JITTER (New Logic) ---
    // Accumulate forces first to avoid order dependency
    const displacements = new Float32Array(state.cells.length * 2);
    const splitRequests = new Set();
    
    if (!isGlobal) {
        state.cells.forEach((cA, idA) => {
            const sidA = idA * 3;
            const massA = stats[sidA + 2];
            const maxErrA = stats[sidA + 1];
            if (massA <= 0) return;

            // Check neighbors stochastically
            const r = cA.maxRad + 2;
            let worstNeighborDelta = 0;

            for (let ang = 0; ang < 6.28; ang += 0.8) {
                const tx = Math.floor(cA.x + Math.cos(ang) * r);
                const ty = Math.floor(cA.y + Math.sin(ang) * r);
                if (tx < 0 || tx >= state.w || ty < 0 || ty >= state.h) continue;
                
                const idB = state.owner[ty * state.w + tx];
                if (idB !== -1 && idB !== idA) {
                    const sidB = idB * 3;
                    const maxErrB = stats[sidB + 1];
                    const massB = stats[sidB + 2];

                    // COMPARISON: If my worst error is worse than neighbor's worst error
                    if (maxErrA > maxErrB * 1.1) {
                        const delta = maxErrA - maxErrB;
                        if (delta > worstNeighborDelta) worstNeighborDelta = delta;

                        // FORCE CALCULATION
                        // 1. Direction: A pushes B away
                        const pushX = Math.cos(ang);
                        const pushY = Math.sin(ang);
                        
                        // 2. Weighting: Mass Ratio (Small pushes Big -> Weak force. Big pushes Small -> Strong force)
                        // Clamp ratio to avoid explosions
                        const massRatio = Math.min(4.0, massA / Math.max(1, massB)); 
                        const force = CFG.jitterStrength * delta * massRatio;

                        displacements[idB * 2] += pushX * force;
                        displacements[idB * 2 + 1] += pushY * force;
                    }
                }
            }

            // If I am frustrated (found neighbors performing better), I also move myself and try to split
            if (worstNeighborDelta > 0) {
                const selfJitter = CFG.jitterStrength * worstNeighborDelta * 0.5;
                const rndAng = Math.random() * 6.28;
                displacements[idA * 2] += Math.cos(rndAng) * selfJitter;
                displacements[idA * 2 + 1] += Math.sin(rndAng) * selfJitter;
                splitRequests.add(idA);
            }
        });

        // Apply Displacements
        for(let i=0; i<state.cells.length; i++) {
            const c = state.cells[i];
            c.x += displacements[i*2];
            c.y += displacements[i*2+1];
            // Clamp
            c.x = Math.max(0, Math.min(state.w - 1, c.x));
            c.y = Math.max(0, Math.min(state.h - 1, c.y));
        }
        if (splitRequests.size > 0) updateOwners();
    }

    // --- 3. GRADIENT DESCENT (Color/Angle) ---
    state.cells.forEach((c, i) => {
        const g = grads[i];
        if (g.n <= 0) return;
        
        const avgErr = stats[i * 3] / g.n;
        const plastic = Math.max(0.1, Math.min(1.0, avgErr * 2.0));
        const f = (1 / g.n) * CFG.lrBase * (1.0 / (c.maxRad * 0.5 + 1.0)) * (1.0 + plastic * 2.0);
        c.dL += g.dl * f; c.dC += g.dc * f; c.dH += g.dh * f; c.angle += g.da * f * 5;
        
        const decay = CFG.decayBase + (1.0 - plastic) * 0.08;
        c.dL *= decay; c.dC *= decay; c.dH *= decay;
        const maxSlope = 0.3 / Math.max(1, c.maxRad);
        c.dL = Math.max(-maxSlope, Math.min(maxSlope, c.dL));
        c.dC = Math.max(-maxSlope, Math.min(maxSlope, c.dC));
        c.dH = Math.max(-maxSlope, Math.min(maxSlope, c.dH));
    });

    // --- 4. TOPO OPS (Split/Merge) ---
    const cycle = 15;
    let didTopo = false;
    // Score logic: Base Score + Massive Bonus if Frustrated (requested split)
    const scores = state.cells.map((c, i) => ({ 
        id: i, 
        score: (stats[i*3] + (stats[i*3+1] * stats[i*3+2] * 0.5)) * (splitRequests.has(i) ? 1000.0 : 1.0) 
    }));

    if (state.iter % cycle === 0) { 
        if (attemptMerges([...scores], stats)) { updateOwners(); didTopo = true; }
    }

    if (state.iter % cycle === 7 || isGlobal) {
        scores.sort((a, b) => b.score - a.score);
        let splitBudget = (state.cells.length < CFG.maxCells) ? Math.ceil(CFG.splitBatch * errRatio) : 0;
        // Prioritize frustrated cells even if budget is tight
        if (splitRequests.size > 0) splitBudget = Math.max(splitBudget, splitRequests.size); 
        
        const top = new Set(scores.slice(0, splitBudget).map(x => x.id));
        let cands = isGlobal ? globalBlueNoise(Math.max(CFG.minRadius, CFG.baseRadius / (state.iter + 1)), state.w, state.h) : [];
        if (!isGlobal) {
            for (let s of scores) {
                if (top.has(s.id)) cands = cands.concat(localSplat(bounds[s.id], CFG.minRadius));
            }
        }
        for (let p of cands) {
            const x = Math.floor(p.x), y = Math.floor(p.y);
            if (x < 0 || x >= state.w || y < 0 || y >= state.h) continue;
            const oid = state.owner[y * state.w + x];
            if (oid !== -1 && state.cells[oid] && top.has(oid)) {
                state.cells.push(createCell(p.x, p.y, state.cells[oid].gen + 1));
                didTopo = true;
            }
        }
    }

    if (didTopo) updateOwners();
    state.iter++;
    document.getElementById('iter').innerText = state.iter;
    document.getElementById('cells').innerText = state.cells.length;
    if (state.iter % 10 === 0) document.getElementById('mse').innerText = (state.currentTotalError / (state.w * state.h)).toFixed(4);
    render();
    requestAnimationFrame(tick);
}

function render() {
    const d = ctx.createImageData(state.w, state.h);
    const buf = d.data;
    const mode = CFG.views[state.viewIdx];
    for (let i = 0; i < state.w * state.h; i++) {
        const k = i * 4;
        let rgb = [0, 0, 0];
        if (mode.startsWith('Original')) rgb = oklch2rgb(state.base[k], state.base[k + 1], state.base[k + 2]);
        else if (mode.startsWith('Error')) { const v = Math.min(1, state.err[i] * 0.5) * 255; rgb = [v, v, v]; }
        else if (mode.startsWith('Veroni')) { 
            const id = state.owner[i]; 
            if (id !== -1 && state.cells[id]) rgb = state.cells[id].rndColor; 
        }
        else {
            const id = state.owner[i];
            if (id !== -1 && state.cells[id]) {
                const c = state.cells[id];
                const x = i % state.w, y = Math.floor(i / state.w);
                const proj = (x - c.x) * Math.cos(c.angle) + (y - c.y) * Math.sin(c.angle);
                rgb = oklch2rgb(c.L + c.dL * proj, c.C + c.dC * proj, c.H + c.dH * proj);
            }
        }
        buf[k] = rgb[0]; buf[k + 1] = rgb[1]; buf[k + 2] = rgb[2]; buf[k + 3] = 255;
    }
    ctx.putImageData(d, 0, 0);
}

const cvs = document.getElementById('cvs');
const ctx = cvs.getContext('2d');
const fileInput = document.getElementById('file-input');

function processImage(img) {
    state.running = false;
    const aspect = img.width / img.height;
    state.w = CFG.res; state.h = Math.round(CFG.res / aspect);
    cvs.width = state.w; cvs.height = state.h;
    ctx.drawImage(img, 0, 0, state.w, state.h);
    const raw = ctx.getImageData(0, 0, state.w, state.h).data;
    state.base = new Float32Array(state.w * state.h * 4);
    for (let i = 0; i < raw.length; i += 4) {
        const o = rgb2oklch(raw[i], raw[i + 1], raw[i + 2]);
        state.base[i] = o[0]; state.base[i + 1] = o[1]; state.base[i + 2] = o[2]; state.base[i + 3] = 255;
    }
    state.smooth = blur(state.base, state.w, state.h);
    state.err = new Float32Array(state.w * state.h).fill(0);
    state.owner = new Int32Array(state.w * state.h).fill(-1);
    state.dists = new Float32Array(state.w * state.h).fill(Infinity);
    state.cells = [];
    const pts = globalBlueNoise(30, state.w, state.h);
    pts.forEach(p => state.cells.push(createCell(p.x, p.y, 0)));
    updateOwners();
    state.iter = 0; state.initialTotalError = null; render();
}

fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => { const img = new Image(); img.onload = () => processImage(img); img.src = evt.target.result; };
    reader.readAsDataURL(file);
};
document.getElementById('btn-reset').onclick = () => {
    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.onload = () => processImage(img);
    img.src = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png";
};
document.getElementById('btn-run').onclick = () => { state.running = !state.running; if (state.running) tick(); };
document.getElementById('btn-view').onclick = () => {
    state.viewIdx = (state.viewIdx + 1) % CFG.views.length;
    document.getElementById('btn-view').innerText = "Mode: " + CFG.views[state.viewIdx];
    render();
};
document.getElementById('btn-reset').click();
</script>
</body>
</html>

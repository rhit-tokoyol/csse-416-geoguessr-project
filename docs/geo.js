// In-browser version of demo_predict.ClipCell: CLIP image features -> trained head
// -> map cells, adjusted by the driving-side detector when the head has one.
// The head is either the built-in one in ./model or a .geomodel file the visitor loads
// (made by export_web.py --bundle). Loaded files stay in the browser.
import { CLIPVisionModelWithProjection, Tensor, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1";

env.allowLocalModels = false;   // CLIP comes from the Hugging Face CDN; heads come from ./model or a file

// Which CLIP the page runs is decided by the loaded model file (meta.clip_model, meta.clip_dim,
// meta.image_size). Files made before that was recorded are the original B/32 setup.
const DEFAULT_CLIP = "Xenova/clip-vit-base-patch32", DEFAULT_DIM = 512, DEFAULT_SIZE = 224;
let meta = null, W = null, kind = null, clip = null, clipDtype = null, clipName = null;
const builtIns = new Map();   // folder -> { meta, W, kind }: the heads that ship with the site

// ---------- heads ----------
const fail = (msg) => { throw new Error(msg); };
const isInt = (v) => Number.isInteger(v) && v > 0;

// Checks a head's metadata and weights fully before anything uses them, so a wrong or
// damaged file gives a clear message instead of strange guesses.
function useHead(m, floats, fallbackName) {
  if (!m || typeof m !== "object") fail("The model file has no metadata.");
  const n = Array.isArray(m.cells) ? m.cells.length : fail("The model file has no cell list.");
  if (n < 2) fail("The model file needs at least 2 cells.");
  for (const key of ["cell_lat", "cell_lon", "cell_country"]) {
    if (!Array.isArray(m[key]) || m[key].length !== n) fail(`The model file’s ${key} doesn’t match its ${n} cells.`);
  }
  if (!m.cell_lat.every(v => Number.isFinite(v) && Math.abs(v) <= 90) ||
      !m.cell_lon.every(v => Number.isFinite(v) && Math.abs(v) <= 180)) fail("The model file has cell coordinates off the map.");
  const left = Array.isArray(m.cell_left) && m.cell_left.length === n ? m.cell_left : new Array(n).fill(0.5);
  if (!left.every(v => Number.isFinite(v) && v >= 0 && v <= 1)) fail("The model file’s cell_left values must be between 0 and 1.");
  // The backbone is a Hugging Face repo with an ONNX build (the page downloads it from the Hub), so it
  // has to look like "owner/name".
  const clipModel = m.clip_model ?? DEFAULT_CLIP;
  if (typeof clipModel !== "string" || !/^[\w.-]+\/[\w.-]+$/.test(clipModel)) fail("The model file’s clip_model must be a Hugging Face name like owner/name.");
  const dim = m.clip_dim ?? DEFAULT_DIM;
  if (!isInt(dim) || dim > 4096) fail("The model file’s clip_dim must be a whole number up to 4096.");
  const S = Array.isArray(m.image_size) ? m.image_size[0] : DEFAULT_SIZE;
  if (!isInt(S) || S < 64 || S > 1024 || (Array.isArray(m.image_size) && m.image_size[1] !== S)) fail("The model file’s image_size must be a square size between 64 and 1024.");
  const ok3 = (a) => Array.isArray(a) && a.length === 3 && a.every(Number.isFinite);
  if (!ok3(m.mean) || !ok3(m.std) || !m.std.every(v => v > 0)) fail("The model file’s mean/std must be 3 numbers each.");
  const squash = Array.isArray(m.resize) && m.resize.length === 2 && m.resize[0] === S && m.resize[1] === S && !m.center_crop;
  const crop = isInt(m.resize) && m.resize >= S && m.resize <= 1024 && m.center_crop === S;
  if (!squash && !crop) fail(`The model file’s resize must be [${S}, ${S}] (whole photo) or a shorter side of ${S}+ with a ${S} center crop.`);
  const fw = m.fusion_weight ?? 0;
  if (!Number.isFinite(fw) || fw < 0 || fw > 20) fail("The model file’s fusion_weight must be between 0 and 20.");

  const allowed = new Set(["trunk.0.weight", "trunk.0.bias", "cell_out.weight", "cell_out.bias",
    "side_net.0.weight", "side_net.0.bias", "side_net.3.weight", "side_net.3.bias", "side_out.weight", "side_out.bias",
    "stem0.weight", "stem0.bias", "res1a.weight", "res1a.bias", "res1b.weight", "res1b.bias",
    "stem1.weight", "stem1.bias", "res2a.weight", "res2a.bias", "res2b.weight", "res2b.bias"]);
  const w = {};
  for (const t of Array.isArray(m.tensors) ? m.tensors : fail("The model file lists no weights.")) {
    if (!allowed.has(t?.name)) fail(`Unknown weight “${String(t?.name).slice(0, 40)}” in the model file.`);
    if (!Array.isArray(t.shape) || !t.shape.every(isInt) || !Number.isInteger(t.offset) || t.offset < 0) fail(`Weight ${t.name} has a bad shape or offset.`);
    const size = t.shape.reduce((a, b) => a * b, 1);
    if (t.offset + size > floats.length) fail(`Weight ${t.name} runs past the end of the file. Is it complete?`);
    w[t.name] = { data: floats.subarray(t.offset, t.offset + size), shape: t.shape };
  }
  const shape = (name, want) => {
    const s = w[name]?.shape || fail(`The model file is missing ${name}.`);
    if (s.length !== want.length || s.some((v, i) => v !== want[i])) fail(`${name} should be [${want}] but is [${s}].`);
  };
  let k = "plain";
  if (w["stem0.weight"]) {
    // Residual MLP head (export_web.py folds its BatchNorm layers into the Linear layers): stem0, a
    // residual block (res1a, res1b), stem1, a second block (res2a, res2b), then cell_out. No driving side.
    const w1 = w["stem0.weight"].shape[0], w2 = w["stem1.weight"]?.shape[0] || fail("The model file is missing stem1.weight.");
    shape("stem0.weight", [w1, dim]); shape("stem0.bias", [w1]);
    for (const r of ["res1a", "res1b"]) { shape(`${r}.weight`, [w1, w1]); shape(`${r}.bias`, [w1]); }
    shape("stem1.weight", [w2, w1]); shape("stem1.bias", [w2]);
    for (const r of ["res2a", "res2b"]) { shape(`${r}.weight`, [w2, w2]); shape(`${r}.bias`, [w2]); }
    shape("cell_out.weight", [n, w2]); shape("cell_out.bias", [n]);
    k = "residual";
  } else {
    const hidden = w["trunk.0.weight"]?.shape[0] || fail("The model file is missing trunk.0.weight.");
    shape("trunk.0.weight", [hidden, dim]); shape("trunk.0.bias", [hidden]);
    shape("cell_out.weight", [n, hidden]); shape("cell_out.bias", [n]);
    if (w["side_net.0.weight"]) {
      const s = w["side_net.0.weight"].shape[0];
      shape("side_net.0.weight", [s, dim]); shape("side_net.0.bias", [s]);
      shape("side_net.3.weight", [1, s]); shape("side_net.3.bias", [1]);
      k = "mirror";
    } else if (w["side_out.weight"]) {
      shape("side_out.weight", [1, hidden]); shape("side_out.bias", [1]);
      k = "shared";
    }
  }
  const name = (typeof m.name === "string" && m.name.trim() ? m.name.trim() : fallbackName).slice(0, 80);
  meta = { ...m, name, cell_left: left, fusion_weight: k === "plain" || k === "residual" ? 0 : fw, head_kind: k,
           clip_model: clipModel, clip_dim: dim, image_size: [S, S] };
  W = w; kind = k;
  return meta;
}

export function activeModel() { return meta; }

// The CLIP backbone the active head needs (the page reloads CLIP when this changes).
export function wantedClip() { return meta?.clip_model ?? DEFAULT_CLIP; }

// The models that ship with the site: the default is docs/model; others live in their own folders
// (for example docs/model/run12), each holding meta.json and head.bin (export_web.py --out-dir).
async function fetchHead(base) {
  const [m, buf] = await Promise.all([
    fetch(`${base}/meta.json`).then(r => r.ok ? r.json() : fail(`Couldn't load ${base}/meta.json`)),
    fetch(`${base}/head.bin`).then(r => r.ok ? r.arrayBuffer() : fail(`Couldn't load ${base}/head.bin`)),
  ]);
  if (buf.byteLength % 4) fail(`${base}/head.bin is damaged.`);
  useHead(m, new Float32Array(buf), "Built-in model");   // validates first; only then becomes the active head
  meta.base = base;
  builtIns.set(base, { meta, W, kind });
  return meta;
}

// Loads a built-in head the first time; afterwards returns it without switching the active head.
export async function loadHead(base = "model") {
  return builtIns.has(base) ? builtIns.get(base).meta : fetchHead(base);
}

// Makes the built-in head in folder `base` the active one (loading it first if needed).
export async function selectBuiltIn(base) {
  if (!builtIns.has(base)) return fetchHead(base);
  ({ meta, W, kind } = builtIns.get(base));
  return meta;
}

export function useBuiltIn(base = "model") {
  if (!builtIns.has(base)) fail("The built-in model hasn't loaded.");
  ({ meta, W, kind } = builtIns.get(base));
  return meta;
}

export function isBuiltIn(m) { return [...builtIns.values()].some(b => b.meta === m); }

// A .geomodel file: "GEOMODEL1\n", uint32 metadata length (little-endian), metadata JSON,
// then float32 weights (export_web.py --bundle).
export async function loadBundle(file) {
  if (file.size > 200e6) fail("That file is over 200 MB, too big to be a head for this site.");
  const buf = await file.arrayBuffer(), bytes = new Uint8Array(buf);
  if (buf.byteLength < 14 || new TextDecoder().decode(bytes.subarray(0, 10)) !== "GEOMODEL1\n") {
    fail("That isn’t a .geomodel file. Make one with: python export_web.py <model.pt> --bundle name.geomodel");
  }
  const len = new DataView(buf).getUint32(10, true), start = 14 + len;
  if (start > buf.byteLength || start % 4 || (buf.byteLength - start) % 4) fail("The .geomodel file is damaged or incomplete.");
  let m;
  try { m = JSON.parse(new TextDecoder().decode(bytes.subarray(14, start))); }
  catch { fail("The .geomodel file’s metadata isn’t readable."); }
  const prev = { meta, W, kind };
  try { return useHead(m, new Float32Array(buf, start), file.name.replace(/\.geomodel$/i, "")); }
  catch (e) { ({ meta, W, kind } = prev); throw e; }   // keep the model that was working
}

// dtype "fp16" (~176 MB) and "fp32" (~352 MB) both match the Python model's guesses; the 8-bit
// "q8" file (~89 MB) changed the top guess on half of our test photos, so the page doesn't offer it.
// The file sizes above are for B/32; larger backbones are proportionally bigger downloads.
export async function loadClip(dtype = "fp16", onProgress = () => {}) {
  const name = wantedClip();
  if (clip && clipDtype === dtype && clipName === name) return;
  clip = null; clipName = null;           // drop the old backbone before loading another
  clip = await CLIPVisionModelWithProjection.from_pretrained(name, {
    dtype, device: "wasm",
    progress_callback: (p) => { if (p.status === "progress") onProgress(p); },
  });
  clipDtype = dtype; clipName = name;
}

// One throwaway run on a blank image: the model's first run is slow because it sets itself up,
// so doing it in the background makes the visitor's first real guess fast.
export async function warmUp() {
  const c = document.createElement("canvas");
  c.width = c.height = meta?.image_size?.[0] ?? DEFAULT_SIZE;
  await predict(c);
}

// PIL's BILINEAR resize (Image.resize), which training used through torchvision.
// When shrinking it averages over a triangle filter as wide as the scale factor (antialiasing),
// then rounds to 8 bits after each pass. The browser's canvas resize does not, and CLIP's
// features are sensitive to that difference.
function pilWeights(inSize, outSize) {
  const scale = inSize / outSize, filterscale = Math.max(scale, 1), support = filterscale;
  const taps = [];
  for (let o = 0; o < outSize; o++) {
    const center = (o + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize);
    const w = [];
    let sum = 0;
    for (let x = xmin; x < xmax; x++) {
      const t = Math.abs((x - center + 0.5) / filterscale);
      const v = t < 1 ? 1 - t : 0;
      w.push(v); sum += v;
    }
    taps.push({ xmin, w: w.map(v => (sum ? v / sum : 0)) });
  }
  return taps;
}
const clip8 = (v) => Math.min(255, Math.max(0, Math.round(v)));   // PIL rounds to uint8 between passes

export function pilResize(rgba, inW, inH, outW, outH) {
  const hx = pilWeights(inW, outW), hy = pilWeights(inH, outH);
  const mid = new Uint8ClampedArray(outW * inH * 3);              // horizontal pass
  for (let y = 0; y < inH; y++) {
    for (let o = 0; o < outW; o++) {
      const { xmin, w } = hx[o];
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < w.length; k++) {
        const i = (y * inW + xmin + k) * 4;
        r += rgba[i] * w[k]; g += rgba[i + 1] * w[k]; b += rgba[i + 2] * w[k];
      }
      const j = (y * outW + o) * 3;
      mid[j] = clip8(r); mid[j + 1] = clip8(g); mid[j + 2] = clip8(b);
    }
  }
  const out = new Uint8ClampedArray(outW * outH * 3);             // vertical pass
  for (let o = 0; o < outH; o++) {
    const { xmin, w } = hy[o];
    for (let x = 0; x < outW; x++) {
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < w.length; k++) {
        const i = ((xmin + k) * outW + x) * 3;
        r += mid[i] * w[k]; g += mid[i + 1] * w[k]; b += mid[i + 2] * w[k];
      }
      const j = (o * outW + x) * 3;
      out[j] = clip8(r); out[j + 1] = clip8(g); out[j + 2] = clip8(b);
    }
  }
  return out;
}

// Python's round(): halves go to the even number (torchvision's CenterCrop uses it).
const roundHalfEven = (x) => { const f = Math.floor(x), d = x - f; return d > 0.5 || (d === 0.5 && f % 2) ? f + 1 : f; };

// Same preprocessing as training: crop (on whole pixels, like the Python demo), resize with PIL's
// filter, scale to 0-1, normalise. Two resize styles:
//   resize [224, 224]           squash the whole photo (current notebook)
//   resize 224 + center_crop    shorter side to 224, then the middle 224x224 (run 1, torchvision Resize + CenterCrop)
// Returns pixel_values for the image, plus its mirror image when the head compares the two.
function preprocess(source, crop, withMirror) {
  const sw = source.width, sh = source.height;
  const [fx0, fy0, fx1, fy1] = crop || [0, 0, 1, 1];
  const x0 = Math.round(fx0 * sw), y0 = Math.round(fy0 * sh);
  const cw = Math.max(1, Math.round(fx1 * sw) - x0), ch = Math.max(1, Math.round(fy1 * sh) - y0);
  const canvas = document.createElement("canvas");
  canvas.width = cw; canvas.height = ch;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(source, x0, y0, cw, ch, 0, 0, cw, ch);          // 1:1 copy, no scaling
  const px = ctx.getImageData(0, 0, cw, ch).data;

  const S = meta.image_size[0];
  let rgb;
  if (Array.isArray(meta.resize)) {
    rgb = pilResize(px, cw, ch, S, S);
  } else {
    const r = meta.resize;   // torchvision Resize(int): shorter side -> r, longer side truncated
    const [rw, rh] = cw <= ch ? [r, Math.trunc(r * ch / cw)] : [Math.trunc(r * cw / ch), r];
    const big = pilResize(px, cw, ch, rw, rh);
    const top = roundHalfEven((rh - S) / 2), left = roundHalfEven((rw - S) / 2);
    rgb = new Uint8ClampedArray(S * S * 3);
    for (let y = 0; y < S; y++) rgb.set(big.subarray(((top + y) * rw + left) * 3, ((top + y) * rw + left + S) * 3), y * S * 3);
  }

  const batch = withMirror ? 2 : 1, plane = S * S, out = new Float32Array(batch * 3 * plane);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = (y * S + x) * 3, j = y * S + x, jm = y * S + (S - 1 - x);
      for (let c = 0; c < 3; c++) {
        const v = (rgb[i + c] / 255 - meta.mean[c]) / meta.std[c];
        out[c * plane + j] = v;                                    // original
        if (withMirror) out[3 * plane + c * plane + jm] = v;       // mirrored (flip width)
      }
    }
  }
  return { tensor: new Tensor("float32", out, [batch, 3, S, S]), size: [cw, ch] };
}

function linear(x, name) {          // y = W x + b, W stored [out, in] like torch
  const { data, shape: [o, n] } = W[`${name}.weight`], b = W[`${name}.bias`].data;
  const y = new Float32Array(o);
  for (let r = 0; r < o; r++) {
    let s = b[r];
    const row = r * n;
    for (let k = 0; k < n; k++) s += data[row + k] * x[k];
    y[r] = s;
  }
  return y;
}
const relu = (x) => x.map(v => (v > 0 ? v : 0));

// ---------- residual MLP head ----------
// Exact GELU, like torch.nn.GELU: x * Phi(x). erf is Abramowitz & Stegun 7.1.26 (error below 1.5e-7).
function erf(x) {
  const a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return Math.sign(x) * (1 - poly * Math.exp(-a * a));
}
const gelu = (x) => x.map(v => 0.5 * v * (1 + erf(v / Math.SQRT2)));
// stem0 -> GELU -> [residual block] -> stem1 -> GELU -> [residual block] -> cell_out, where a residual
// block is GELU(h + res_b(GELU(res_a(h)))). Dropout is off at prediction time.
function residualLogits(f) {
  let h = gelu(linear(f, "stem0"));
  let r = linear(gelu(linear(h, "res1a")), "res1b");
  h = gelu(h.map((v, i) => v + r[i]));
  h = gelu(linear(h, "stem1"));
  r = linear(gelu(linear(h, "res2a")), "res2b");
  h = gelu(h.map((v, i) => v + r[i]));
  return linear(h, "cell_out");
}
const sideNet = (f) => linear(relu(linear(f, "side_net.0")), "side_net.3")[0];

// ---------- expected-score decoding ----------
// The most probable cell isn't always the best guess: when the probability is spread over cells that
// are close together, a cell in the middle of them earns more points on average. S[i][j] =
// exp(-km(i, j) / 1492.7) is the share of the 5000 points you'd get for guessing cell j when the photo
// is really in cell i (the same formula as the Score box), so the best guess maximises
//   expected(j) = sum_i p_i * S[i][j]
// This is the notebook's predict_expected.
const SCORE_KM = 1492.7, MAX_SIM_CELLS = 4000;   // the table is n x n floats, so very large heads skip this
const simCache = new WeakMap();                   // head metadata -> its n x n table, built once
function similarity() {
  const n = meta.cells.length;
  if (n > MAX_SIM_CELLS) return null;
  let S = simCache.get(meta);
  if (S) return S;
  S = new Float32Array(n * n);
  const rad = Math.PI / 180, lat = meta.cell_lat.map(v => v * rad), lon = meta.cell_lon.map(v => v * rad);
  const cosLat = lat.map(Math.cos);
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const a = Math.sin((lat[j] - lat[i]) / 2) ** 2 + cosLat[i] * cosLat[j] * Math.sin((lon[j] - lon[i]) / 2) ** 2;
      S[i * n + j] = S[j * n + i] = Math.exp(-(2 * 6371 * Math.asin(Math.min(1, Math.sqrt(a)))) / SCORE_KM);
    }
  }
  simCache.set(meta, S);
  return S;
}

// Expected share of the 5000 points for guessing each cell, or null if the table is too big.
function expectedScores(probs) {
  const S = similarity();
  if (!S) return null;
  const n = probs.length, E = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const p = probs[i];
    if (p < 1e-7) continue;                       // negligible probability can't change the ranking
    const row = i * n;
    for (let j = 0; j < n; j++) E[j] += p * S[row + j];
  }
  return E;
}

// Top-k cells. By probability, or (expected = true) by expected points; `prob` is always the probability.
function rank(scores, k, expected = false) {
  const m = Math.max(...scores);
  const e = scores.map(s => Math.exp(s - m));
  const z = e.reduce((a, b) => a + b, 0);
  const probs = e.map(v => v / z);
  const E = expected ? expectedScores(probs) : null;
  const key = E || probs;
  const order = [...key.keys()].sort((a, b) => key[b] - key[a]).slice(0, k);
  return order.map(j => ({
    cell: meta.cells[j], country: meta.cell_country[j],
    lat: meta.cell_lat[j], lon: meta.cell_lon[j], prob: probs[j],
    expected: E ? E[j] : null,                    // share of 5000 points, when ranking by expected score
  }));
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

export async function predict(source, { crop = null, k = 5, expected = true } = {}) {
  if (!clip || clipName !== wantedClip()) fail("CLIP for this model isn't loaded yet.");
  const { tensor, size } = preprocess(source, crop, kind === "mirror");
  const { image_embeds } = await clip({ pixel_values: tensor });
  const emb = image_embeds.data, d = image_embeds.dims[1];
  const f = emb.slice(0, d);

  const h = kind === "residual" ? null : relu(linear(f, "trunk.0"));
  const logits = kind === "residual" ? residualLogits(f) : linear(h, "cell_out");
  const m = Math.max(...logits);
  const lse = m + Math.log(logits.reduce((a, v) => a + Math.exp(v - m), 0));
  const logp = Array.from(logits, v => v - lse);

  // Driving side: mirror heads compare the photo with its mirror image; run-1 heads read the
  // shared layer; plain heads have none, and then the guesses are not adjusted.
  const pLeft = kind === "mirror" ? sigmoid(sideNet(f) - sideNet(emb.slice(d, 2 * d)))
    : kind === "shared" ? sigmoid(linear(h, "side_out")[0]) : null;
  let fused = logp;
  if (pLeft !== null && meta.fusion_weight > 0) {
    const eps = 1e-4, p = Math.min(Math.max(pLeft, eps), 1 - eps), w = meta.fusion_weight;
    fused = logp.map((lp, j) => {
      const L = meta.cell_left[j];
      return lp + w * Math.log(Math.min(Math.max(p * L + (1 - p) * (1 - L), eps), 1));
    });
  }
  return {
    size, p_left_driving: pLeft,
    guesses: rank(fused, k, expected),
    plain_top: rank(logp, 1, expected)[0],
    embedding: f,
  };
}

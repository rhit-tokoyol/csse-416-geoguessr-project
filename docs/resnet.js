// Optional ResNet-101 country model, run in the browser with onnxruntime-web.
// The model file and its metadata (country codes, map pins, resize/normalise numbers) are made by
// export_resnet_onnx.py into docs/model/. The page loads this file only when the visitor ticks
// "Also run ResNet", so a missing model or a failed download never affects the CLIP guesses.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs";
import { pilResize } from "./geo.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = 1;   // threads need cross-origin isolation, which GitHub Pages doesn't send

let meta = null, session = null;
const fail = (msg) => { throw new Error(msg); };

// Checks the metadata fully before anything uses it, so a wrong file gives a clear message.
function validate(m, needUrl) {
  if (!m || typeof m !== "object") fail("The ResNet file has no metadata.");
  const ok3 = (a) => Array.isArray(a) && a.length === 3 && a.every(Number.isFinite);
  if (needUrl && (typeof m.url !== "string" || !m.url)) fail("resnet_meta.json has no model url.");
  if (!Array.isArray(m.classes) || m.classes.length < 2) fail("The ResNet file has no country list.");
  if (!m.centers || typeof m.centers !== "object") fail("The ResNet file has no country positions.");
  if (!Number.isInteger(m.input_size) || !Number.isInteger(m.resize) || m.resize < m.input_size || m.input_size < 32 || m.resize > 2048) fail("The ResNet file has a bad image size.");
  if (!ok3(m.mean) || !ok3(m.std) || !m.std.every(v => v > 0)) fail("The ResNet file's mean/std must be 3 numbers each.");
  return m;
}

export async function loadMeta(base = "model") {
  if (meta) return meta;
  const r = await fetch(`${base}/resnet_meta.json`);
  if (!r.ok) fail("No ResNet model is installed on this site.");
  meta = validate(await r.json(), true);
  return meta;
}

export function activeName() { return meta?.name ?? null; }

// A .resnetmodel file (pack_resnet_bundle.py): "RESNET1\n", uint32 (little-endian) metadata length,
// metadata JSON, then the .onnx bytes. It replaces the site's own model until the page reloads.
export async function loadBundle(file) {
  if (file.size > 400e6) fail("That file is over 400 MB, too big to run in a browser tab.");
  const buf = await file.arrayBuffer(), head = new Uint8Array(buf, 0, Math.min(12, buf.byteLength));
  if (buf.byteLength < 12 || new TextDecoder().decode(head.subarray(0, 8)) !== "RESNET1\n") {
    fail("That isn't a .resnetmodel file. Make one with: python pack_resnet_bundle.py resnet_meta.json resnet_country.onnx");
  }
  const len = new DataView(buf).getUint32(8, true), start = 12 + len;
  if (start >= buf.byteLength) fail("The .resnetmodel file is damaged or incomplete.");
  let m;
  try { m = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 12, len))); }
  catch { fail("The .resnetmodel file's metadata isn't readable."); }
  validate(m, false);
  const s = await ort.InferenceSession.create(new Uint8Array(buf.slice(start)), { executionProviders: ["wasm"] });
  if (!s.inputNames.includes("pixels") || !s.outputNames.includes("logits")) fail("That ONNX model doesn't have the expected pixels/logits inputs and outputs.");
  session = s; meta = m;                       // only replace the working model once the new one loaded
  return meta;
}

// Downloads the .onnx (reporting progress) and starts an inference session.
export async function load(onProgress = () => {}) {
  if (session) return;
  const m = await loadMeta();
  const res = await fetch(m.url);
  if (!res.ok) fail(`Couldn't download the ResNet model (${res.status}). Check "url" in resnet_meta.json.`);
  const total = Number(res.headers.get("content-length")) || (m.size_mb || 0) * 1e6;
  const reader = res.body.getReader(), chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); loaded += value.length;
    onProgress(loaded, total || loaded);
  }
  const bytes = new Uint8Array(loaded);
  let at = 0;
  for (const c of chunks) { bytes.set(c, at); at += c.length; }
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
}

// Python's round(): halves go to the even number (torchvision's CenterCrop uses it).
const roundHalfEven = (x) => { const f = Math.floor(x), d = x - f; return d > 0.5 || (d === 0.5 && f % 2) ? f + 1 : f; };

// Same preprocessing as training: crop the selected box, shorter side to `resize` (PIL bilinear,
// like torchvision), middle input_size x input_size square, scale to 0-1, normalise.
function preprocess(source, crop) {
  const sw = source.width, sh = source.height;
  const [fx0, fy0, fx1, fy1] = crop || [0, 0, 1, 1];
  const x0 = Math.round(fx0 * sw), y0 = Math.round(fy0 * sh);
  const cw = Math.max(1, Math.round(fx1 * sw) - x0), ch = Math.max(1, Math.round(fy1 * sh) - y0);
  const canvas = document.createElement("canvas");
  canvas.width = cw; canvas.height = ch;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(source, x0, y0, cw, ch, 0, 0, cw, ch);          // 1:1 copy, no scaling
  const px = ctx.getImageData(0, 0, cw, ch).data;

  const S = meta.input_size, r = meta.resize;
  const [rw, rh] = cw <= ch ? [r, Math.trunc(r * ch / cw)] : [Math.trunc(r * cw / ch), r];
  const big = pilResize(px, cw, ch, rw, rh);
  const top = roundHalfEven((rh - S) / 2), left = roundHalfEven((rw - S) / 2);

  const plane = S * S, out = new Float32Array(3 * plane);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = ((top + y) * rw + left + x) * 3, j = y * S + x;
      for (let c = 0; c < 3; c++) out[c * plane + j] = (big[i + c] / 255 - meta.mean[c]) / meta.std[c];
    }
  }
  return out;
}

// Top-k countries as [{country, prob, lat, lon}]. `source` is an ImageBitmap/canvas; `crop` is
// [x0, y0, x1, y1] as fractions of the image (or null for the whole image).
export async function predict(source, { crop = null, k = 3 } = {}) {
  if (!session) fail("The ResNet model isn't loaded yet.");
  const S = meta.input_size;
  const { logits } = await session.run({ pixels: new ort.Tensor("float32", preprocess(source, crop), [1, 3, S, S]) });
  const z = Array.from(logits.data), m = Math.max(...z);
  const e = z.map(v => Math.exp(v - m)), sum = e.reduce((a, b) => a + b, 0);
  const probs = e.map(v => v / sum);
  return [...probs.keys()].sort((a, b) => probs[b] - probs[a])
    .filter(j => meta.centers[meta.classes[j]]).slice(0, k)       // a country with no map position can't be pinned
    .map(j => {
      const code = meta.classes[j], [lat, lon] = meta.centers[code];
      return { country: code, prob: probs[j], lat, lon };
    });
}

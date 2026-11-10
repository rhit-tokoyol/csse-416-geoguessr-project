// In-browser version of demo_predict.ClipCell: CLIP image features -> trained head
// -> map cells, adjusted by the mirror-based driving-side detector.
import { CLIPVisionModelWithProjection, Tensor, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1";

env.allowLocalModels = false;   // CLIP comes from the Hugging Face CDN; our head comes from ./model

let meta = null, W = null, clip = null, clipDtype = null;

export async function loadHead(base = "model") {
  if (meta) return meta;
  const [m, buf] = await Promise.all([
    fetch(`${base}/meta.json`).then(r => { if (!r.ok) throw new Error("Couldn't load model/meta.json"); return r.json(); }),
    fetch(`${base}/head.bin`).then(r => { if (!r.ok) throw new Error("Couldn't load model/head.bin"); return r.arrayBuffer(); }),
  ]);
  const all = new Float32Array(buf);
  W = {};
  for (const t of m.tensors) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    W[t.name] = { data: all.subarray(t.offset, t.offset + n), shape: t.shape };
  }
  meta = m;
  return meta;
}

// dtype "fp16" (~176 MB) and "fp32" (~352 MB) both match the Python model's guesses; the 8-bit
// "q8" file (~89 MB) changed the top guess on half of our test photos, so the page doesn't offer it.
export async function loadClip(dtype = "fp16", onProgress = () => {}) {
  if (clip && clipDtype === dtype) return;
  clip = await CLIPVisionModelWithProjection.from_pretrained(meta.clip_model, {
    dtype, device: "wasm",
    progress_callback: (p) => { if (p.status === "progress") onProgress(p); },
  });
  clipDtype = dtype;
}

// One throwaway run on a blank image: the model's first run is slow because it sets itself up,
// so doing it in the background makes the visitor's first real guess fast.
export async function warmUp() {
  const c = document.createElement("canvas");
  c.width = meta.image_size[0]; c.height = meta.image_size[1];
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

// Same preprocessing as training: crop (on whole pixels, like the Python demo), squash to
// 224x224 with PIL's resize, scale to 0-1, normalise.
// Returns pixel_values for the image and its mirror image, batched together.
function preprocess(source, crop) {
  const [w, h] = meta.image_size;
  const sw = source.width, sh = source.height;
  const [fx0, fy0, fx1, fy1] = crop || [0, 0, 1, 1];
  const x0 = Math.round(fx0 * sw), y0 = Math.round(fy0 * sh);
  const cw = Math.max(1, Math.round(fx1 * sw) - x0), ch = Math.max(1, Math.round(fy1 * sh) - y0);
  const canvas = document.createElement("canvas");
  canvas.width = cw; canvas.height = ch;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(source, x0, y0, cw, ch, 0, 0, cw, ch);          // 1:1 copy, no scaling
  const rgb = pilResize(ctx.getImageData(0, 0, cw, ch).data, cw, ch, w, h);
  const plane = w * h, out = new Float32Array(2 * 3 * plane);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 3, j = y * w + x, jm = y * w + (w - 1 - x);
      for (let c = 0; c < 3; c++) {
        const v = (rgb[i + c] / 255 - meta.mean[c]) / meta.std[c];
        out[c * plane + j] = v;                 // original
        out[3 * plane + c * plane + jm] = v;    // mirrored (flip width)
      }
    }
  }
  return { tensor: new Tensor("float32", out, [2, 3, h, w]), size: [cw, ch] };
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
const sideNet = (f) => linear(relu(linear(f, "side_net.0")), "side_net.3")[0];

function rank(scores, k) {
  const m = Math.max(...scores);
  const e = scores.map(s => Math.exp(s - m));
  const z = e.reduce((a, b) => a + b, 0);
  const probs = e.map(v => v / z);
  const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, k);
  return order.map(j => ({
    cell: meta.cells[j], country: meta.cell_country[j],
    lat: meta.cell_lat[j], lon: meta.cell_lon[j], prob: probs[j],
  }));
}

export async function predict(source, { crop = null, k = 5 } = {}) {
  const { tensor, size } = preprocess(source, crop);
  const { image_embeds } = await clip({ pixel_values: tensor });
  const emb = image_embeds.data, d = image_embeds.dims[1];
  const f = emb.slice(0, d), fm = emb.slice(d, 2 * d);

  const logits = linear(relu(linear(f, "trunk.0")), "cell_out");
  const m = Math.max(...logits);
  const lse = m + Math.log(logits.reduce((a, v) => a + Math.exp(v - m), 0));
  const logp = Array.from(logits, v => v - lse);

  const pLeft = 1 / (1 + Math.exp(-(sideNet(f) - sideNet(fm))));
  const eps = 1e-4, p = Math.min(Math.max(pLeft, eps), 1 - eps), w = meta.fusion_weight;
  const fused = logp.map((lp, j) => {
    const L = meta.cell_left[j];
    return lp + w * Math.log(Math.min(Math.max(p * L + (1 - p) * (1 - L), eps), 1));
  });
  return {
    size, p_left_driving: pLeft,
    guesses: rank(fused, k),
    plain_top: rank(logp, 1)[0],
    embedding: f,
  };
}

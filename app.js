import {
  AutoModel,
  AutoProcessor,
  RawImage,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = true;
env.localModelPath = new URL("./models/", import.meta.url).href;
env.backends.onnx ??= {};
env.backends.onnx.wasm ??= {};
env.backends.onnx.wasm.wasmPaths = new URL("./vendor/ort/", import.meta.url).href;

const MODEL_ID = "Xenova/yolov9-c_all";
const SAMPLES = [
  { name: "街道", url: "./samples/street.jpg" },
  { name: "球场", url: "./samples/crowd.jpg" },
];

const els = {
  fileInput: document.getElementById("fileInput"),
  camBtn: document.getElementById("camBtn"),
  samples: document.getElementById("samples"),
  threshold: document.getElementById("threshold"),
  thrLabel: document.getElementById("thrLabel"),
  status: document.getElementById("status"),
  progressWrap: document.getElementById("progressWrap"),
  progressBar: document.getElementById("progressBar"),
  count: document.getElementById("count"),
  ms: document.getElementById("ms"),
  canvas: document.getElementById("canvas"),
  placeholder: document.getElementById("placeholder"),
  video: document.getElementById("video"),
  runtimeChip: document.getElementById("runtimeChip"),
};

const scratch = document.createElement("canvas");
const state = {
  model: null,
  processor: null,
  cameraOn: false,
  looping: false,
  sourceUrl: null,
  busy: false,
};

function setStatus(text) {
  els.status.textContent = text;
}

function yieldUI() {
  return new Promise((resolve) => setTimeout(resolve, 30));
}

function setProgress(ratio) {
  els.progressWrap.hidden = ratio <= 0 || ratio >= 1;
  els.progressBar.style.width = `${Math.round(Math.max(0, Math.min(1, ratio)) * 100)}%`;
}

function isPerson(label) {
  return String(label).toLowerCase() === "person";
}

function isRiderVehicle(label) {
  const name = String(label).toLowerCase();
  return name === "bicycle" || name === "motorcycle";
}

function riderBoxFromVehicle(box) {
  const width = box.xmax - box.xmin;
  const height = box.ymax - box.ymin;
  return {
    xmin: box.xmin + width * 0.2,
    xmax: box.xmax - width * 0.2,
    ymin: box.ymin,
    ymax: box.ymin + height * 0.58,
  };
}

function boxArea(box) {
  return Math.max(0, box.xmax - box.xmin) * Math.max(0, box.ymax - box.ymin);
}

function iou(a, b) {
  const x1 = Math.max(a.xmin, b.xmin);
  const y1 = Math.max(a.ymin, b.ymin);
  const x2 = Math.min(a.xmax, b.xmax);
  const y2 = Math.min(a.ymax, b.ymax);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = boxArea(a) + boxArea(b) - inter;
  return union <= 0 ? 0 : inter / union;
}

function nms(dets, iouThr = 0.5) {
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const keep = [];
  for (const det of sorted) {
    if (!keep.some((other) => iou(other.box, det.box) >= iouThr)) keep.push(det);
  }
  return keep;
}

function scaleBox(box, inW, inH, origW, origH) {
  const maxX = Math.max(box.xmin, box.xmax);
  const maxY = Math.max(box.ymin, box.ymax);
  if (maxX > inW * 1.15 || maxY > inH * 1.15) return box;
  return {
    xmin: (box.xmin / inW) * origW,
    ymin: (box.ymin / inH) * origH,
    xmax: (box.xmax / inW) * origW,
    ymax: (box.ymax / inH) * origH,
  };
}

function toPercent(box, width, height) {
  const maxV = Math.max(box.xmin, box.ymin, box.xmax, box.ymax);
  if (maxV <= 1.01) {
    return {
      xmin: box.xmin * 100,
      ymin: box.ymin * 100,
      xmax: box.xmax * 100,
      ymax: box.ymax * 100,
    };
  }
  return {
    xmin: (box.xmin / width) * 100,
    ymin: (box.ymin / height) * 100,
    xmax: (box.xmax / width) * 100,
    ymax: (box.ymax / height) * 100,
  };
}

function clampBox(box, width, height) {
  const p = toPercent(box, width, height);
  const xmin = Math.max(0, Math.min(100, p.xmin));
  const ymin = Math.max(0, Math.min(100, p.ymin));
  const xmax = Math.max(0, Math.min(100, p.xmax));
  const ymax = Math.max(0, Math.min(100, p.ymax));
  if (xmax - xmin < 0.8 || ymax - ymin < 0.8) return null;
  return { xmin, ymin, xmax, ymax };
}

function paint(imageEl, dets) {
  const w = imageEl.naturalWidth || imageEl.videoWidth || imageEl.width;
  const h = imageEl.naturalHeight || imageEl.videoHeight || imageEl.height;
  if (!w || !h) return;
  const canvas = els.canvas;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imageEl, 0, 0, w, h);
  ctx.lineWidth = Math.max(2, Math.round(w / 280));
  ctx.font = `${Math.max(14, Math.round(w / 42))}px ui-monospace, Consolas, sans-serif`;
  for (const d of dets) {
    const box = clampBox(d.box, w, h);
    if (!box) continue;
    const x = (box.xmin / 100) * w;
    const y = (box.ymin / 100) * h;
    const bw = ((box.xmax - box.xmin) / 100) * w;
    const bh = ((box.ymax - box.ymin) / 100) * h;
    ctx.strokeStyle = "#f0b429";
    ctx.fillStyle = "#f0b429";
    ctx.strokeRect(x, y, bw, bh);
    const tag = `行人 ${d.score.toFixed(2)}`;
    const tw = ctx.measureText(tag).width + 10;
    ctx.fillRect(x, Math.max(0, y - 22), tw, 22);
    ctx.fillStyle = "#101510";
    ctx.fillText(tag, x + 5, Math.max(16, y - 6));
  }
  els.placeholder.style.display = "none";
}

async function toRawImage(source) {
  if (typeof source === "string") {
    return RawImage.read(new URL(source, window.location.href).href);
  }
  if (source instanceof HTMLImageElement) {
    const canvas = document.createElement("canvas");
    canvas.width = source.naturalWidth || source.width;
    canvas.height = source.naturalHeight || source.height;
    canvas.getContext("2d").drawImage(source, 0, 0);
    return toRawImage(canvas);
  }
  if (source instanceof HTMLCanvasElement) {
    const blob = await new Promise((resolve) => source.toBlob(resolve, "image/jpeg", 0.95));
    return RawImage.read(URL.createObjectURL(blob));
  }
  throw new Error("不支持的图像输入");
}

function pedestriansFrom(raw, personThr, riderThr) {
  const people = nms(
    raw.filter((d) => isPerson(d.label) && d.score >= personThr),
    0.5
  );
  const extras = [];
  for (const vehicle of raw.filter((d) => isRiderVehicle(d.label) && d.score >= riderThr)) {
    const hasRider = people.some((person) => iou(person.box, vehicle.box) >= 0.05);
    if (hasRider) continue;
    extras.push({
      label: "person",
      score: vehicle.score,
      box: riderBoxFromVehicle(vehicle.box),
    });
  }
  return nms([...people, ...extras], 0.5);
}

async function detectRaw(source, scoreFloor) {
  const image = await toRawImage(source);
  const inputs = await state.processor(image);
  const result = await state.model(inputs);
  const tensor =
    result.outputs ||
    result.output0 ||
    Object.values(result).find((value) => value && typeof value.tolist === "function");
  let rows = tensor?.tolist ? tensor.tolist() : [];
  while (rows.length && Array.isArray(rows[0]) && Array.isArray(rows[0][0])) rows = rows[0];
  const dims = inputs.pixel_values?.dims;
  const inH = dims?.[2] || image.height;
  const inW = dims?.[3] || image.width;
  const raw = [];
  for (const row of rows) {
    if (!row || row.length < 6) continue;
    const [xmin, ymin, xmax, ymax, score, id] = row;
    if (Number(score) < scoreFloor) continue;
    const label =
      state.model.config.id2label?.[id] ??
      state.model.config.id2label?.[String(id)] ??
      state.model.config.id2label?.[String(Math.round(Number(id)))] ??
      (Math.round(Number(id)) === 0 ? "person" : "object");
    if (!isPerson(label) && !isRiderVehicle(label)) continue;
    raw.push({
      label: String(label),
      score: Number(score),
      box: scaleBox({ xmin, ymin, xmax, ymax }, inW, inH, image.width, image.height),
    });
  }
  return raw;
}

async function detect(source) {
  const personThr = Number(els.threshold.value);
  const riderThr = 0.3;
  const raw = await detectRaw(source, 0.15);
  return pedestriansFrom(raw, personThr, riderThr);
}

async function runOnUrl(url) {
  if (!state.model || state.busy) return;
  state.busy = true;
  setStatus("正在检测，大约 1–3 秒，页面可能短暂无响应。");
  await yieldUI();
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    const t0 = performance.now();
    const dets = await detect(url);
    const visible = dets.filter((d) => {
      const imgW = img.naturalWidth;
      const imgH = img.naturalHeight;
      return clampBox(d.box, imgW, imgH);
    });
    paint(img, visible);
    els.count.textContent = String(visible.length);
    els.ms.textContent = `${Math.round(performance.now() - t0)} ms`;
    setStatus(`检出 ${dets.length} 名行人。`);
  } catch (err) {
    console.error(err);
    setStatus(`推理失败：${err.message || err}`);
  } finally {
    state.busy = false;
  }
}

function grabCameraFrame(maxSide = 960) {
  const video = els.video;
  const sw = video.videoWidth;
  const sh = video.videoHeight;
  const scale = Math.min(1, maxSide / Math.max(sw, sh));
  scratch.width = Math.max(1, Math.round(sw * scale));
  scratch.height = Math.max(1, Math.round(sh * scale));
  scratch.getContext("2d").drawImage(video, 0, 0, scratch.width, scratch.height);
  return scratch;
}

async function runCameraFrame() {
  if (!state.model || !state.cameraOn || state.busy) return;
  state.busy = true;
  try {
    const frame = grabCameraFrame();
    const t0 = performance.now();
    const dets = await detect(frame);
    paint(frame, dets);
    els.count.textContent = String(dets.length);
    els.ms.textContent = `${Math.round(performance.now() - t0)} ms`;
    setStatus(`实时检测中 · ${dets.length} 人。关闭摄像头可停止。`);
  } finally {
    state.busy = false;
  }
}

async function cameraLoop() {
  if (state.looping) return;
  state.looping = true;
  while (state.cameraOn && state.model) {
    try {
      await runCameraFrame();
    } catch (err) {
      console.error(err);
      setStatus(`推理失败：${err.message || err}`);
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  state.looping = false;
}

async function loadModel() {
  setStatus("正在加载 YOLOv9，首次约需几秒到十几秒…");
  const model = await AutoModel.from_pretrained(MODEL_ID, {
    dtype: "q8",
    progress_callback: (info) => {
      if (info?.status === "progress" && info.total) {
        setProgress(info.loaded / info.total);
        setStatus(`读取权重 ${info.file || ""} ${(info.progress || 0).toFixed(0)}%`);
      }
    },
  });
  const processor = await AutoProcessor.from_pretrained(MODEL_ID);
  if (processor.feature_extractor) {
    processor.feature_extractor.size = { shortest_edge: 640 };
  } else if (processor.size) {
    processor.size = { shortest_edge: 640 };
  }
  state.model = model;
  state.processor = processor;
  setProgress(1);
  els.runtimeChip.textContent = navigator.gpu ? "WebGPU 可用" : "WASM";
  setStatus("模型就绪。点样例、上传图片，或打开摄像头。");
}

function stopCamera() {
  state.cameraOn = false;
  const stream = els.video.srcObject;
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
  }
  els.video.srcObject = null;
  els.camBtn.textContent = "打开摄像头";
}

SAMPLES.forEach((sample) => {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.title = sample.name;
  btn.style.backgroundImage = `url("${sample.url}")`;
  btn.addEventListener("click", async () => {
    stopCamera();
    state.sourceUrl = sample.url;
    if (!state.model) {
      setStatus("模型还在加载，请稍候再点样例。");
      return;
    }
    await runOnUrl(sample.url);
  });
  els.samples.appendChild(btn);
});

els.fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  stopCamera();
  const url = URL.createObjectURL(file);
  state.sourceUrl = url;
  if (state.model) await runOnUrl(url);
});

els.camBtn.addEventListener("click", async () => {
  if (state.cameraOn) {
    stopCamera();
    setStatus("摄像头已关闭。");
    return;
  }
  if (!state.model) {
    setStatus("模型还在加载，请稍候。");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 960 } },
      audio: false,
    });
    els.video.srcObject = stream;
    await els.video.play();
    state.cameraOn = true;
    els.camBtn.textContent = "关闭摄像头";
    cameraLoop();
  } catch (err) {
    setStatus(`无法打开摄像头：${err.message || err}`);
  }
});

els.threshold.addEventListener("input", async () => {
  els.thrLabel.textContent = Number(els.threshold.value).toFixed(2);
  if (state.busy || state.cameraOn || !state.sourceUrl || !state.model) return;
  await runOnUrl(state.sourceUrl);
});

loadModel().catch((err) => {
  console.error(err);
  setStatus(
    `加载失败：${err.message || err}。请用 http://127.0.0.1:8765 打开，并确认 models/ 下已有 YOLO 权重。`
  );
});

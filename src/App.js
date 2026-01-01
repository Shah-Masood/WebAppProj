import React, { useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

async function testLambda() {
  try {
    const url = process.env.REACT_APP_LAMBDA_URL;

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ from: "react", ts: Date.now() }),
    });

    const data = await res.json();
    console.log("Lambda response:", data);
    alert("Lambda says: " + JSON.stringify(data));
  } catch (e) {
    console.error(e);
    alert("Request failed: " + e.message);
  }
}

const ML_URL = process.env.REACT_APP_ML_URL;

export async function analyzeSkin(payload) {
  const res = await fetch(ML_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return await res.json();
}

export default function App() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const landmarkerRef = useRef(null);
  const rafRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);

  const [status, setStatus] = useState("Stopped");
  const [faces, setFaces] = useState(0);
  const [debug, setDebug] = useState("");
  const [scores, setScores] = useState({ lighting: 0, redness: 0, shine: 0 });

  <button onClick={testLambda}>
  Test Lambda
  </button>


  // 1) Load the FaceLandmarker once on mount
  useEffect(() => {
    let cancelled = false;

    async function initLandmarker() {
      try {
        setDebug("Loading FaceLandmarker…");

        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        const landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numFaces: 1,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
        });

        if (cancelled) return;
        landmarkerRef.current = landmarker;
        setDebug("FaceLandmarker loaded ✅");
      } catch (e) {
        console.error(e);
        setDebug(`FaceLandmarker load failed: ${e?.message || String(e)}`);
      }
    }

    initLandmarker();

    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      try {
        landmarkerRef.current?.close?.();
      } catch {}
    };
  }, []);

  // 2) Start camera
  async function startCamera() {
    try {
      setStatus("Starting camera…");
      setDebug("Requesting camera permission…");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });

      const video = videoRef.current;
      video.srcObject = stream;

      await new Promise((resolve) => {
        video.onloadedmetadata = () => resolve();
      });

      await video.play();
      setDebug(`Video ready ✅ (${video.videoWidth}x${video.videoHeight})`);
      setStatus("Running");
      runLoop();
    } catch (e) {
      console.error(e);
      setStatus("Stopped");
      setDebug(`Camera failed: ${e?.message || String(e)}`);
    }
  }

  function stopAll() {
    setStatus("Stopped");
    setFaces(0);
    setScores({ lighting: 0, redness: 0, shine: 0 });

    if (rafRef.current) cancelAnimationFrame(rafRef.current);

    const video = videoRef.current;
    const stream = video?.srcObject;
    if (stream && stream.getTracks) stream.getTracks().forEach((t) => t.stop());
    if (video) video.srcObject = null;

    // clear overlay
    const canvas = overlayRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    setDebug("Stopped.");
  }

  // 3) Main loop
  function runLoop() {
    const landmarker = landmarkerRef.current;
    const video = videoRef.current;

    if (!landmarker) {
      setDebug("FaceLandmarker not loaded yet (wait 1–2s)...");
      setStatus("Stopped");
      return;
    }
    if (!video) return;

    const step = () => {
      if (video.readyState >= 2 && video.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = video.currentTime;

        const nowMs = performance.now();
        const result = landmarker.detectForVideo(video, nowMs);

        const landmarks = result?.faceLandmarks || [];
        setFaces(landmarks.length);

        if (landmarks.length > 0) {
          drawAndScore(landmarks[0]); // one face MVP
        } else {
          clearOverlay();
          setScores({ lighting: 0, redness: 0, shine: 0 });
        }
      }
      rafRef.current = requestAnimationFrame(step);
    };

    rafRef.current = requestAnimationFrame(step);
  }

  function clearOverlay() {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function drawAndScore(lm) {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return;

    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.clearRect(0, 0, w, h);

    // Regions (simple, tweakable)
    const leftCheek = polyFrom(lm, [50, 187, 205, 36, 142, 126, 100, 47], w, h);
    const rightCheek = polyFrom(lm, [280, 411, 425, 266, 371, 355, 329, 277], w, h);
    const nose = polyFrom(lm, [1, 2, 98, 327, 168], w, h);

    // Draw ROI overlays
    drawPoly(ctx, leftCheek, "rgba(0,255,0,0.20)");
    drawPoly(ctx, rightCheek, "rgba(0,255,0,0.20)");
    drawPoly(ctx, nose, "rgba(0,255,0,0.20)");

    // Lighting (use cheeks combined)
    const lighting = lightingQualityFromPolys(ctx, [leftCheek, rightCheek]);

    let redness = 0;
    let shine = 0;
    const lightingOk = lighting >= 35;

    if (lightingOk) {
      redness = rednessFromPolys(ctx, [leftCheek, rightCheek]);
      shine = shineFromPolys(ctx, [nose]);
    }

    // HUD
    drawHUD(ctx, {
      lighting,
      redness,
      shine,
      lightingOk,
    });

    setScores({
      lighting: Math.round(lighting),
      redness: Math.round(redness),
      shine: Math.round(shine),
    });
  }

  const label = (v) => {
    if (v >= 75) return "High";
    if (v >= 45) return "Medium";
    if (v > 0) return "Low";
    return "—";
  };

  return (
    <div style={{ padding: 16, fontFamily: "Arial, sans-serif" }}>
      <h1 style={{ marginTop: 0 }}>SkinScan (MVP)</h1>

      <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
        <button onClick={startCamera} style={{ padding: "10px 14px" }}>
          Start Scan
        </button>
        <button onClick={stopAll} style={{ padding: "10px 14px" }}>
          Stop
        </button>

        <div>
          <div>
            <b>Status:</b> {status}
          </div>
          <div>
            <b>Faces:</b> {faces}
          </div>
        </div>
      </div>

      <div style={{ display: "flex", gap: 12, marginTop: 12, flexWrap: "wrap" }}>
        <ScoreCard title="Lighting" value={scores.lighting} note={scores.lighting < 35 ? "Too dark" : "OK"} />
        <ScoreCard title="Redness" value={scores.redness} note={label(scores.redness)} />
        <ScoreCard title="Shine/Oil" value={scores.shine} note={label(scores.shine)} />
      </div>

      <div style={{ marginTop: 16, position: "relative", maxWidth: 520 }}>
        <video
          ref={videoRef}
          playsInline
          muted
          style={{
            width: "100%",
            borderRadius: 18,
            background: "#000",
            transform: "scaleX(-1)", // selfie mirror for UX
          }}
        />
        <canvas
          ref={overlayRef}
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            width: "100%",
            height: "100%",
            pointerEvents: "none",
            transform: "scaleX(-1)", // match mirrored video
          }}
        />
      </div>

      <p style={{ color: "#666", marginTop: 12 }}>
        Tip: Use bright, even lighting. Avoid backlight. Keep your face centered.
      </p>

      <pre
        style={{
          background: "#f5f5f5",
          padding: 10,
          borderRadius: 10,
          fontSize: 12,
          overflowX: "auto",
        }}
      >
        {debug}
      </pre>
    </div>
  );
}

function ScoreCard({ title, value, note }) {
  return (
    <div style={{ border: "1px solid #333", borderRadius: 12, padding: 12, minWidth: 140 }}>
      <div style={{ fontSize: 14, opacity: 0.8 }}>{title}</div>
      <div style={{ fontSize: 28, fontWeight: 700 }}>{value}</div>
      <div style={{ fontSize: 13, opacity: 0.8 }}>{note}</div>
    </div>
  );
}

/** ---------- helpers ---------- **/

function toPx(p, w, h) {
  return { x: p.x * w, y: p.y * h };
}
function polyFrom(lm, idxs, w, h) {
  return idxs.map((i) => toPx(lm[i], w, h));
}
function drawPoly(ctx, poly, fillStyle) {
  if (!poly || poly.length < 3) return;
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(poly[0].x, poly[0].y);
  for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i].x, poly[i].y);
  ctx.closePath();
  ctx.fillStyle = fillStyle;
  ctx.fill();
  ctx.restore();
}
function drawHUD(ctx, { lighting, redness, shine, lightingOk }) {
  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.fillRect(12, 12, 290, 92);
  ctx.fillStyle = "white";
  ctx.font = "14px sans-serif";
  ctx.fillText(`Lighting: ${Math.round(lighting)} ${lightingOk ? "" : "(too dark)"}`, 22, 36);
  ctx.fillText(`Redness: ${Math.round(redness)}`, 22, 58);
  ctx.fillText(`Shine: ${Math.round(shine)}`, 22, 80);
  ctx.restore();
}

// Sample pixels from multiple polys, cheaply (bounding box + stride + point-in-poly)
function samplePolys(ctx, polys, maxSamples = 3000) {
  const pts = polys.flat();
  if (pts.length < 3) return [];

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }
  minX = Math.max(0, Math.floor(minX));
  minY = Math.max(0, Math.floor(minY));
  maxX = Math.min(ctx.canvas.width - 1, Math.ceil(maxX));
  maxY = Math.min(ctx.canvas.height - 1, Math.ceil(maxY));

  const w = maxX - minX + 1;
  const h = maxY - minY + 1;
  if (w <= 2 || h <= 2) return [];

  const img = ctx.getImageData(minX, minY, w, h).data;

  const totalPixels = w * h;
  const step = Math.max(1, Math.floor(Math.sqrt(totalPixels / maxSamples)));

  const out = [];
  for (let y = 0; y < h; y += step) {
    for (let x = 0; x < w; x += step) {
      const px = minX + x;
      const py = minY + y;

      // inside ANY of the polys
      let inside = false;
      for (const poly of polys) {
        if (pointInPoly({ x: px, y: py }, poly)) {
          inside = true;
          break;
        }
      }
      if (!inside) continue;

      const i = (y * w + x) * 4;
      out.push([img[i], img[i + 1], img[i + 2]]); // RGB
    }
  }
  return out;
}

function lightingQualityFromPolys(ctx, polys) {
  const rgb = samplePolys(ctx, polys, 2500);
  if (rgb.length < 80) return 0;

  let sumY = 0, sumY2 = 0;
  for (const [r, g, b] of rgb) {
    const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    sumY += y;
    sumY2 += y * y;
  }
  const mean = sumY / rgb.length;
  const variance = sumY2 / rgb.length - mean * mean;
  const std = Math.sqrt(Math.max(0, variance));

  const meanScore = clamp((mean / 255) * 100, 0, 100);
  const contrastScore = clamp((std / 64) * 100, 0, 100);
  return clamp(0.75 * meanScore + 0.25 * contrastScore, 0, 100);
}

function rednessFromPolys(ctx, polys) {
  const rgb = samplePolys(ctx, polys, 2500);
  if (rgb.length < 80) return 0;

  let acc = 0;
  for (const [r, g, b] of rgb) {
    const gb = (g + b) / 2;
    acc += (r - gb); // redness proxy
  }
  const mean = acc / rgb.length; // range roughly [-255, 255]
  // map to 0-100 (tune constants as needed)
  return clamp(((mean + 20) / 120) * 100, 0, 100);
}

function shineFromPolys(ctx, polys) {
  const rgb = samplePolys(ctx, polys, 2500);
  if (rgb.length < 80) return 0;

  // Shine proxy: high brightness but low colorfulness (specular highlights)
  let shiny = 0;
  for (const [r, g, b] of rgb) {
    const maxc = Math.max(r, g, b);
    const minc = Math.min(r, g, b);
    const v = maxc; // value-ish
    const sat = maxc === 0 ? 0 : (maxc - minc) / maxc; // 0..1

    // tweak these thresholds
    if (v > 210 && sat < 0.35) shiny++;
  }
  const frac = shiny / rgb.length; // 0..1
  return clamp(frac * 250, 0, 100); // scale up for sensitivity
}

function pointInPoly(pt, poly) {
  // ray-casting
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x, yi = poly[i].y;
    const xj = poly[j].x, yj = poly[j].y;

    const intersect =
      yi > pt.y !== yj > pt.y &&
      pt.x < ((xj - xi) * (pt.y - yi)) / (yj - yi + 1e-9) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function clamp(v, a, b) {
  return Math.max(a, Math.min(b, v));

}

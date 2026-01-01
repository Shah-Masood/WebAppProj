import React, { useEffect, useMemo, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";
import "./App.css";

const LIGHTING_OK_THRESHOLD = 55; // tweak to taste
const AUTO_CALL_MS = 2500;

function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

function stripDataUrlHeader(dataUrl) {
  // "data:image/jpeg;base64,AAAA..." -> "AAAA..."
  if (!dataUrl) return "";
  const idx = dataUrl.indexOf(",");
  return idx >= 0 ? dataUrl.slice(idx + 1) : dataUrl;
}

function Card({ title, value, sub }) {
  return (
    <div className="card">
      <div className="cardTitle">{title}</div>
      <div className="cardValue">{value}</div>
      <div className="cardSub">{sub}</div>
    </div>
  );
}

export default function App() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);

  const landmarkerRef = useRef(null);
  const rafRef = useRef(null);
  const lastAutoCallRef = useRef(0);

  const [running, setRunning] = useState(false);
  const [faces, setFaces] = useState(0);
  const [lighting, setLighting] = useState(0);

  // ML states
  const [mlWired, setMlWired] = useState(false);
  const [mlStatus, setMlStatus] = useState("Idle"); // Idle | Running | Done | Failed
  const [mlError, setMlError] = useState("");

  const [acneValue, setAcneValue] = useState("—");
  const [drynessValue, setDrynessValue] = useState("—");
  const [mlRednessValue, setMlRednessValue] = useState("—");

  const [debug, setDebug] = useState("");

  const ML_URL = useMemo(() => {
    const raw = process.env.REACT_APP_ML_URL || process.env.REACT_APP_LAMBDA_URL || "";
    return raw.trim();
  }, []);

  // Load FaceLandmarker once
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
      try {
        landmarkerRef.current?.close?.();
      } catch {}
      landmarkerRef.current = null;
    };
  }, []);

  // Start camera
  async function startScan() {
    setMlError("");
    setMlStatus("Running");
    setRunning(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      // kick loop
      loop();
    } catch (e) {
      console.error(e);
      setRunning(false);
      setMlStatus("Failed");
      setMlError(`Camera error: ${e?.message || String(e)}`);
    }
  }

  // Stop camera
  function stopScan() {
    setRunning(false);
    setMlStatus("Idle");

    // stop raf
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;

    // stop stream
    const v = videoRef.current;
    const stream = v?.srcObject;
    if (stream && stream.getTracks) {
      stream.getTracks().forEach((t) => t.stop());
    }
    if (v) v.srcObject = null;

    // clear overlay
    const canvas = overlayRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  function computeLightingFromVideoFrame(videoEl) {
    // sample a small downscaled frame for brightness
    const tmp = document.createElement("canvas");
    const w = 64;
    const h = 48;
    tmp.width = w;
    tmp.height = h;

    const ctx = tmp.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(videoEl, 0, 0, w, h);
    const img = ctx.getImageData(0, 0, w, h).data;

    let sum = 0;
    for (let i = 0; i < img.length; i += 4) {
      // luminance approx
      sum += 0.2126 * img[i] + 0.7152 * img[i + 1] + 0.0722 * img[i + 2];
    }
    const avg = sum / (img.length / 4);
    return Math.round(avg);
  }

  function drawNoseCheekGuides(ctx, w, h, landmarks) {
    // Simple green guide triangles near cheeks based on a few landmark points.
    // (This is lightweight “guide overlay”, not a medical thing.)
    // Indices are from MediaPipe face landmark topology.
    const leftCheek = landmarks[234];  // approx left cheek
    const rightCheek = landmarks[454]; // approx right cheek
    const noseTip = landmarks[1];      // approx nose tip

    if (!leftCheek || !rightCheek || !noseTip) return;

    ctx.save();
    ctx.globalAlpha = 0.6;
    ctx.fillStyle = "rgb(0, 255, 0)";

    const lx = leftCheek.x * w, ly = leftCheek.y * h;
    const rx = rightCheek.x * w, ry = rightCheek.y * h;
    const nx = noseTip.x * w, ny = noseTip.y * h;

    // left triangle
    ctx.beginPath();
    ctx.moveTo(nx, ny);
    ctx.lineTo(lx, ly);
    ctx.lineTo(lx, ly + 40);
    ctx.closePath();
    ctx.fill();

    // right triangle
    ctx.beginPath();
    ctx.moveTo(nx, ny);
    ctx.lineTo(rx, ry);
    ctx.lineTo(rx, ry + 40);
    ctx.closePath();
    ctx.fill();

    ctx.restore();
  }

  async function callLambdaWithCurrentFrame() {
    if (!ML_URL) {
      setMlStatus("Failed");
      setMlError("Missing REACT_APP_ML_URL env var (Amplify environment variables).");
      return;
    }

    const videoEl = videoRef.current;
    if (!videoEl) return;

    // draw current frame into a canvas and send
    const canvas = document.createElement("canvas");
    canvas.width = videoEl.videoWidth || 640;
    canvas.height = videoEl.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
    const image_b64 = stripDataUrlHeader(dataUrl);

    try {
      setMlError("");
      setMlStatus("Running");
      setMlWired(true);

      const res = await fetch(ML_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64 }),
      });

      const text = await res.text();
      let payload;
      try {
        payload = JSON.parse(text);
      } catch {
        payload = { ok: false, error: `Non-JSON response: ${text.slice(0, 120)}` };
      }

      if (!res.ok || payload?.ok === false) {
        setMlStatus("Failed");
        setMlError(payload?.error || `HTTP ${res.status}`);
        return;
      }

      // ✅ YOUR RESPONSE: { ok: true, acne_class: 3 }
      if (payload && typeof payload.acne_class !== "undefined") {
        setAcneValue(String(payload.acne_class));
      } else if (typeof payload.acne_prob !== "undefined") {
        // if you switch back to binary later
        setAcneValue(`${Math.round(payload.acne_prob * 100)}%`);
      }

      // placeholders until your backend returns these:
      if (typeof payload.dryness !== "undefined") setDrynessValue(String(payload.dryness));
      if (typeof payload.ml_redness !== "undefined") setMlRednessValue(String(payload.ml_redness));

      setMlStatus("Done");
    } catch (e) {
      console.error(e);
      setMlStatus("Failed");
      setMlError(e?.message || "Failed to fetch");
    }
  }

  async function loop() {
    if (!running) return;

    const videoEl = videoRef.current;
    const canvas = overlayRef.current;
    const landmarker = landmarkerRef.current;

    if (videoEl && canvas && landmarker && videoEl.readyState >= 2) {
      const w = videoEl.videoWidth;
      const h = videoEl.videoHeight;

      // fit overlay
      canvas.width = w;
      canvas.height = h;

      // update lighting
      const light = computeLightingFromVideoFrame(videoEl);
      setLighting(light);

      // detect face landmarks
      const nowMs = performance.now();
      const res = landmarker.detectForVideo(videoEl, nowMs);

      const numFaces = res?.faceLandmarks?.length || 0;
      setFaces(numFaces);

      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, w, h);

      if (numFaces > 0) {
        const drawingUtils = new DrawingUtils(ctx);
        // optional: draw a few connections
        drawingUtils.drawLandmarks(res.faceLandmarks[0], {
          radius: 1,
        });

        drawNoseCheekGuides(ctx, w, h, res.faceLandmarks[0]);
      }

      // AUTO CALL Lambda when stable conditions
      const lightingOk = light >= LIGHTING_OK_THRESHOLD;
      const faceOk = numFaces > 0;

      if (running && lightingOk && faceOk) {
        const t = Date.now();
        if (t - lastAutoCallRef.current >= AUTO_CALL_MS) {
          lastAutoCallRef.current = t;
          // don’t await to keep loop smooth
          callLambdaWithCurrentFrame();
        }
      }
    }

    rafRef.current = requestAnimationFrame(loop);
  }

  // Manual analyze button
  async function analyzeNow() {
    await callLambdaWithCurrentFrame();
  }

  const lightingOkLabel = lighting >= LIGHTING_OK_THRESHOLD ? "OK" : "Too dark";

  return (
    <div className="app">
      <h1>SkinScan (MVP)</h1>

      <div className="topRow">
        <button onClick={startScan} disabled={running}>Start Scan</button>
        <button onClick={stopScan} disabled={!running}>Stop</button>
        <button onClick={analyzeNow} disabled={!running}>Analyze Now</button>

        <div className="statusBlock">
          <div><b>Status:</b> {running ? "Running" : "Stopped"}</div>
          <div><b>Faces:</b> {faces}</div>
        </div>
      </div>

      <div className="grid">
        <Card title="Lighting" value={lighting} sub={lightingOkLabel} />
        <Card title="Redness" value={0} sub="—" />
        <Card title="Shine/Oil" value={0} sub="—" />

        <div className="card">
          <div className="cardTitle">ML Status</div>
          <div className="cardValue" style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span>{mlStatus}</span>
            <span>{mlStatus === "Done" ? "✅" : mlStatus === "Failed" ? "❌" : ""}</span>
          </div>
          <div className="cardSub">
            Lambda wired {ML_URL ? "✅" : "❌"}
          </div>
        </div>

        <Card title="Acne" value={acneValue} sub={acneValue === "—" ? "no result yet" : "from Lambda"} />
        <Card title="Dryness" value={drynessValue} sub={drynessValue === "—" ? "no result yet" : "from Lambda"} />
        <Card title="ML Redness" value={mlRednessValue} sub={mlRednessValue === "—" ? "no result yet" : "from Lambda"} />
      </div>

      {mlError ? <div className="error">ML error: {mlError}</div> : null}
      {debug ? <div className="debug">{debug}</div> : null}

      <div className="videoWrap">
        <video ref={videoRef} className="video" playsInline muted />
        <canvas ref={overlayRef} className="overlay" />
      </div>

      <div className="tip">
        Tip: Use bright, even lighting. Avoid backlight. Keep your face centered.
        <br />
        Live mode: when lighting is OK and a face is detected, the app auto-calls Lambda every ~2.5s.
      </div>
    </div>
  );
}

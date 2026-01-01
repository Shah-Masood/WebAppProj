import React, { useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

/* =======================
   Lambda test helper
======================= */
async function testLambda() {
  try {
    const url = process.env.REACT_APP_LAMBDA_URL;
    if (!url) throw new Error("REACT_APP_LAMBDA_URL not set");

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ from: "react", ts: Date.now() }),
    });

    const data = await res.json();
    alert("Lambda response:\n" + JSON.stringify(data, null, 2));
  } catch (e) {
    alert("Lambda request failed: " + e.message);
  }
}

/* =======================
   App
======================= */
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

  /* =======================
     Load FaceLandmarker
  ======================= */
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
        });

        if (!cancelled) {
          landmarkerRef.current = landmarker;
          setDebug("FaceLandmarker loaded ✅");
        }
      } catch (e) {
        setDebug("FaceLandmarker failed: " + e.message);
      }
    }

    initLandmarker();

    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      landmarkerRef.current?.close?.();
    };
  }, []);

  /* =======================
     Camera control
  ======================= */
  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });

      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      setStatus("Running");
      runLoop();
    } catch (e) {
      setDebug("Camera failed: " + e.message);
    }
  }

  function stopAll() {
    setStatus("Stopped");
    setFaces(0);
    setScores({ lighting: 0, redness: 0, shine: 0 });

    if (rafRef.current) cancelAnimationFrame(rafRef.current);

    const video = videoRef.current;
    const stream = video?.srcObject;
    stream?.getTracks?.().forEach((t) => t.stop());
    if (video) video.srcObject = null;
  }

  function runLoop() {
    const landmarker = landmarkerRef.current;
    const video = videoRef.current;
    if (!landmarker || !video) return;

    const step = () => {
      if (video.readyState >= 2 && video.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = video.currentTime;
        const result = landmarker.detectForVideo(video, performance.now());
        setFaces(result.faceLandmarks?.length || 0);
      }
      rafRef.current = requestAnimationFrame(step);
    };

    rafRef.current = requestAnimationFrame(step);
  }

  /* =======================
     UI
  ======================= */
  return (
    <div style={{ padding: 16, fontFamily: "Arial, sans-serif" }}>
      <h1>SkinScan (MVP)</h1>

      <div style={{ display: "flex", gap: 12 }}>
        <button onClick={startCamera}>Start Scan</button>
        <button onClick={stopAll}>Stop</button>
        <button onClick={testLambda}>Test Lambda</button>
      </div>

      <p>Status: {status}</p>
      <p>Faces detected: {faces}</p>

      <video
        ref={videoRef}
        muted
        playsInline
        style={{ width: 360, borderRadius: 12, background: "#000" }}
      />
      <canvas ref={overlayRef} />

      <pre
        style={{
          marginTop: 12,
          background: "#f5f5f5",
          padding: 10,
          borderRadius: 8,
          fontSize: 12,
        }}
      >
        {debug}
      </pre>
    </div>
  );
}

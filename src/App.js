import React, { useEffect, useRef, useState } from "react";
import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";

export default function App() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const detectorRef = useRef(null);
  const rafRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);

  const [status, setStatus] = useState("Stopped");
  const [faces, setFaces] = useState(0);
  const [debug, setDebug] = useState("");

  // 1) Load the detector once on mount
  useEffect(() => {
    let cancelled = false;

    async function initDetector() {
      try {
        setDebug("Loading face detector…");

        const vision = await FilesetResolver.forVisionTasks(
          // CDN hosts the wasm; easiest for CRA/Amplify
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        const detector = await FaceDetector.createFromOptions(vision, {
          baseOptions: {
            // Short-range face detector model
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
          },
          runningMode: "VIDEO",
        });

        if (cancelled) return;
        detectorRef.current = detector;
        setDebug("Detector loaded ✅");
      } catch (e) {
        console.error(e);
        setDebug(`Detector load failed: ${e?.message || String(e)}`);
      }
    }

    initDetector();

    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      try {
        detectorRef.current?.close?.();
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

      // Wait for metadata so videoWidth/videoHeight are valid
      await new Promise((resolve) => {
        video.onloadedmetadata = () => resolve();
      });

      await video.play();
      setDebug(`Video ready ✅ (${video.videoWidth}x${video.videoHeight})`);
      setStatus("Running");
      runDetectionLoop();
    } catch (e) {
      console.error(e);
      setStatus("Stopped");
      setDebug(`Camera failed: ${e?.message || String(e)}`);
    }
  }

  function stopAll() {
    setStatus("Stopped");
    setFaces(0);

    if (rafRef.current) cancelAnimationFrame(rafRef.current);

    const video = videoRef.current;
    const stream = video?.srcObject;
    if (stream && stream.getTracks) {
      stream.getTracks().forEach((t) => t.stop());
    }
    if (video) video.srcObject = null;

    setDebug("Stopped.");
  }

  // 3) Detection loop
  function runDetectionLoop() {
    const detector = detectorRef.current;
    const video = videoRef.current;

    if (!detector) {
      setDebug("Detector not loaded yet (wait 1–2s)...");
      setStatus("Stopped");
      return;
    }
    if (!video) return;

    const step = () => {
      // Only run when video has advanced
      if (video.readyState >= 2 && video.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = video.currentTime;

        const nowMs = performance.now();
        const result = detector.detectForVideo(video, nowMs);

        const detections = result?.detections || [];
        setFaces(detections.length);

        drawBoxes(detections);
      }
      rafRef.current = requestAnimationFrame(step);
    };

    rafRef.current = requestAnimationFrame(step);
  }

  function drawBoxes(detections) {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return;

    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);

    // Draw each bounding box
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#00ff00";
    ctx.font = "18px Arial";
    ctx.fillStyle = "#00ff00";

    detections.forEach((d, i) => {
      const box = d.boundingBox;
      if (!box) return;
      ctx.strokeRect(box.originX, box.originY, box.width, box.height);
      ctx.fillText(`Face ${i + 1}`, box.originX + 6, box.originY - 8);
    });
  }

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

      <div style={{ marginTop: 16, position: "relative", maxWidth: 520 }}>
        <video
          ref={videoRef}
          playsInline
          muted
          style={{
            width: "100%",
            borderRadius: 18,
            background: "#000",
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
          }}
        />
      </div>

      <p style={{ color: "#666", marginTop: 12 }}>
        Tip: Use bright, even lighting. Avoid backlight.
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

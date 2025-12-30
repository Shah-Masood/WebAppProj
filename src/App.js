import React, { useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const faceLandmarkerRef = useRef(null);
  const rafRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);

  const [status, setStatus] = useState("Ready");
  const [running, setRunning] = useState(false);
  const [faces, setFaces] = useState(0);

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function initFaceLandmarker() {
    if (faceLandmarkerRef.current) return faceLandmarkerRef.current;

    setStatus("Loading face model...");

    // Loads the WASM + assets for MediaPipe Tasks (hosted CDN)
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        // Model hosted on CDN (no backend)
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU", // will fall back if not available
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });

    faceLandmarkerRef.current = faceLandmarker;
    setStatus("Model loaded");
    return faceLandmarker;
  }

  async function start() {
    try {
      setStatus("Requesting camera...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 720 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current = stream;

      const video = videoRef.current;
      video.srcObject = stream;

      // On iOS Safari you need these
      video.playsInline = true;
      video.muted = true;

      // Wait for metadata then play
      await new Promise((res) => {
        video.onloadedmetadata = () => res();
      });

      // Try play once; ignore interruption warnings
      try {
        await video.play();
      } catch (_) {}

      // Init model
      await initFaceLandmarker();

      setRunning(true);
      setStatus("Detecting...");
      loop();
    } catch (e) {
      console.error(e);
      setStatus("Camera blocked or unavailable");
      setRunning(false);
    }
  }

  function stop() {
    setRunning(false);
    setFaces(0);
    setStatus("Stopped");

    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    lastVideoTimeRef.current = -1;

    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) track.stop();
      streamRef.current = null;
    }

    const video = videoRef.current;
    if (video) video.srcObject = null;

    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx && ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  function loop() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const faceLandmarker = faceLandmarkerRef.current;

    if (!video || !canvas || !faceLandmarker) return;
    if (!running) return;

    const ctx = canvas.getContext("2d");
    const drawingUtils = new DrawingUtils(ctx);

    // Match canvas to video
    const vw = video.videoWidth || 720;
    const vh = video.videoHeight || 720;
    if (canvas.width !== vw) canvas.width = vw;
    if (canvas.height !== vh) canvas.height = vh;

    const nowInMs = performance.now();

    // Only run detection when the video has advanced
    if (video.currentTime !== lastVideoTimeRef.current) {
      lastVideoTimeRef.current = video.currentTime;

      const result = faceLandmarker.detectForVideo(video, nowInMs);

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const detected = result?.faceLandmarks?.length || 0;
      setFaces(detected);

      if (detected > 0) {
        // draw mesh connections (face oval + lips + eyes + etc)
        for (const landmarks of result.faceLandmarks) {
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_TESSELATION,
            { lineWidth: 1 }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
            { lineWidth: 2 }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
            { lineWidth: 2 }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
            { lineWidth: 2 }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LIPS,
            { lineWidth: 2 }
          );
        }
      }
    }

    rafRef.current = requestAnimationFrame(loop);
  }

  return (
    <div style={{ padding: 16, fontFamily: "system-ui, Arial" }}>
      <h1 style={{ marginTop: 0 }}>SkinScan (MVP)</h1>

      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 12 }}>
        {!running ? (
          <button onClick={start} style={{ padding: "10px 14px" }}>
            Start Scan
          </button>
        ) : (
          <button onClick={stop} style={{ padding: "10px 14px" }}>
            Stop
          </button>
        )}

        <div>
          <div style={{ fontSize: 14 }}>
            <b>Status:</b> {status}
          </div>
          <div style={{ fontSize: 14 }}>
            <b>Faces:</b> {faces}
          </div>
        </div>
      </div>

      {/* Camera + overlay */}
      <div
        style={{
          position: "relative",
          width: "100%",
          maxWidth: 520,
          aspectRatio: "1 / 1",
          borderRadius: 18,
          overflow: "hidden",
          background: "#000",
        }}
      >
        <video
          ref={videoRef}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            objectFit: "cover",
            transform: "scaleX(-1)", // mirror for selfie mode
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            pointerEvents: "none",
            transform: "scaleX(-1)", // mirror overlay to match video
          }}
        />
      </div>

      <p style={{ marginTop: 12, color: "#666" }}>
        Tip: bright, even lighting. Avoid backlight.
      </p>
    </div>
  );
}

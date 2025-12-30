import React, { useRef, useState } from "react";

export default function App() {
  const videoRef = useRef(null);
  const [error, setError] = useState("");
  const [started, setStarted] = useState(false);

  const startCamera = async () => {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setStarted(true);
    } catch (e) {
      setError(e?.message || "Camera permission denied");
    }
  };

  const stopCamera = () => {
    const s = videoRef.current?.srcObject;
    if (s) s.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setStarted(false);
  };

  return (
    <div style={{ padding: 16, fontFamily: "system-ui" }}>
      <h2>SkinScan (MVP)</h2>

      {!started ? (
        <button onClick={startCamera} style={{ padding: "12px 16px", fontSize: 16 }}>
          Start Scan
        </button>
      ) : (
        <button onClick={stopCamera} style={{ padding: "12px 16px", fontSize: 16 }}>
          Stop
        </button>
      )}

      {error && <p style={{ color: "crimson" }}>{error}</p>}

      <div style={{ marginTop: 16 }}>
        <video
          ref={videoRef}
          playsInline
          muted
          style={{
            width: "100%",
            maxWidth: 420,
            borderRadius: 16,
            background: "#111",
          }}
        />
      </div>

      <p style={{ marginTop: 12, opacity: 0.7 }}>
        Tip: Use bright, even lighting. Avoid backlight.
      </p>
    </div>
  );
}


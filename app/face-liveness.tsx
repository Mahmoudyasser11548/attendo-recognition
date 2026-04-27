"use client";

import { useEffect, useEffectEvent, useRef, useState } from "react";
import * as faceapi from "face-api.js";

export default function FaceRecognition() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const animationRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const referenceDescriptorRef = useRef<Float32Array | null>(null);
  const blinkDoneRef = useRef(false);
  const turnDoneRef = useRef(false);

  const [referenceDescriptor, setReferenceDescriptor] =
    useState<Float32Array | null>(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Initializing...");
  const [blinkDone, setBlinkDone] = useState(false);
  const [turnDone, setTurnDone] = useState(false);

  const loadModels = async () => {
    const modelUrl = "/models";

    await faceapi.nets.tinyFaceDetector.loadFromUri(modelUrl);
    await faceapi.nets.faceLandmark68Net.loadFromUri(modelUrl);
    await faceapi.nets.faceRecognitionNet.loadFromUri(modelUrl);
  };

  const loadReferenceImage = async () => {
    const img = await faceapi.fetchImage("/reference.jpg");

    const detection = await faceapi
      .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection) {
      setStatus("Reference image face not found");
      return false;
    }

    referenceDescriptorRef.current = detection.descriptor;
    setReferenceDescriptor(detection.descriptor);
    setStatus("Reference image loaded");
    return true;
  };

  const startVideo = async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus("Camera not supported on this device");
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 240 },
          height: { ideal: 180 },
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;

        await new Promise<void>((resolve) => {
          videoRef.current!.onloadedmetadata = () => resolve();
        });

        await videoRef.current.play();
      }
    } catch (err) {
      console.error("Camera error:", err);
      setStatus("Failed to access camera");
    }
  };

  const getEyeRatio = (eye: faceapi.Point[]) => {
    const vertical =
      Math.abs(eye[1].y - eye[5].y) + Math.abs(eye[2].y - eye[4].y);
    const horizontal = Math.abs(eye[0].x - eye[3].x);

    return vertical / horizontal;
  };

  let isProcessing = false;

  // eslint-disable-next-line react-hooks/immutability
  const detect = useEffectEvent(async () => {
    if (isProcessing) return;
    // eslint-disable-next-line react-hooks/immutability
    isProcessing = true;

    try {
      if (!videoRef.current || !referenceDescriptorRef.current) {
        // eslint-disable-next-line react-hooks/immutability
        scheduleDetect(1000);
        return;
      }

      const detection = await faceapi
        .detectSingleFace(
          videoRef.current,
          new faceapi.TinyFaceDetectorOptions({
            inputSize: 128, // 🔥 lower = safer
            scoreThreshold: 0.6,
          }),
        )
        .withFaceLandmarks();

      if (!detection) {
        setStatus("Align your face inside the frame");
        scheduleDetect(1000);
        return;
      }

      const landmarks = detection.landmarks;

      // 👁️ Blink detection
      if (!blinkDoneRef.current) {
        setStatus("Blink your eyes");

        const eyeRatio =
          getEyeRatio(landmarks.getLeftEye()) +
          getEyeRatio(landmarks.getRightEye());

        if (eyeRatio > 0.85 && eyeRatio < 1.19) {
          blinkDoneRef.current = true;
          setBlinkDone(true);
          setProgress(50);
        }

        scheduleDetect(800);
        return;
      }

      // 🔄 Head turn detection
      if (!turnDoneRef.current) {
        setStatus("Turn your head slightly");

        const nose = landmarks.getNose();
        const jaw = landmarks.getJawOutline();
        const center = (jaw[0].x + jaw[16].x) / 2;
        const noseX = nose[3].x;

        if (Math.abs(noseX - center) > 20) {
          turnDoneRef.current = true;
          setTurnDone(true);
          setProgress(80);
        }

        scheduleDetect(800);
        return;
      }

      // 🧠 FINAL recognition (ONLY ONCE)
      setStatus("Verifying face...");

      const fullDetection = await faceapi
        .detectSingleFace(
          videoRef.current,
          new faceapi.TinyFaceDetectorOptions({
            inputSize: 128,
            scoreThreshold: 0.6,
          }),
        )
        .withFaceLandmarks()
        .withFaceDescriptor();
      console.log(fullDetection);
      if (!fullDetection) {
        scheduleDetect(1000);
        return;
      }

      const distance = faceapi.euclideanDistance(
        fullDetection.descriptor,
        referenceDescriptorRef.current,
      );
      console.log(distance);

      if (distance < 0.6) {
        setProgress(100);
        setStatus("Face recognized");
        return;
      }

      setStatus(`Face not recognized (${distance.toFixed(2)})`);
      scheduleDetect(1200);
    } catch (err) {
      console.error("Detect error:", err);
      setStatus("Detection error - retrying...");
      scheduleDetect(1500);
    } finally {
      isProcessing = false;
    }
  });

  const scheduleDetect = (delay = 1000) => {
    setTimeout(() => {
      // eslint-disable-next-line react-hooks/rules-of-hooks
      void detect();
    }, delay);
  };

  useEffect(() => {
    let isMounted = true;

    const start = async () => {
      try {
        setStatus("Loading face models...");
        await loadModels();
        if (!isMounted) return;

        setStatus("Preparing reference image...");
        const referenceLoaded = await loadReferenceImage();
        if (!isMounted || !referenceLoaded) return;

        setStatus("Starting camera...");
        await startVideo();
        if (!isMounted) return;

        setStatus("Camera ready. Align your face");
        setTimeout(() => {
          scheduleDetect(1000);
        }, 2000);
      } catch (error) {
        console.error(error);
        setStatus("Error starting face detection");
      }
    };

    void start();

    return () => {
      isMounted = false;

      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }

      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  return (
    <div className="h-screen bg-[#2E2A47] text-white flex flex-col items-center justify-center px-4">
      <h2 className="text-xl mb-4">Face Recognition</h2>

      <div className="w-[260px] h-[260px] rounded-2xl overflow-hidden border border-gray-500">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          disablePictureInPicture
          className="w-full h-full object-cover"
        />
      </div>

      <p className="mt-4 text-lg">{progress}%</p>

      <div className="w-4/5 h-2 bg-gray-600 rounded-full mt-2">
        <div
          className="h-full bg-green-400 rounded-full transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>

      <p className="mt-3 text-sm">{status}</p>

      <div className="mt-4 bg-[#3A365A] p-3 rounded-xl text-xs">
        Keep your face centered and look forward.
      </div>

      <div className="mt-2 text-xs text-gray-300">
        Reference ready: {referenceDescriptor ? "yes" : "no"} | Blink done:{" "}
        {blinkDone ? "yes" : "no"} | Turn done: {turnDone ? "yes" : "no"}
      </div>
    </div>
  );
}

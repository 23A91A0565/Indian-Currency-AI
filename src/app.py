"""
app.py — FastAPI Web Application & REST API
===========================================
Serves the mobile-responsive web interface and REST API endpoints
for Indian banknote classification.

Endpoints:
- GET  /                     : Serves the responsive single-page web UI
- GET  /api/health           : Service health, compute device, and loaded model info
- POST /api/predict          : Accepts uploaded image and returns denomination prediction
- GET  /api/sample/{denom}   : Returns a sample note image for quick testing
"""

import io
import os
import socket
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src.classifier import CurrencyClassifier

# Initialize FastAPI App
app = FastAPI(
    title="Indian Currency Classifier API",
    description="Mahatma Gandhi (New) Series Banknote Recognition System",
    version="2.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing) for mobile browsers & frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static frontend assets (HTML, CSS, JS)
FRONTEND_DIR = Path("frontend")
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Initialize Classifier instance
MODEL_PATH = "models/model_currency.pth"
classifier: Optional[CurrencyClassifier] = None

try:
    classifier = CurrencyClassifier(checkpoint_path=MODEL_PATH)
except Exception as e:
    print(f"[!] Primary checkpoint '{MODEL_PATH}' could not be loaded: {e}")


def get_local_ip() -> str:
    """
    Finds the machine's local IPv4 address so mobile devices on the same Wi-Fi
    can easily access the web application.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


@app.get("/")
async def serve_ui():
    """
    Serves the main single-page web application.
    """
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(str(index_file))


@app.get("/api/health")
async def health_check():
    """
    Returns service health status, compute hardware, and supported classes.
    """
    return {
        "status": "healthy",
        "device": str(classifier.device) if classifier else "not_loaded",
        "model_loaded": classifier is not None,
        "classes": classifier.classes if classifier else [],
        "supported_series": "Mahatma Gandhi (New) Series: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500"
    }


@app.post("/api/predict")
async def predict_currency(file: UploadFile = File(...)):
    """
    Accepts an uploaded banknote photo (from camera or file picker)
    and returns predicted denomination, confidence score, and probabilities.
    """
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please train the model with `python train.py` first."
        )

    # Read uploaded file bytes
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    # Run inference using the classifier
    prediction = classifier.predict(image)

    return JSONResponse(content={
        "denomination": prediction["denomination"],
        "confidence": prediction["confidence"],
        "raw_class": prediction["raw_class"],
        "name": prediction["name"],
        "details": prediction["details"],
        "probabilities": prediction["probabilities"],
    })


@app.get("/api/sample/{denomination}")
async def get_sample_image(denomination: str):
    """
    Returns a sample banknote image for the given denomination.
    Checks data/ directories first; falls back to an on-the-fly generated image.
    """
    clean_denom = denomination.replace("₹", "").strip()

    # Check for actual sample in data directories
    for candidate_dir in [Path("data") / clean_denom, Path("data/val") / clean_denom]:
        if candidate_dir.exists():
            images = list(candidate_dir.glob("*.jpg")) + list(candidate_dir.glob("*.png"))
            if images:
                with open(images[0], "rb") as f:
                    return Response(content=f.read(), media_type="image/jpeg")

    # Fallback palette for synthetic preview note
    PALETTE = {
        "10": (123, 75, 42),
        "20": (195, 200, 60),
        "50": (30, 160, 205),
        "100": (168, 145, 192),
        "200": (230, 175, 35),
        "500": (128, 138, 135),
    }
    rgb_color = PALETTE.get(clean_denom, (120, 120, 120))
    canvas = np.full((220, 500, 3), rgb_color, dtype=np.uint8)

    # Draw border and denomination text
    cv2.rectangle(canvas, (10, 10), (490, 210), (255, 255, 255), 2)
    cv2.putText(
        canvas,
        f"INR {clean_denom}",
        (160, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3
    )

    success, buffer = cv2.imencode(".jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode sample image.")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")

# RupeeLens: Indian Currency Recognition System 🇮🇳

A modular, clean, and beginner-friendly deep learning system for recognizing Mahatma Gandhi (New) Series Indian banknotes with mobile camera support, interactive FastAPI web interface, and PyTorch ResNet-34 transfer learning.

---

## 📌 Target Denominations
Strictly the **Mahatma Gandhi (New) Series** active legal tender:
- **₹10** (Chocolate Brown, Konark Sun Temple)
- **₹20** (Greenish Yellow, Ellora Caves)
- **₹50** (Fluorescent Blue, Hampi Chariot)
- **₹100** (Lavender, Rani Ki Vav, 4 bleed lines)
- **₹200** (Bright Yellow, Sanchi Stupa, 4 bleed lines + 2 circles)
- **₹500** (Stone Grey, Red Fort, 5 bleed lines)

*Note: Pre-2016 (₹1000) and withdrawn (₹2000) notes are explicitly excluded.*

---

## 📁 Project Structure

```
indian-currency-classifier/
├── src/                        # Core Python Application Modules
│   ├── __init__.py             # Package exports
│   ├── classifier.py           # PyTorch ResNet-34 inference engine
│   ├── denominations.py        # RBI official banknote metadata & specifications
│   ├── training.py             # Transfer learning training pipeline
│   ├── data_prep.py            # Dataset organization & sorting logic
│   └── app.py                  # FastAPI application, REST endpoints, and static mounts
├── frontend/                   # Web Interface
│   ├── index.html              # Responsive single-page web interface
│   ├── css/
│   │   └── style.css           # Modern dark-mode styling & animations
│   └── js/
│       └── app.js              # Camera capture, client-side optimizer, & result rendering
├── models/                     # Model weights
│   ├── README.md
│   └── model_currency.pth      # Trained ResNet-34 weights (Mahatma Gandhi Series)
├── data/                       # Dataset organized by denomination class (10, 20, 50, 100, 200, 500)
├── predict.py                  # CLI entry point for single-image prediction
├── train.py                    # CLI entry point for model training
├── server.py                   # CLI entry point for web application server
├── prepare_data.py             # CLI entry point for dataset organization
├── test_pipeline.py            # Automated test pipeline verifying training & inference
└── requirements.txt            # Project dependencies
```

---

## 🚀 Quickstart Commands

### 1. Verification Test (`test_pipeline.py`)
Run the end-to-end automated test to verify that PyTorch, torchvision, and the classifier inference are working properly:

```bash
python test_pipeline.py
```

### 2. CLI Banknote Inference (`predict.py`)
Run deep learning classification on an input banknote photo:

```bash
python predict.py --image path/to/banknote.jpg
```

Example output:
```
=======================================================
 🇮🇳 INDIAN CURRENCY RECOGNITION RESULT
=======================================================
  Denomination : ₹500 (Five Hundred Rupees)
  Confidence   : 99.97%
  Base Color   : Stone Grey
  Motif        : Red Fort with Indian flag
  Dimensions   : 66 mm x 150 mm
  Bleed Lines  : 5 angular bleed lines on both margins
-------------------------------------------------------
  Class Probabilities:
    ₹10   :   0.0% | 
    ₹20   :   0.0% | 
    ₹50   :   0.0% | 
    ₹100  :   0.0% | 
    ₹200  :   0.0% | 
    ₹500  : 100.0% | ████████████████████████
=======================================================
```

### 3. Web Application & API Server (`server.py`)
Launches the FastAPI backend and single-page mobile interface on port 8000:

```bash
python server.py
```

The server binds to `0.0.0.0:8000` so mobile phones on the same network can access it.

### 4. Dataset Auto-Organization (`prepare_data.py`)
Scans downloaded currency images (e.g., from Kaggle or Mendeley) and organizes them into `data/<class>`:

```bash
python prepare_data.py --source raw_data/"Indian currency" --target data
```

### 5. Model Training Pipeline (`train.py`)
Fine-tunes a PyTorch **ResNet-34** backbone with transfer learning, data augmentations (RandomRotation ±15°, ColorJitter), CrossEntropyLoss, and Adam optimizer:

```bash
python train.py --epochs 5 --batch-size 16 --lr 0.0001 --output models/model_currency.pth
```

---

## 📱 Mobile Camera Access Setup

1. **Connect to the same Wi-Fi**: Make sure your mobile phone and PC are connected to the same local network / Wi-Fi.
2. **Find PC's Local IP**: The server prints your local IP on startup (e.g., `http://192.168.1.15:8000`).
3. **Open on Mobile**: Open Safari or Chrome on your smartphone and navigate to:
   ```
   http://<YOUR_PC_IP>:8000
   ```
4. **Direct Camera Capture**: Tap **"Snap Camera"** to trigger your smartphone's camera viewfinder directly (`capture="environment"`).
5. **Instant Results**: RupeeLens analyzes the image in real-time, displaying the identified denomination, confidence score, official RBI banknote specifications, and class probability distribution.

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Responsive Mobile Web Interface |
| `POST` | `/api/predict` | Multipart image upload for banknote classification |
| `GET` | `/api/sample/{denomination}` | Quick test demo image (`10`, `20`, `50`, `100`, `200`, `500`) |
| `GET` | `/api/health` | System status, device info, and active classes |
| `GET` | `/docs` | Interactive Swagger API documentation |
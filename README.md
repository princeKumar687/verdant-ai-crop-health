# Verdant AI — AI-Based Crop Disease Advisory System

A responsive Flask web app with a premium agriculture dashboard, leaf-image upload, TensorFlow model inference, and educational crop-care guidance.

## What is included
- Responsive dashboard UI (HTML/CSS/JavaScript)
- Flask backend with image validation and `/api/analyze`
- TensorFlow inference hook for a trained Keras classifier
- MobileNetV2 training script for a folder-based crop-leaf dataset
- Rule-based advisory text for common tomato and potato classes
- In-session recent scan history in the browser

## Important: AI model requirement
The website code is complete, but **a trained model is not bundled**. Disease detection becomes active only after you train a model and place these files in `models/`:
- `crop_disease_model.keras`
- `class_names.json`

Without them, the app still runs and the dashboard explains that model setup is required; it will not pretend to detect a disease.

## 1. Requirements
- Python 3.10 or 3.11 recommended for compatible TensorFlow builds
- Internet access for the first installation and ImageNet MobileNetV2 weights
- A licensed crop leaf image dataset. PlantVillage is a common educational starting point; field photos can differ from clean dataset images.

## 2. Install and run
Open a terminal in this project folder.

### Windows (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

### macOS / Linux
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

If TensorFlow installation fails, check TensorFlow's current official compatibility instructions for your OS/Python version.

## 3. Train the model
Download and extract a licensed folder-organized dataset so each disease class has its own folder, for example:
```
dataset/
  Tomato___Early_blight/
  Tomato___Late_blight/
  Tomato___healthy/
  Potato___Early_blight/
  Potato___Late_blight/
  Potato___healthy/
```
The class folder names become the model labels. Then run:
```bash
python train_model.py
```
On Windows you can optionally choose a dataset path:
```powershell
$env:CROP_DATA_DIR="C:\path\to\dataset"
python train_model.py
```
The script writes the model and class names into `models/`. Restart `python app.py` after training.

**Training note:** the starter script uses a frozen ImageNet MobileNetV2 feature extractor and a small classification head. For a real deployment, evaluate held-out field images, review per-class precision/recall, tune thresholds, and consider fine-tuning. High accuracy on a clean benchmark does not guarantee reliable field diagnosis.

## 4. API
`POST /api/analyze` with multipart form field `image` returns JSON with crop, predicted label, confidence, advisory actions, timestamp and uploaded image URL.

`GET /api/status` returns whether a model loaded successfully.

## 5. Before deployment
- Replace Flask's development server with a production WSGI server.
- Add authentication, rate limits, CSRF/security hardening, retention rules and consent if saving user images.
- Configure upload storage and HTTPS.
- Do not expose debug mode in production.
- Review all agronomic advice with a qualified local agricultural expert.
- Do not recommend or apply pesticides solely from this model. Follow local labels and extension guidance.

## Limitations
This is a student-project starter, not a validated diagnostic product. Image predictions can be wrong due to lighting, crop variety, mixed symptoms, nutrient deficiencies, pests, or diseases absent from the training set. Confirm uncertain or high-impact cases with a local agriculture professional.

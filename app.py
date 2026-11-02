import os, json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
ALLOWED = {"png", "jpg", "jpeg", "webp"}
MODEL_PATH = os.path.join("models", "crop_disease_model.keras")
CLASS_NAMES_PATH = os.path.join("models", "class_names.json")

# Advisory text is general educational guidance, not a pesticide prescription.
ADVISORIES = {
    "Pepper__bell___Bacterial_spot": {
        "crop": "Pepper",
        "disease": "Bacterial Spot",
        "severity": "Moderate",
        "symptoms": "Small dark, water-soaked spots on leaves and fruits.",
        "actions": [
            "Remove severely affected leaves carefully.",
            "Avoid overhead watering and keep leaves dry.",
            "Maintain proper spacing and airflow.",
            "Consult a local agricultural expert for approved treatment."
        ]
    },

    "Pepper__bell___healthy": {
        "crop": "Pepper",
        "disease": "Healthy",
        "severity": "Low",
        "symptoms": "No visible signs of the supported pepper diseases.",
        "actions": [
            "Continue regular crop monitoring.",
            "Maintain balanced irrigation and nutrition."
        ]
    },

    "Potato___Early_blight": {
        "crop": "Potato",
        "disease": "Early Blight",
        "severity": "Moderate",
        "symptoms": "Dark leaf spots with concentric rings, usually on older leaves.",
        "actions": [
            "Remove affected plant debris.",
            "Avoid overhead irrigation.",
            "Maintain proper spacing and airflow.",
            "Seek local agricultural guidance for approved treatment."
        ]
    },

    "Potato___Late_blight": {
        "crop": "Potato",
        "disease": "Late Blight",
        "severity": "High",
        "symptoms": "Dark, water-soaked lesions that can spread rapidly.",
        "actions": [
            "Isolate severely affected plants where practical.",
            "Avoid working among wet plants.",
            "Seek prompt advice from a local agricultural expert."
        ]
    },

    "Potato___healthy": {
        "crop": "Potato",
        "disease": "Healthy",
        "severity": "Low",
        "symptoms": "No visible signs of the supported potato diseases.",
        "actions": [
            "Continue regular crop monitoring.",
            "Use healthy seed tubers.",
            "Maintain balanced irrigation and nutrition."
        ]
    },

    "Tomato_Bacterial_spot": {
        "crop": "Tomato",
        "disease": "Bacterial Spot",
        "severity": "Moderate",
        "symptoms": "Small dark spots on leaves and fruits.",
        "actions": [
            "Remove badly affected plant material.",
            "Avoid overhead watering.",
            "Keep foliage dry and improve airflow.",
            "Consult a local agricultural expert."
        ]
    },

    "Tomato_Early_blight": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "severity": "Moderate",
        "symptoms": "Brown spots with concentric rings, often on older leaves.",
        "actions": [
            "Remove badly affected leaves.",
            "Avoid overhead irrigation.",
            "Improve spacing and airflow.",
            "Rotate crops where possible."
        ]
    },

    "Tomato_Late_blight": {
        "crop": "Tomato",
        "disease": "Late Blight",
        "severity": "High",
        "symptoms": "Fast-spreading dark, water-soaked patches on leaves.",
        "actions": [
            "Remove severely affected plant material where practical.",
            "Avoid working among wet plants.",
            "Seek prompt local agricultural advice."
        ]
    },

    "Tomato_Leaf_Mold": {
        "crop": "Tomato",
        "disease": "Leaf Mold",
        "severity": "Moderate",
        "symptoms": "Yellow patches on upper leaf surfaces with olive-green mold underneath.",
        "actions": [
            "Improve ventilation around plants.",
            "Avoid excessive humidity.",
            "Remove severely affected leaves.",
            "Avoid wetting foliage during irrigation."
        ]
    },

    "Tomato_Septoria_leaf_spot": {
        "crop": "Tomato",
        "disease": "Septoria Leaf Spot",
        "severity": "Moderate",
        "symptoms": "Small circular spots with dark borders and pale centers.",
        "actions": [
            "Remove infected lower leaves.",
            "Avoid overhead watering.",
            "Remove plant debris after harvest.",
            "Maintain proper plant spacing."
        ]
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "crop": "Tomato",
        "disease": "Two-Spotted Spider Mites",
        "severity": "Moderate",
        "symptoms": "Tiny yellow or white speckles on leaves and possible fine webbing.",
        "actions": [
            "Inspect the undersides of leaves.",
            "Remove heavily affected leaves where practical.",
            "Monitor plants regularly for mite activity.",
            "Seek expert advice for suitable control methods."
        ]
    },

    "Tomato__Target_Spot": {
        "crop": "Tomato",
        "disease": "Target Spot",
        "severity": "Moderate",
        "symptoms": "Brown circular lesions with concentric rings on leaves.",
        "actions": [
            "Remove infected leaves where practical.",
            "Improve airflow around plants.",
            "Avoid overhead irrigation.",
            "Monitor nearby plants for symptoms."
        ]
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "crop": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "severity": "High",
        "symptoms": "Yellowing, upward curling leaves and stunted plant growth.",
        "actions": [
            "Remove severely affected plants where appropriate.",
            "Monitor whitefly populations.",
            "Control weeds around the crop.",
            "Consult a local agricultural expert."
        ]
    },

    "Tomato__Tomato_mosaic_virus": {
        "crop": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "severity": "High",
        "symptoms": "Mosaic-like light and dark green patterns on leaves.",
        "actions": [
            "Remove infected plants where appropriate.",
            "Disinfect gardening tools.",
            "Wash hands after handling infected plants.",
            "Use disease-free seeds and seedlings."
        ]
    },

    "Tomato_healthy": {
        "crop": "Tomato",
        "disease": "Healthy",
        "severity": "Low",
        "symptoms": "No visible signs of the supported tomato diseases.",
        "actions": [
            "Continue regular crop monitoring.",
            "Maintain balanced irrigation and nutrition.",
            "Keep the growing area clean."
        ]
    }

    
    
}


def load_model():
    try:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
                class_names = json.load(f)
            return model, class_names
    except Exception as exc:
        app.logger.warning("Model could not be loaded: %s", exc)
    return None, []

MODEL, CLASS_NAMES = load_model()

def get_advisory(class_name):
    if class_name in ADVISORIES:
        return ADVISORIES[class_name]
    parts = class_name.split("___", 1)
    crop = parts[0].replace("_", " ") if parts else "Unknown crop"
    disease = parts[1].replace("_", " ") if len(parts) > 1 else class_name
    healthy = "healthy" in disease.lower()
    return {
        "crop": crop, "disease": "No disease detected" if healthy else disease,
        "severity": "Low" if healthy else "Needs review",
        "symptoms": "The model's class label is shown below. Confirm the result against field symptoms.",
        "actions": ["Compare the result with visible symptoms on several plants.",
                    "Take clear photos in natural light and rescan if the result seems wrong.",
                    "Consult a local agricultural extension officer before treatment decisions."]
    }

@app.route("/")
def index():
    return render_template("index.html", model_ready=MODEL is not None)

@app.route("/api/status")
def status():
    return jsonify({"model_ready": MODEL is not None, "classes": len(CLASS_NAMES)})

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Choose a leaf image first."}), 400
    file = request.files["image"]
    if not file.filename or "." not in file.filename:
        return jsonify({"error": "Please select a PNG, JPG, JPEG or WEBP image."}), 400
    ext = file.filename.rsplit(".", 1)[1].lower()
    if ext not in ALLOWED:
        return jsonify({"error": "Unsupported file type. Use PNG, JPG, JPEG or WEBP."}), 400
    safe_name = secure_filename(file.filename)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    saved_name = f"{stamp}_{safe_name}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)
    try:
        file.save(path)
        img = Image.open(path).convert("RGB")
        if MODEL is None:
            return jsonify({
                "error": "The AI model is not installed yet. Follow README.md to train a model and place it in the models/ folder.",
                "model_ready": False
            }), 503
        import tensorflow as tf
        image = img.resize((224, 224))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        probs = MODEL.predict(np.expand_dims(arr, 0), verbose=0)[0]
        idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
        confidence = float(probs[idx]) * 100
        if confidence < 55:
            result = {
                "crop": label.split("___")[0].replace("_", " "),
                "disease": "Uncertain result — rescan recommended",
                "severity": "Uncertain",
                "symptoms": "The model confidence is low. This image may be unclear or outside supported classes.",
                "actions": ["Take a sharp photo of one affected leaf in natural daylight.",
                            "Include both healthy and affected areas without fingers covering the leaf.",
                            "Ask a local agricultural expert to confirm before taking action."]
            }
        else:
            result = get_advisory(label)
        result.update({
            "confidence": round(confidence, 1),
            "label": label,
            "image_url": "/" + path.replace("\\", "/"),
            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "disclaimer": "AI prediction is an aid, not a confirmed diagnosis. Verify with a qualified local agriculture professional."
        })
        return jsonify(result)
    except Exception as exc:
        app.logger.exception("Image analysis failed")
        return jsonify({"error": "Could not analyze this image. Please try another clear image."}), 400
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip().lower().replace("bright", "blight")
    if not message:
        return jsonify({"reply": "Please apna sawal likho."}), 400

    reply = (
        "Main Verdant AI Crop Doctor hoon. "
        "Pehle Disease Detection section mein leaf ki photo analyze karo. "
        "Phir disease ka naam yahan likho, main available crop-care "
        "guidance bataunga."
    )

    for key, advice in ADVISORIES.items():
        crop = advice.get("crop", "").lower()
        disease = advice.get("disease", "").lower()

        if key.lower() in message or crop in message or disease in message:
            actions = advice.get("actions", [])
            reply = (
                f"Crop: {advice.get('crop', 'Unknown')}\n"
                f"Possible disease: {advice.get('disease', 'Unknown')}\n"
                f"Severity: {advice.get('severity', 'Unknown')}\n"
                f"Symptoms: {advice.get('symptoms', 'Not available')}\n\n"
                "Care suggestions:\n- " + "\n- ".join(actions)
            )
            break

    return jsonify({"reply": reply})
@app.route("/farm-registry", methods=["GET", "POST"])
def farm_registry():
    import sqlite3

    conn = sqlite3.connect("farm_registry.db")
    conn.row_factory = sqlite3.Row

    conn.execute("""
        CREATE TABLE IF NOT EXISTS farm_activities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop TEXT NOT NULL,
            activity TEXT NOT NULL,
            activity_date TEXT NOT NULL,
            location TEXT NOT NULL,
            details TEXT
        )
    """)

    if request.method == "POST":
        crop = request.form.get("crop", "").strip()
        activity = request.form.get("activity", "").strip()
        activity_date = request.form.get("date", "").strip()
        location = request.form.get("location", "").strip()
        details = request.form.get("details", "").strip()

        if crop and activity and activity_date and location:
            conn.execute("""
                INSERT INTO farm_activities
                (crop, activity, activity_date, location, details)
                VALUES (?, ?, ?, ?, ?)
            """, (crop, activity, activity_date, location, details))
            conn.commit()

    records = conn.execute("""
        SELECT * FROM farm_activities
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return render_template("farm_registry.html", records=records)
if __name__ == "__main__":
    app.run(debug=True, port=5000)


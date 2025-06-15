import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
import requests
from werkzeug.utils import secure_filename
from flask_cors import CORS

# --- CONFIGURATION ---
LEVEL1_MODEL_PATH = "best_fish_model.pth"
LEVEL2_MODELS_DIR = "fish_models/level2/"
LEVEL1_CLASSES_PATH = "level1_classes.json"
LEVEL2_CLASSES_DIR = "level2_species_classes/"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FLASK INIT ---
app = Flask(__name__)
CORS(app)  # Add this line
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- TRANSFORMS ---
COMMON_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

LEVEL1_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- UTILS ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_model(model_type, num_classes, model_path, pretrained=True):
    if model_type == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif model_type == "resnet18":
        model = models.resnet18(weights=None)
    else:
        raise ValueError("Unknown model type")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def predict(model, img_tensor):
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return pred.item(), conf.item()

def get_inaturalist_info(species_name):
    url = "https://api.inaturalist.org/v1/search"
    params = {
        "q": species_name,
        "sources": "taxa"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])
        for result in results:
            if result.get("record", {}).get("name", "").lower() == species_name.lower():
                record = result["record"]
                return {
                    "name": record.get("name"),
                    "common_name": record.get("preferred_common_name"),
                    "description": record.get("wikipedia_summary"),
                    "image": record.get("default_photo", {}).get("medium_url"),
                    "rank": record.get("rank"),
                    "taxonomy": [a.get("name") for a in record.get("ancestors", [])]
                }
        return {"message": "No exact match found in iNaturalist"}
    except Exception as e:
        return {"error": str(e)}

# --- MODELS INIT ---
LEVEL1_CLASSES = load_json(LEVEL1_CLASSES_PATH)
level1_model = load_model("resnet50", len(LEVEL1_CLASSES), LEVEL1_MODEL_PATH, pretrained=True)

# --- PREDICTION FUNCTION ---
def predict_fish_species(image_path):
    img = Image.open(image_path).convert("RGB")
    img_level1 = LEVEL1_TRANSFORM(img)

    family_index, conf_family = predict(level1_model, img_level1)
    predicted_family = LEVEL1_CLASSES[family_index]

    # Load species classes
    species_classes_path = os.path.join(LEVEL2_CLASSES_DIR, f"level2_{predicted_family}_species.json")
    level2_model_path = os.path.join(LEVEL2_MODELS_DIR, f"level2_{predicted_family}.pth")

    if os.path.exists(species_classes_path) and os.path.exists(level2_model_path):
        species_classes = load_json(species_classes_path)
        if len(species_classes) >= 2:
            img_level2 = COMMON_TRANSFORM(img)
            level2_model = load_model("resnet18", len(species_classes), level2_model_path, pretrained=False)
            species_index, conf_species = predict(level2_model, img_level2)
            predicted_species = species_classes[species_index]
            return {
                "family": predicted_family,
                "confidence_family": round(conf_family * 100, 2),
                "species": predicted_species,
                "confidence_species": round(conf_species * 100, 2)
            }
        else:
            return {
                "family": predicted_family,
                "confidence_family": round(conf_family * 100, 2),
                "species": species_classes[0],
                "confidence_species": 100.0
            }
    else:
        return {
            "family": predicted_family,
            "confidence_family": round(conf_family * 100, 2),
            "species": "Unknown",
            "confidence_species": 0.0
        }

# --- ROUTES ---
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    # Vérifie si un fichier image a été envoyé dans la requête
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400  # Erreur 400 = Bad Request

    file = request.files["image"]

    # Vérifie si un fichier a été sélectionné par l'utilisateur
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Vérifie que le fichier est autorisé (extension valide)
    if file and allowed_file(file.filename):
        # Sécurise le nom du fichier pour éviter les attaques (injection de chemins)
        filename = secure_filename(file.filename)
        # Crée le chemin complet pour enregistrer l'image temporairement
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)  # Sauvegarde le fichier sur le serveur

        # Appel à la fonction de prédiction (modèle CNN)
        prediction = predict_fish_species(filepath)
        # Récupère des informations complémentaires depuis iNaturalist
        inat_info = get_inaturalist_info(prediction["species"])

        # Retourne les résultats au format JSON
        return jsonify({
            "prediction": prediction,  # Résultat du modèle (espèce, famille, confiance)
            "inaturalist_info": inat_info  # Métadonnées iNaturalist (description, habitat)
        })

    # Si le fichier n'a pas une extension autorisée (.jpg, .png, etc.)
    return jsonify({"error": "Invalid file type"}), 400

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Fish Classification API is running."})

# --- MAIN ---
if __name__ == "__main__":
    app.run(debug=True, port=5000)


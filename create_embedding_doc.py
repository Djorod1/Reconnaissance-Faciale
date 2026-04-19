import os
import torch
import numpy as np
from app.model import load_facenet_model
from app.face_utils import read_image_from_bytes, preprocess_face, get_embedding
from app.face_detect import extract_faces
from app.db import load_embedding

# ---------------------------
# CONFIGURATION GLOBALE
# ---------------------------

# Chemin du modèle entraîné (FaceNet fine-tuné sur un dataset africain)
MODEL_PATH = "facenet_africain_finetuned.pth"

# Dossier où sont stockées les embeddings des visages enregistrés
EMBEDDINGS_DIR = "embeddings"

# Choix du périphérique : GPU (cuda) si disponible, sinon CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Seuil de distance pour considérer qu'un visage est reconnu
# Plus petit = plus strict (moins d'erreurs mais risque de "non reconnu")
THRESHOLD = 0.8

# ---------------------------
# FONCTIONS PRINCIPALES
# ---------------------------

def load_model(path):
    """
    Charge le modèle FaceNet fine-tuné depuis un fichier .pth
    et le met en mode évaluation.
    
    Args:
        path (str): chemin du fichier modèle
    Returns:
        model (torch.nn.Module): modèle chargé et prêt à l'inférence
    """
    model = load_facenet_model()  # Charge l'architecture FaceNet
    model.load_state_dict(torch.load(path, map_location=DEVICE))  # Charge les poids
    model.to(DEVICE)  # Déplace le modèle sur CPU ou GPU
    model.eval()  # Mode évaluation (pas d'entraînement)
    return model


def euclidean_distance(t1, t2):
    """
    Calcule la distance euclidienne entre deux vecteurs d'embeddings.
    
    Args:
        t1, t2: vecteurs (torch.Tensor ou np.ndarray)
    Returns:
        float: distance entre t1 et t2
    """
    # Conversion en tenseurs PyTorch si ce sont des tableaux NumPy
    if isinstance(t1, np.ndarray):
        t1 = torch.tensor(t1)
    if isinstance(t2, np.ndarray):
        t2 = torch.tensor(t2)
    
    # Norme L2 (distance euclidienne)
    return torch.norm(t1 - t2).item()


def get_all_saved_embeddings():
    """
    Charge toutes les embeddings stockées dans le dossier EMBEDDINGS_DIR.
    Chaque fichier .pt correspond à une personne connue.
    
    Returns:
        dict: {nom_personne: embedding_tensor}
    """
    embeddings = {}
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".pt"):  # On ne prend que les fichiers de type PyTorch
            name = file[:-3]  # Enlève l'extension .pt pour obtenir le nom
            embedding = load_embedding(name)  # Charge l'embedding
            embeddings[name] = embedding
    return embeddings


def recognize_face(model, image_path):
    """
    Tente de reconnaître un visage dans une image.
    Compare l'embedding du visage trouvé avec ceux enregistrés.
    
    Args:
        model (torch.nn.Module): modèle FaceNet chargé
        image_path (str): chemin vers l'image à analyser
    """
    # 1️⃣ Lecture de l'image en bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # 2️⃣ Conversion des bytes en image exploitable
    image = read_image_from_bytes(image_bytes)

    # 3️⃣ Détection des visages dans l'image
    faces = extract_faces(image)
    if not faces:
        print("Aucun visage détecté dans l'image.")
        return

    # 4️⃣ Prétraitement du visage (redimensionnement, normalisation…)
    face_tensor = preprocess_face(faces[0])

    # 5️⃣ Extraction de l'embedding avec FaceNet
    embedding = get_embedding(model, face_tensor, DEVICE)

    # 6️⃣ Chargement des embeddings enregistrées
    saved_embeddings = get_all_saved_embeddings()
    if not saved_embeddings:
        print("Aucune embedding enregistrée dans le dossier.")
        return

    # 7️⃣ Comparaison avec toutes les embeddings enregistrées
    distances = {}
    for name, saved_emb in saved_embeddings.items():
        dist = euclidean_distance(embedding, saved_emb)
        distances[name] = dist

    # 8️⃣ Trouver le visage le plus proche
    best_match = min(distances, key=distances.get)  # nom avec la plus petite distance
    best_distance = distances[best_match]

    # 9️⃣ Vérification du seuil
    if best_distance < THRESHOLD:
        print(f"Visage reconnu : {best_match} (distance = {best_distance:.3f})")
    else:
        print(f"Visage inconnu (distance minimale = {best_distance:.3f})")


# ---------------------------
# EXÉCUTION DIRECTE DU SCRIPT
# ---------------------------
if __name__ == "__main__":
    # Chemin vers l'image à reconnaître (à remplacer par celle que tu veux tester)
    IMAGE_TO_RECOGNIZE = "rom.jpg"
    
    # Charger le modèle
    model = load_model(MODEL_PATH)
    
    # Lancer la reconnaissance
    recognize_face(model, IMAGE_TO_RECOGNIZE)

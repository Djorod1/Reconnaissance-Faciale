import os
import torch
from app.model import load_facenet_model
from app.face_utils import read_image_from_bytes, preprocess_face, get_embedding
from app.face_detect import extract_faces
from app.db import save_embedding

MODEL_PATH = "facenet_africain_djorod.pth"
EMBEDDINGS_DIR = "embeddings"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(path):
    model = load_facenet_model()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def register_face(name: str, image_path: str):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image = read_image_from_bytes(image_bytes)
    faces = extract_faces(image)

    if not faces:
        print("Aucun visage détecté.")
        return

    face_tensor = preprocess_face(faces[0])
    model = load_model(MODEL_PATH)
    embedding = get_embedding(model, face_tensor, DEVICE)

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    save_embedding(name, embedding)

    print(f"Embedding enregistré pour {name} dans '{EMBEDDINGS_DIR}/{name}.pt'.")

if __name__ == "__main__":
    nom_personne = "Roderic"
    chemin_image = "Roderic.jpg"
    register_face(nom_personne, chemin_image)

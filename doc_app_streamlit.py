"""
Script Streamlit : Système de Reconnaissance Faciale
Auteur : École d'été IA 2025
Description : 
    - Reconnaissance faciale sur images importées et flux webcam.
    - Affiche visage détecté côte à côte avec l'image originale.
    - Pour la webcam : rectangle autour des visages + nom et distance en temps réel.
"""

import os
import torch
import numpy as np
import streamlit as st
from app.model import load_facenet_model
from app.face_utils import read_image_from_bytes, preprocess_face, get_embedding
from app.face_detect import extract_faces
from app.db import load_embedding
from PIL import Image
import cv2

# --- CONFIGURATION ---
MODEL_PATH = "facenet_africain_djorod.pth"  # Modèle Facenet pré-entraîné
EMBEDDINGS_DIR = "embeddings"               # Dossier des embeddings des visages connus
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # GPU si disponible
THRESHOLD = 0.8  # Distance max pour considérer un visage reconnu

# --- PAGE STREAMLIT ---
st.set_page_config(page_title="Reconnaissance Faciale", page_icon="📸", layout="centered")

# --- STYLE CSS ---
st.markdown("""
    <style>
        body { background-color: #F7F9FB; }
        .title { text-align: center; font-size: 32px !important; color: #0E4D92; font-weight: bold; }
        .subtitle { text-align: center; color: #555; font-size: 18px !important; margin-bottom: 20px; }
        .result-card { padding: 15px; border-radius: 10px; background-color: #E8F0FE; border: 1px solid #B0BEC5; text-align: center; font-size: 18px; }
        .success-card { background-color: #E6F4EA; border: 1px solid #81C784; }
        .error-card { background-color: #FFEBEE; border: 1px solid #E57373; }
        .warning-card { background-color: #FFF3E0; border: 1px solid #FFB74D; }
    </style>
""", unsafe_allow_html=True)

# Titre et sous-titre
st.markdown('<div class="title">📸 Système de Reconnaissance Faciale</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Modèle : <b>facenet_africain_djorod.pth</b> — École d\'été IA 2025</div>', unsafe_allow_html=True)
st.write("---")

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model(path):
    """Charge le modèle Facenet et le met en mode évaluation."""
    model = load_facenet_model()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model(MODEL_PATH)

# --- FONCTIONS UTILES ---

def euclidean_distance(t1, t2):
    """Calcule la distance euclidienne entre deux embeddings."""
    if isinstance(t1, np.ndarray):
        t1 = torch.tensor(t1)
    if isinstance(t2, np.ndarray):
        t2 = torch.tensor(t2)
    return torch.norm(t1 - t2).item()

def get_all_saved_embeddings():
    """Charge tous les embeddings existants depuis le dossier EMBEDDINGS_DIR."""
    embeddings = {}
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".pt"):
            name = file[:-3]
            embedding = load_embedding(name)
            embeddings[name] = embedding
    return embeddings

def recognize_face(image):
    """
    Reconnaissance faciale pour une image (PIL ou ndarray).
    Retourne : (nom, distance, visage détecté)
    """
    faces = extract_faces(image)
    if not faces:
        return None, None, None  # Aucun visage détecté
    face_img = faces[0]  # On prend le premier visage détecté
    face_tensor = preprocess_face(face_img)
    embedding = get_embedding(model, face_tensor, DEVICE)
    saved_embeddings = get_all_saved_embeddings()
    if not saved_embeddings:
        return None, None, face_img
    # Calcul des distances avec les embeddings connus
    distances = {name: euclidean_distance(embedding, saved_emb)
                 for name, saved_emb in saved_embeddings.items()}
    best_match = min(distances, key=distances.get)
    best_distance = distances[best_match]
    if best_distance < THRESHOLD:
        return best_match, best_distance, face_img
    else:
        return "Inconnu", best_distance, face_img

# --- INTERFACE UTILISATEUR ---
option = st.radio("🔽 Choisissez une option :", ["📂 Importer une image", "🎥 Utiliser la webcam"])

# -------------------- IMPORT IMAGE --------------------
if option == "📂 Importer une image":
    uploaded_file = st.file_uploader("Importer une image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        original_img = Image.open(uploaded_file)
        st.image(original_img, caption="Image importée", width=300)

        if st.button("🔍 Analyser l'image"):
            name, distance, face_img = recognize_face(np.array(original_img))

            if name is None:
                st.markdown('<div class="result-card error-card">❌ Aucun visage détecté</div>', unsafe_allow_html=True)
            else:
                # Affichage côte à côte : original / visage détecté
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_img, caption="Image originale", use_column_width=True)
                with col2:
                    if face_img is not None:
                        st.image(face_img, caption="Visage détecté", use_column_width=True)

                # Affichage du résultat
                if name == "Inconnu":
                    st.markdown(f'<div class="result-card warning-card">⚠️ Visage inconnu (distance = {distance:.3f})</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-card success-card">✅ Visage reconnu : <b>{name}</b> (distance = {distance:.3f})</div>', unsafe_allow_html=True)

# -------------------- WEBCAM EN TEMPS RÉEL --------------------
elif option == "🎥 Utiliser la webcam":
    run = st.checkbox("📷 Activer la webcam")
    FRAME_WINDOW = st.image([])  # Placeholder pour le flux webcam
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Impossible de lire la webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = extract_faces(rgb_frame)
        display_frame = rgb_frame.cop

        # Parcours de tous les visages détectés
        for face_img in faces:
            name, distance, _ = recognize_face(face_img)
            h, w, _ = face_img.shape
            # Dessin du rectangle autour du visage
            y1, x1 = 0, 0
            y2, x2 = h, w
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Affichage du label
            if name is None:
                label = "Aucun visage"
            elif name == "Inconnu":
                label = f"Inconnu ({distance:.2f})"
            else:
                label = f"{name} ({distance:.2f})"
            cv2.putText(display_frame, label, (x1, max(y1-10,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Affichage du flux dans Streamlit
        FRAME_WINDOW.image(display_frame)

    cap.release()
    st.success("Webcam désactivée.")
else:
    st.error("Veuillez sélectionner une option pour continuer.")4
# --- INSTALLATION DES DÉPENDANCES ---
# Pour installer les dépendances nécessaires, exécutez :
# pip install -r requirements.txt
# Pour lancer l'application,                                                                                                              utilisez :                                                                                                                                                                                                    
# pip install face_recognition
# pip install streamlit

# streamlit run app_streamlit.py
# --- FIN DU SCRIPT ---

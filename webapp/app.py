import streamlit as st
import requests
from PIL import Image
import os
import io
import numpy as np

API_URL = "https://project8api-h8bgdghmaeh7cnch.francecentral-01.azurewebsites.net/predict"
# API_URL = "http://127.0.0.1:8000/predict"
SAMPLE_DIR = "webapp/test_images"

CITYSCAPES_PALETTE = np.array([
	[0, 0, 0], [128, 64, 128], [70, 70, 70], [220, 220, 0], [107, 142, 35], [70, 130, 180], [220, 20, 60], [0, 0, 142]
], dtype=np.uint8)

MAPPING_ARRAY = np.array([
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7
])

st.set_page_config(page_title="Future Vision Transport", layout="wide")
st.title("Démo : Segmentation Cityscapes")

available_images = [f for f in os.listdir(SAMPLE_DIR) if f.endswith("leftImg8bit.png")]

if not available_images:
	st.error(f"Aucune image trouvée dans le dossier '{SAMPLE_DIR}'.")
else:
	selected_img_name = st.selectbox("Choisissez une image de test :", available_images)

	img_path = os.path.join(SAMPLE_DIR, selected_img_name)
	mask_name = selected_img_name.replace("leftImg8bit.png", "gtFine_labelIds.png")
	mask_path = os.path.join(SAMPLE_DIR, mask_name)

	col1, col2, col3 = st.columns(3)
	
	original_image = Image.open(img_path)
	with col1:
		st.subheader("Image Réelle")
		st.image(original_image, width="content")

	with col2:
		st.subheader("Vrai Masque")
		if os.path.exists(mask_path):

			true_mask = Image.open(mask_path)

			try:
				cat_mask = MAPPING_ARRAY[np.array(true_mask)]
				true_mask_rgb = CITYSCAPES_PALETTE[cat_mask]
				st.image(true_mask_rgb, width="content")
			except IndexError:
				st.warning("Mapping des IDs requis pour coloriser ce masque.")
		else:
			st.warning("Masque réel introuvable.")

	with col3:
		st.subheader("Prédiction de l'IA")
		with st.spinner("L'API charge..."):
			try:
				with open(img_path, "rb") as f:
					files = {"file": (selected_img_name, f, "image/png")}
					response = requests.post(API_URL, files=files)
				
				if response.status_code == 200:

					pred_image = Image.open(io.BytesIO(response.content))
					st.image(pred_image, width="content")
				else:
					st.error(f"Erreur de l'API : {response.status_code}")
			except requests.exceptions.ConnectionError:
				st.error("Impossible de contacter l'API. Est-elle bien lancée sur le port 8000 ?")
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, RedirectResponse
import torch
import segmentation_models_pytorch as smp
from torchvision.transforms import v2
from PIL import Image
import numpy as np
import io
from contextlib import asynccontextmanager


device = torch.device("cpu")
model = None
transform_img = None

CITYSCAPES_PALETTE = np.array([
	[0, 0, 0], [128, 64, 128], [70, 70, 70], [220, 220, 0], [107, 142, 35], [70, 130, 180], [220, 20, 60], [0, 0, 142]
], dtype=np.uint8)

@asynccontextmanager
async def lifespan(app: FastAPI):

	global model, transform_img
	print("⏳ Chargement du modèle...")
	
	model = smp.Unet("mobilenet_v2", encoder_weights=None, classes=8)

	model.load_state_dict(torch.load("../outputs/models/cityscapes_mobilenet_v2_data_augmentation.pth", map_location=device))
	model.to(device)
	model.eval()
	
	transform_img = v2.Compose([
		v2.ToImage(),
		v2.Resize((512, 1024), interpolation=v2.InterpolationMode.BILINEAR),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	print("✅ Modèle chargé et prêt !")
	yield

app = FastAPI(lifespan=lifespan, title="Cityscapes Segmentation API")

@app.get("/")
def home():
	response = RedirectResponse(url="/docs")
	return response

@app.post("/predict")
async def predict_mask(file: UploadFile = File(...)):
	
	image_bytes = await file.read()
	image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
	
	input_tensor = transform_img(image).unsqueeze(0).to(device) #type: ignore
	
	with torch.no_grad():
		output = model(input_tensor) #type: ignore
		pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
		
	pred_mask_rgb = CITYSCAPES_PALETTE[pred_mask]
	
	result_image = Image.fromarray(pred_mask_rgb)
	img_byte_arr = io.BytesIO()
	result_image.save(img_byte_arr, format='PNG')
	
	return Response(content=img_byte_arr.getvalue(), media_type="image/png")
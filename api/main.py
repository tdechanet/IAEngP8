from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, RedirectResponse
import onnxruntime as ort
from PIL import Image
import numpy as np
import io
from contextlib import asynccontextmanager
from pathlib import Path

CITYSCAPES_PALETTE = np.array([
    [0, 0, 0], [128, 64, 128], [70, 70, 70], [220, 220, 0], 
    [107, 142, 35], [70, 130, 180], [220, 20, 60], [0, 0, 142]
], dtype=np.uint8)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "mobilenet_v2.onnx"

ort_session = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ort_session
    ort_session = ort.InferenceSession(MODEL_PATH)
    yield

app = FastAPI(lifespan=lifespan, title="Cityscapes Segmentation API")

@app.get("/")
def home():
    return RedirectResponse(url="/docs")

@app.post("/predict")
async def predict_mask(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    image_resized = image.resize((1024, 512), Image.Resampling.BILINEAR)
    img_ndarray = np.array(image_resized, dtype=np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_ndarray = (img_ndarray - mean) / std
    
    img_ndarray = np.transpose(img_ndarray, (2, 0, 1))
    img_ndarray = np.expand_dims(img_ndarray, axis=0)

    input_name = ort_session.get_inputs()[0].name #type: ignore
    ort_inputs = {input_name: img_ndarray}
    ort_outs = ort_session.run(None, ort_inputs) #type: ignore

    pred_mask = np.argmax(ort_outs[0], axis=1)[0] #type: ignore
    
    pred_mask_rgb = CITYSCAPES_PALETTE[pred_mask]
    result_image = Image.fromarray(pred_mask_rgb)
    
    result_image = result_image.resize(image.size, Image.Resampling.NEAREST)
    
    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='PNG')
    
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")
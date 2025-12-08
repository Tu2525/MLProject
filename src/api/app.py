import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
from src.models.model import CNNtoRNN
from src.preprocessing.transforms import get_transforms, Vocabulary
from config.config import config
import pandas as pd
import time
import logging

# configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = FastAPI(title="Image Captioning API")

# Load model and vocabulary
# Note: In a real production app, you'd load the vocab from a saved file (pickle/json)
# Here we rebuild it from the captions file for simplicity, which is slow but works.
print("Loading Vocabulary...")
df = pd.read_csv(config.CAPTIONS_FILE)
vocab = Vocabulary(config.FREQ_THRESHOLD)
vocab.build_vocabulary(df["caption"].tolist())

print("Loading Model...")
model = CNNtoRNN(
    embed_size=config.EMBED_SIZE,
    hidden_size=config.HIDDEN_SIZE,
    vocab_size=len(vocab),
    num_layers=config.NUM_LAYERS
).to(config.DEVICE)

# Load weights if they exist
try:
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    print("Model loaded successfully.")
except FileNotFoundError:
    print("No model checkpoint found. Please train the model first.")

model.eval()
transform = get_transforms(config.IMAGE_SIZE)

@app.post("/predict")
async def predict_caption(file: UploadFile = File(...)):
    start_ts = time.time()
    image_data = await file.read()
    file_size = len(image_data)
    logging.info(f"Received file: filename={file.filename} size={file_size} bytes")

    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        logging.exception("Failed to open image")
        return {"error": "Failed to open image. Make sure a valid image file is uploaded."}

    image_tensor = transform(image).unsqueeze(0).to(config.DEVICE)

    # Do not squeeze the batch dimension — caption_image expects a batched tensor
    caption_tokens = model.caption_image(image_tensor, vocab)
    caption_text = " ".join(caption_tokens)

    elapsed = time.time() - start_ts
    logging.info(f"Processed file in {elapsed:.3f}s, caption_length={len(caption_tokens)}")

    return {"caption": caption_text, "processing_time_s": round(elapsed, 3)}

@app.get("/")
def root():
    return {"message": "Welcome to the Image Captioning API"}

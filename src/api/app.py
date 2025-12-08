from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
from src.models.model import CNNtoRNN
from src.preprocessing.transforms import get_transforms, Vocabulary
from config.config import config
import pandas as pd

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
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    image_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
    
    caption = model.caption_image(image_tensor.squeeze(0), vocab)
    
    return {"caption": " ".join(caption)}

@app.get("/")
def root():
    return {"message": "Welcome to the Image Captioning API"}

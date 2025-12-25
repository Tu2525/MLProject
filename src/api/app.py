import io
import os
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
from transformers import GPT2Tokenizer

from config.config import config
from src.models.model import get_model
from src.preprocessing.transforms import get_transforms

app = FastAPI(title="Object Captioning LLM")

# Load Model
device = config.DEVICE
model = get_model(config).to(device)

# For ResNetGPT2 legacy support
if config.MODEL_TYPE == "resnet_gpt2":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model_llm.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained model.")
    else:
        print("Warning: No trained model found for ResNetGPT2.")
else:
    tokenizer = None # Handled internally by SOTA models

model.eval()
transform = get_transforms(train=False)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read Image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Generate Caption
        if config.MODEL_TYPE == "resnet_gpt2":
            img_tensor = transform(image).to(device)
            caption = model.generate_caption(img_tensor, tokenizer)
        else:
            # SOTA models take PIL image directly
            caption = model.generate_caption(image)
        
        return {
            "caption": caption
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Speaking LLM Captioner</title>
        <style>
            body { font-family: sans-serif; display: flex; flex-direction: column; align-items: center; padding: 50px; background: #222; color: #fff; }
            .box { border: 2px dashed #00ffcc; padding: 40px; border-radius: 10px; text-align: center; cursor: pointer; }
            img { max-width: 400px; margin: 20px; border-radius: 10px; }
            #result { font-size: 24px; margin: 20px; color: #00ffcc; }
            button { padding: 10px 20px; background: #00ffcc; border: none; border-radius: 5px; font-weight: bold; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>image captioner gamed</h1>
        <div class="box" onclick="document.getElementById('file').click()">
            Click to Upload Image
            <input type="file" id="file" style="display:none" onchange="upload(this.files[0])">
        </div>
        <img id="preview">
        <div id="result"></div>

        <script>
            function upload(file) {
                let preview = document.getElementById('preview');
                preview.src = URL.createObjectURL(file);
                
                let formData = new FormData();
                formData.append('file', file);
                
                document.getElementById('result').innerText = "Thinking...";
                
                fetch('/predict', { method: 'POST', body: formData })
                .then(r => {
                    if (!r.ok) throw new Error("Server Error");
                    return r.json();
                })
                .then(data => {
                    document.getElementById('result').innerText = data.caption;
                })
                .catch(e => {
                    document.getElementById('result').innerText = "Error: " + e;
                });
            }
        </script>
    </body>
    </html>
    """

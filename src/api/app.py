import os
import sys
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
from src.models.model import CNNtoRNN
from src.data.dataset import Vocabulary
from config.config import config

app = FastAPI(title="Image Captioning API")

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
# ...existing code...
# Load model and vocab
device = config.DEVICE
checkpoint_path = config.MODEL_SAVE_PATH if os.path.exists(config.MODEL_SAVE_PATH) else config.CHECKPOINT_PATH

if not os.path.exists(checkpoint_path):
    # Fallback to root if not moved yet
    root_best = os.path.join(config.ROOT_DIR, "best_model.pth")
    root_checkpoint = os.path.join(config.ROOT_DIR, "checkpoint.pth")
    checkpoint_path = root_best if os.path.exists(root_best) else root_checkpoint

if not os.path.exists(checkpoint_path):
    print(f"Warning: Model checkpoint not found at {checkpoint_path}. API will start but /predict will fail.")
    model = None
    vocab = None
else:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    itos = checkpoint["vocab_itos"]
    vocab = Vocabulary(config.FREQ_THRESHOLD)
    vocab.itos = {int(k): v for k, v in itos.items()}
    vocab.stoi = {v: int(k) for k, v in vocab.itos.items()}

    model = CNNtoRNN(
        embed_size=config.EMBED_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        vocab_size=len(vocab),
        encoder_dim=512,
        attention_dim=config.EMBED_SIZE,
        device=device
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = transform(image).to(device)
        
        caption = model.caption_image(image, vocab)
        return {"caption": caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Captioning</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; background-color: #f0f2f5; }
            .container { background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); width: 400px; text-align: center; }
            #drop-area { border: 2px dashed #007bff; border-radius: 8px; padding: 2rem; cursor: pointer; transition: background 0.3s; margin-bottom: 1rem; }
            #drop-area.highlight { background-color: #e7f3ff; }
            #preview { max-width: 100%; max-height: 300px; margin-top: 1rem; display: none; border-radius: 4px; }
            #result { margin-top: 1.5rem; font-weight: bold; color: #333; min-height: 1.5em; }
            .loading { color: #666; font-style: italic; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Image Captioning</h1>
            <div id="drop-area">
                <p>Drag and drop an image here or click to select</p>
                <input type="file" id="fileElem" accept="image/*" style="display:none" onchange="handleFiles(this.files)">
            </div>
            <img id="preview">
            <div id="result"></div>
        </div>

        <script>
            let dropArea = document.getElementById('drop-area');
            let resultDiv = document.getElementById('result');
            let preview = document.getElementById('preview');

            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
            });

            function preventDefaults (e) {
                e.preventDefault();
                e.stopPropagation();
            }

            ['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'), false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'), false);
            });

            dropArea.addEventListener('drop', handleDrop, false);
            dropArea.onclick = () => document.getElementById('fileElem').click();

            function handleDrop(e) {
                let dt = e.dataTransfer;
                let files = dt.files;
                handleFiles(files);
            }

            function handleFiles(files) {
                if (files.length > 0) {
                    uploadFile(files[0]);
                    displayPreview(files[0]);
                }
            }

            function displayPreview(file) {
                let reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onloadend = function() {
                    preview.src = reader.result;
                    preview.style.display = 'block';
                }
            }

            function uploadFile(file) {
                resultDiv.innerHTML = '<span class="loading">Generating caption...</span>';
                let formData = new FormData();
                formData.append('file', file);

                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.caption) {
                        resultDiv.innerText = data.caption;
                    } else {
                        resultDiv.innerText = 'Error: ' + (data.detail || 'Unknown error');
                    }
                })
                .catch(error => {
                    console.error(error);
                    resultDiv.innerText = 'Error uploading file';
                });
            }
        </script>
    </body>
    </html>
    """


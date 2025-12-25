# Image Captioning with SOTA Models

This project provides a unified API for Image Captioning using various State-of-the-Art (SOTA) models as well as a custom ResNet+GPT2 implementation.

## Supported Models

1.  **BLIP (Bootstrapping Language-Image Pre-training)**
    *   Model: `Salesforce/blip-image-captioning-large`
    *   Status: **Default** (Best Performance)
    *   Description: Produces highly accurate and detailed captions.

2.  **ViT-GPT2**
    *   Model: `nlpconnect/vit-gpt2-image-captioning`
    *   Status: Available
    *   Description: Uses Vision Transformer (ViT) encoder and GPT-2 decoder.

3.  **ResNet50 + GPT-2 (Custom)**
    *   Model: Custom implementation trained from scratch.
    *   Status: Legacy / Experimental
    *   Description: Good for learning purposes or custom datasets.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Edit `config/config.py` to select the model:

```python
class Config:
    # ...
    MODEL_TYPE = "blip" # Options: "blip", "vit_gpt2", "resnet_gpt2"
```

## Running the API

Start the FastAPI server:

```bash
python main.py --mode api
```

Open your browser at `http://localhost:8001` to use the drag-and-drop interface.

## Training (ResNet+GPT2 only)

To train the custom model:

1.  Set `MODEL_TYPE = "resnet_gpt2"` in config.
2.  Run:
    ```bash
    python main.py --mode train
    ```

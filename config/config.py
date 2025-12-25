import os
import torch

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Assuming dataset is in the parent directory as per previous context
    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'flickr30k_resized', 'Images')
    CAPTIONS_FILE = os.path.join(os.path.dirname(BASE_DIR), 'captions30k.txt')
    
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    BATCH_SIZE = 128 # Increased for 16GB VRAM
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10
    NUM_WORKERS = 4 # Enable multiprocessing for data loading
    
    # Model Config
    MODEL_TYPE = "blip"        # Options: "resnet_gpt2", "vit_gpt2", "blip"
    ENCODER_MODEL = "resnet50" # Using ResNet50 (only for resnet_gpt2)
    DECODER_MODEL = "gpt2"     # Using GPT-2 Base (only for resnet_gpt2)
    EMBED_DIM = 768            # GPT-2 hidden size
    MAX_SEQ_LEN = 40
    
    # Image Config
    IMAGE_SIZE = (224, 224)

config = Config()

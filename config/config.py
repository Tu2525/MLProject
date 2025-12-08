import torch
import os

class Config:
    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, 'Images')
    CAPTIONS_FILE = os.path.join(ROOT_DIR, 'captions.txt')
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'models', 'image_captioning_model.pth')
    LOG_DIR = os.path.join(ROOT_DIR, 'logs')

    # Hyperparameters
    BATCH_SIZE = 128  # Increased from 32 to utilize 16GB VRAM
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 30
    EMBED_SIZE = 256
    HIDDEN_SIZE = 256
    NUM_LAYERS = 1
    NUM_WORKERS = 8  # Increased to 8 for i7 10850H
    
    # Image Preprocessing
    IMAGE_SIZE = (299, 299)
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Vocabulary
    FREQ_THRESHOLD = 5
    
    # Validation
    VAL_SPLIT = 0.2
    SHUFFLE_DATASET = True

config = Config()

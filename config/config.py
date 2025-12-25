import os
# Set allocator config immediately to ensure it applies before CUDA init
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

class Config:
    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Auto-detect resized images for performance
    DATA_DIR = os.path.join(ROOT_DIR, 'flickr30k_resized\Images')
    if not os.path.exists(DATA_DIR):
        print(f"Resized images not found at {DATA_DIR}. Using original Images folder. Run src/preprocessing/resize_images.py for faster training.")
        DATA_DIR = os.path.join(ROOT_DIR, 'Images')
        
    CAPTIONS_FILE = os.path.join(ROOT_DIR, 'captions30k.txt')
    MODEL_DIR = os.path.join(ROOT_DIR, 'models')
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'best_model.pth')
    CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'checkpoint.pth')
    LOG_DIR = os.path.join(ROOT_DIR, 'logs')

    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  
    
    # Hyperparameters
    BATCH_SIZE = 512  # Increased for RTX 5000 (16GB VRAM)
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 100  # Increased from 50 to 100
    EMBED_SIZE = 512  # Increased from 256 for better capacity
    HIDDEN_SIZE = 512  # Increased from 256 for better capacity
    NUM_LAYERS = 1 
    NUM_WORKERS = 0  # Set to 0 for Windows
    PIN_MEMORY = True
    CACHE_IMAGES = False # Disabled to match cnntraining.py behavior (or keep if user wants optimization)
    
    # Image Preprocessing
    IMAGE_SIZE = (224, 224)
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Vocabulary
    FREQ_THRESHOLD = 5
    
    # Validation
    VAL_SPLIT = 0.1 # 90/10 split
    SHUFFLE_DATASET = True

config = Config()

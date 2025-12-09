import os
# Set allocator config immediately to ensure it applies before CUDA init
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

class Config:
    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Auto-detect resized images for performance
    DATA_DIR = os.path.join(ROOT_DIR, 'Images_Resized')
    if not os.path.exists(DATA_DIR):
        print(f"Resized images not found at {DATA_DIR}. Using original Images folder. Run src/preprocessing/resize_images.py for faster training.")
        DATA_DIR = os.path.join(ROOT_DIR, 'Images')
        
    CAPTIONS_FILE = os.path.join(ROOT_DIR, 'captions.txt')
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'models', 'image_captioning_model.pth')
    LOG_DIR = os.path.join(ROOT_DIR, 'logs')

    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  
    
    # Hyperparameters
    BATCH_SIZE = 512  
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 100
    EMBED_SIZE = 256 
    HIDDEN_SIZE = 256
    NUM_LAYERS = 1 
    NUM_WORKERS = 2  
    PIN_MEMORY = True
    CACHE_IMAGES = True # Enable in-memory caching to bypass slow NTFS reads
    
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

import argparse
import uvicorn
import torch  # Added import
from src.data.dataset import get_loaders
from src.models.model import CNNtoRNN
from src.training.trainer import train
from src.preprocessing.transforms import get_transforms
from config.config import config

def run_training():
    print("Starting Training...")
    transform = get_transforms(config.IMAGE_SIZE)
    train_loader, val_loader, dataset = get_loaders(
        config.DATA_DIR,
        config.CAPTIONS_FILE,
        transform=transform,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        val_split=config.VAL_SPLIT
    )

    model = CNNtoRNN(
        embed_size=config.EMBED_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        vocab_size=len(dataset.vocab),
        num_layers=config.NUM_LAYERS
    ).to(config.DEVICE)

    train(config, train_loader, val_loader, dataset, model)

def run_api():
    print("Starting API...")
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    # Print device info only once in the main process
    print(f"Using device: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    parser = argparse.ArgumentParser(description="Image Captioning Project")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "api"], help="Mode: train or api")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        run_training()
    elif args.mode == "api":
        run_api()

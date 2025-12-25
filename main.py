import argparse
import uvicorn
import torch
from torch.utils.data import DataLoader, random_split

from config.config import config
from src.data.dataset import CaptionDataset
from src.models.model import ResNetGPT2
from src.training.trainer import train_model
from src.preprocessing.transforms import get_transforms

def run_training():
    print("Initializing Dataset...")
    transform = get_transforms(train=True)
    dataset = CaptionDataset(config.DATA_DIR, config.CAPTIONS_FILE, transform=transform)
    
    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    
    print("Initializing Model (ResNet50 + GPT-2)...")
    model = ResNetGPT2()
    
    print("Starting Training...")
    train_model(model, train_loader, val_loader, config, dataset.tokenizer)

def run_api():
    print("Starting Speaking LLM API...")
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8001, reload=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "api"])
    args = parser.parse_args()
    
    if args.mode == "train":
        run_training()
    elif args.mode == "api":
        run_api()

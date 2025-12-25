import argparse
import uvicorn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler
import os

from src.data.dataset import FlickrDataset, MyCollate
from src.models.model import CNNtoRNN
from src.training.trainer import train_model, load_checkpoint
from config.config import config

def run_training():
    print("Starting Training...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_dataset = FlickrDataset(
        config.DATA_DIR, 
        config.CAPTIONS_FILE, 
        transform=transform, 
        cache_images=config.CACHE_IMAGES
    )
    pad_idx = full_dataset.vocab.stoi["<PAD>"]

    train_size = int((1 - config.VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, 
        collate_fn=MyCollate(pad_idx)
    )
    
    val_loader = DataLoader(
        val_set, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, 
        collate_fn=MyCollate(pad_idx)
    )

    model = CNNtoRNN(
        embed_size=config.EMBED_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        vocab_size=len(full_dataset.vocab),
        encoder_dim=512,
        attention_dim=config.EMBED_SIZE,
        device=config.DEVICE
    ).to(config.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    scaler = GradScaler('cuda')

    start_epoch, best_val_loss = load_checkpoint(config.CHECKPOINT_PATH, model, optimizer, scheduler, config.DEVICE)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        scaler=scaler,
        num_epochs=config.NUM_EPOCHS,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        device=config.DEVICE,
        accumulation_steps=2,
        vocab=full_dataset.vocab,
        checkpoint_path=config.CHECKPOINT_PATH,
        best_model_path=config.MODEL_SAVE_PATH
    )

def run_api():
    print("Starting API...")
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    print(f"Using device: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="Image Captioning Project")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "api"], help="Mode: train or api")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        torch.cuda.empty_cache()
        run_training()
    elif args.mode == "api":
        run_api()
    elif args.mode == "eval":
        from src.preprocessing.transforms import get_transforms
        from src.data.dataset import FlickrDataset, MyCollate
        from src.models.model import CNNtoRNN
        from src.training.trainer import print_bleu_score
        import random

        # Load dataset and transforms
        print("Loading dataset...")
        transform = get_transforms(config.IMAGE_SIZE, train=False)

        dataset = FlickrDataset(
            config.DATA_DIR, 
            config.CAPTIONS_FILE, 
            transform=transform, 
            cache_images=config.CACHE_IMAGES
        )
        
        val_loader = DataLoader(
            dataset, batch_size=1, shuffle=True, 
            num_workers=0, collate_fn=MyCollate(dataset.vocab.stoi["<PAD>"])
        )

        model_path = config.MODEL_SAVE_PATH
        print(f"Loading model from {model_path}...")
        model = CNNtoRNN(
            embed_size=config.EMBED_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            vocab_size=len(dataset.vocab),
            encoder_dim=512,
            attention_dim=config.EMBED_SIZE,
            device=config.DEVICE
        ).to(config.DEVICE)

        try:
            checkpoint = torch.load(model_path, map_location=config.DEVICE)
            model.load_state_dict(checkpoint["state_dict"])
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            exit(1)

        model.eval()

        # Calculate BLEU score first
        print("\nCalculating BLEU score...")
        print_bleu_score(val_loader, model, dataset, config.DEVICE)

        num_samples = 5
        print(f"\nEvaluating on {num_samples} random examples from validation set...")

        indices = list(range(len(dataset)))
        random.shuffle(indices)
        sample_indices = indices[:num_samples]

        for i, idx in enumerate(sample_indices):
            img, caption_tensor = dataset[idx]

            # Convert target caption to text
            target_caption = []
            for token_idx in caption_tensor:
                word = dataset.vocab.itos[token_idx.item()]
                if word == "<EOS>":
                    break
                if word != "<SOS>" and word != "<PAD>":
                    target_caption.append(word)

            # Generate prediction
            img_tensor = img.unsqueeze(0).to(config.DEVICE)
            with torch.no_grad():
                prediction = model.caption_image(img_tensor, dataset.vocab)

            print(f"\nExample {i+1}/{num_samples}")
            print(f"Target:     {' '.join(target_caption)}")
            print(f"Prediction: {' '.join(prediction)}")

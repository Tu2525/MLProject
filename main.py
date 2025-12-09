import argparse
import uvicorn
import torch  # Added import
import os
from src.data.dataset import get_loaders
from src.models.model import CNNtoRNN
from src.training.trainer import train, print_bleu_score
from src.preprocessing.transforms import get_transforms
from config.config import config

def run_training():
    print("Starting Training...")
    train_transform = get_transforms(config.IMAGE_SIZE, train=True)
    val_transform = get_transforms(config.IMAGE_SIZE, train=False)
    
    
    train_loader, val_loader, dataset = get_loaders(
        config.DATA_DIR,
        config.CAPTIONS_FILE,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        val_split=config.VAL_SPLIT,
        pin_memory=config.PIN_MEMORY,
        cache_images=config.CACHE_IMAGES,
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
        torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark for speed

    parser = argparse.ArgumentParser(description="Image Captioning Project")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "api", "eval"], help="Mode: train, api, or eval")
    parser.add_argument("--image", type=str, help="Path to image for evaluation")
    parser.add_argument("--model_path", type=str, default=config.MODEL_SAVE_PATH, help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of random samples to evaluate")
    
    # Add arguments to override model architecture for evaluation
    parser.add_argument("--embed_size", type=int, default=config.EMBED_SIZE, help="Embed size for model")
    parser.add_argument("--hidden_size", type=int, default=config.HIDDEN_SIZE, help="Hidden size for model")
    parser.add_argument("--num_layers", type=int, default=config.NUM_LAYERS, help="Number of layers for model")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        torch.cuda.empty_cache() # Clear VRAM before training
        run_training()
    elif args.mode == "api":
        run_api()
    elif args.mode == "eval":
        # Load dataset and transforms
        print("Loading dataset...")
        transform = get_transforms(config.IMAGE_SIZE, train=False)
        
        # We need the validation loader to get validation images
        train_loader, val_loader, dataset = get_loaders(
            config.DATA_DIR,
            config.CAPTIONS_FILE,
            train_transform=transform,
            val_transform=transform,
            batch_size=1,
            num_workers=0
        )
            
        print(f"Loading model from {args.model_path}...")
        # Use args to initialize model, allowing override of config defaults
        model = CNNtoRNN(
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            vocab_size=len(dataset.vocab),
            num_layers=args.num_layers
        ).to(config.DEVICE)
        
        try:
            checkpoint = torch.load(args.model_path, map_location=config.DEVICE)
            model.load_state_dict(checkpoint["state_dict"])
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            exit(1)
            
        model.eval()

        if args.image:
            from src.utils.utils import evaluate_model
            caption = evaluate_model(model, dataset, args.image, config.DEVICE)
            print(f"\nGenerated Caption: {caption}")
        else:
            # Calculate BLEU score first
            print("\nCalculating BLEU score...")
            print_bleu_score(val_loader, model, dataset, config.DEVICE)
            
            import random
            print(f"\nEvaluating on {args.num_samples} random examples from validation set...")
            
            val_subset = val_loader.dataset
            indices = list(range(len(val_subset)))
            random.shuffle(indices)
            sample_indices = indices[:args.num_samples]
            
            for i, idx in enumerate(sample_indices):
                img, caption_tensor = val_subset[idx]
                
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
                
                print(f"\nExample {i+1}/{args.num_samples}")
                print(f"Target:     {' '.join(target_caption)}")
                print(f"Prediction: {' '.join(prediction)}")

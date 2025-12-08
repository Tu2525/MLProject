import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.utils.utils import save_checkpoint, load_checkpoint, print_examples, plot_loss_curves
import os

def train_one_epoch(loader, model, optimizer, criterion, device, vocab):
    model.train()
    loop = tqdm(loader, total=len(loader), leave=True)
    total_loss = 0
    
    scaler = torch.amp.GradScaler('cuda')
    
    for idx, (imgs, captions) in enumerate(loop):
        imgs = imgs.to(device)
        captions = captions.to(device)

        with torch.amp.autocast('cuda'):
            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()

        loop.set_description(f"Loss: {loss.item():.4f}")
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

def evaluate(loader, model, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for idx, (imgs, captions) in enumerate(loader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            total_loss += loss.item()
            
    model.train()
    return total_loss / len(loader)

def train(config, train_loader, val_loader, dataset, model):
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    model.to(config.DEVICE)
    
    # Load existing checkpoint if available to continue training
    start_epoch = 0
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Loading existing checkpoint from {config.MODEL_SAVE_PATH}")
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Note: We don't have epoch saved in checkpoint currently, so we start from 0 or assume continuation.
        # Ideally, save epoch in checkpoint. For now, let's just load weights and train for NUM_EPOCHS more.
        print("Resuming training with loaded weights...")

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Early stopping parameters
    patience = 5
    epochs_no_improve = 0

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
        train_loss = train_one_epoch(train_loader, model, optimizer, criterion, config.DEVICE, dataset.vocab)
        
        val_loss = evaluate(val_loader, model, criterion, config.DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Validation loss improved. Saving model to {config.MODEL_SAVE_PATH}")
            save_checkpoint(model, optimizer, filename=config.MODEL_SAVE_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Epochs without improvement: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
        
        # print_examples(model, config.DEVICE, dataset)
        
    # Plot and save graphs
    plot_loss_curves(train_losses, val_losses, config.LOG_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import os
import json

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['epoch'] + 1, checkpoint.get('best_val_loss', float('inf'))
    return 0, float('inf')

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, scaler, num_epochs, start_epoch, best_val_loss, device, accumulation_steps, vocab):
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}")

        optimizer.zero_grad()
        for batch_idx, (imgs, captions) in enumerate(loop):
            imgs = imgs.to(device)
            captions = captions.to(device)

            with autocast('cuda'):
                outputs = model(imgs, captions)
                targets = captions[:, 1:]
                outputs = outputs.reshape(-1, outputs.shape[2])
                targets = targets.reshape(-1)
                loss = criterion(outputs, targets)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps
            loop.set_postfix(loss=loss.item() * accumulation_steps)

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, captions in val_loader:
                imgs = imgs.to(device)
                captions = captions.to(device)
                with autocast('cuda'):
                    outputs = model(imgs, captions)
                    targets = captions[:, 1:]
                    outputs = outputs.reshape(-1, outputs.shape[2])
                    targets = targets.reshape(-1)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Best: {best_val_loss:.4f}")
        
        # Sanity check
        test_img, _ = next(iter(val_loader))
        print(f"Sanity Check: {model.caption_image(test_img[0].to(device), vocab)}")

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "vocab_itos": vocab.itos
        }
        torch.save(checkpoint, "checkpoint.pth")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, "best_model.pth")
            print("--> BEST MODEL SAVED!")

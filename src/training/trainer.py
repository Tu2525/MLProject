import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.utils.utils import save_checkpoint, load_checkpoint, print_examples

def train_one_epoch(loader, model, optimizer, criterion, device, vocab):
    model.train()
    loop = tqdm(loader, total=len(loader), leave=True)
    
    for idx, (imgs, captions) in enumerate(loop):
        imgs = imgs.to(device)
        captions = captions.to(device)

        outputs = model(imgs, captions[:-1])
        loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Loss: {loss.item():.4f}")
        loop.set_postfix(loss=loss.item())

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
    
    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
        train_one_epoch(train_loader, model, optimizer, criterion, config.DEVICE, dataset.vocab)
        
        val_loss = evaluate(val_loader, model, criterion, config.DEVICE)
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Validation loss improved. Saving model to {config.MODEL_SAVE_PATH}")
            save_checkpoint(model, optimizer, filename=config.MODEL_SAVE_PATH)
        
        # print_examples(model, config.DEVICE, dataset)

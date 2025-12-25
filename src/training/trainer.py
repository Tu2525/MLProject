import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

def train_model(model, train_loader, val_loader, config, tokenizer):
    model = model.to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    best_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        
        for images, input_ids, attention_mask in loop:
            images = images.to(config.DEVICE)
            input_ids = input_ids.to(config.DEVICE)
            attention_mask = attention_mask.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            # Logits: [batch, seq_len+1, vocab_size]
            logits = model(images, input_ids, attention_mask)
            
            # Shift logits and labels for next-token prediction
            # We want to predict input_ids based on previous context
            # The model output at index i corresponds to prediction for token at i+1
            # Input sequence to model: [Image, T1, T2, T3, ...]
            # Output logits:           [P1,    P2, P3, P4, ...]
            # Targets:                 [T1,    T2, T3, T4, ...]
            
            # We discard the last logit because we don't have a target for it
            shift_logits = logits[:, :-1, :].contiguous()
            # We use input_ids as targets
            shift_labels = input_ids.contiguous()
            
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_train_loss:.4f}")
        
        # Save checkpoint
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, "best_model_llm.pth"))
            print("Saved Best Model!")
            
        torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, "last_checkpoint_llm.pth"))

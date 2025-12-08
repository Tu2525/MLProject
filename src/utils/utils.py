import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def plot_loss_curves(train_losses, val_losses, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(log_dir, 'loss_curve.png')
    plt.savefig(plot_path)
    print(f"Loss curve saved to {plot_path}")
    plt.close()

def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    # Add some example images to test folder and load them here
    # For now just a placeholder
    print("Example generation not implemented yet")
    model.train()

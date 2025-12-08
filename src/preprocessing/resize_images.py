import os
import sys
# Add project root to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 

from PIL import Image
from tqdm import tqdm
import concurrent.futures
from config.config import config

def resize_image(args):
    img_path, save_path, size = args
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize(size, Image.Resampling.LANCZOS)
            img.save(save_path, "JPEG", quality=90)
    except Exception as e:
        print(f"Error resizing {img_path}: {e}")

def main():
    source_dir = config.DATA_DIR
    # If DATA_DIR is already the resized one, we need to find the original
    if "Resized" in source_dir:
        print("Config points to Resized folder already. Please check paths.")
        return

    target_dir = source_dir + "_Resized"
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    print(f"Resizing images from {source_dir} to {target_dir}...")
    print(f"Target Size: {config.IMAGE_SIZE}")
    
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    tasks = []
    for img_file in image_files:
        src_path = os.path.join(source_dir, img_file)
        dst_path = os.path.join(target_dir, img_file)
        if not os.path.exists(dst_path):
            tasks.append((src_path, dst_path, config.IMAGE_SIZE))
            
    print(f"Found {len(tasks)} images to resize.")
    
    # Use ProcessPoolExecutor for CPU-bound task
    # Leave some cores free for system
    max_workers = max(1, os.cpu_count() - 2)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(resize_image, tasks), total=len(tasks)))
        
    print("Done! Update config.py to point to the new directory if you haven't already.")

if __name__ == "__main__":
    main()

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

class CaptionDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, max_length=40):
        self.root_dir = root_dir
        self.transform = transform
        self.max_length = max_length
        
        # Load captions
        # Format: image,caption (csv)
        self.df = pd.read_csv(captions_file, delimiter=',')
        
        # Rename columns to match expected internal names if necessary, or just use them directly
        # The file has 'image' and 'caption' columns based on inspection
        self.df.rename(columns={'image': 'image_name', 'caption': 'comment'}, inplace=True)
        
        self.df['image_name'] = self.df['image_name'].str.strip()
        self.df['comment'] = self.df['comment'].str.strip()
        self.df = self.df.dropna()
        
        self.captions = self.df['comment'].tolist()
        self.images = self.df['image_name'].tolist()
        
        # Initialize Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # GPT2 doesn't have a pad token, so we use eos_token as pad_token
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback for missing images or errors, return next item
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        # Tokenize caption
        # We add a special prefix to prompt the model if desired, but for direct captioning:
        # Format: [Image Feature] -> Caption
        encoding = self.tokenizer(
            caption,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return image, input_ids, attention_mask

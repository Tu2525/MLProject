import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from src.preprocessing.transforms import Vocabulary

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        
        # Get image and caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        # Initialize vocabulary and build it
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())
        
        # Optimization: Pre-numericalize captions to save CPU time during training
        self.numericalized_captions = []
        for caption in self.captions:
            numericalized_caption = [self.vocab.stoi["<SOS>"]]
            numericalized_caption += self.vocab.numericalize(caption)
            numericalized_caption.append(self.vocab.stoi["<EOS>"])
            self.numericalized_captions.append(torch.tensor(numericalized_caption))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id = self.imgs[index]
        img_path = os.path.join(self.root_dir, img_id)
        
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Handle missing images gracefully, maybe return a dummy or skip
            # For simplicity, let's create a black image if missing
            img = Image.new('RGB', (299, 299), color='black')

        if self.transform is not None:
            img = self.transform(img)

        return img, self.numericalized_captions[index]

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets

def get_loaders(root_folder, annotation_file, transform, batch_size=32, num_workers=4, shuffle=True, pin_memory=True, val_split=0.2):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )
    
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return train_loader, val_loader, dataset

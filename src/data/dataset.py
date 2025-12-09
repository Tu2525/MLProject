import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from tqdm import tqdm
from src.preprocessing.transforms import Vocabulary

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5, vocab=None, cache_images=False, shared_cache=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        self.cache_images = cache_images
        
        # Use shared cache if provided, otherwise create new dict
        self.cached_images = shared_cache if shared_cache is not None else {}
        
        # Get image and caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        # Initialize vocabulary and build it
        if vocab is None:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocabulary(self.captions.tolist())
        else:
            self.vocab = vocab
            
        # Pre-cache images if requested AND not already cached (shared_cache might be populated)
        if self.cache_images and not self.cached_images:
            print("Pre-loading images into RAM to bypass NTFS bottleneck...")
            unique_imgs = self.imgs.unique()
            for img_id in tqdm(unique_imgs, desc="Caching Images"):
                img_path = os.path.join(self.root_dir, img_id)
                try:
                    with Image.open(img_path) as img:
                        # Convert to RGB and load into memory
                        self.cached_images[img_id] = img.convert("RGB")
                except FileNotFoundError:
                    pass
        
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
        
        if self.cache_images and img_id in self.cached_images:
            img = self.cached_images[img_id]
        else:
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

def get_loaders(root_folder, annotation_file, train_transform, val_transform, batch_size=32, num_workers=4, shuffle=True, pin_memory=True, val_split=0.2, cache_images=False):
    # 1. Create a base dataset to build the vocabulary
    # Note: We don't cache here to save time/memory, only cache in the actual datasets if needed
    base_dataset = FlickrDataset(root_folder, annotation_file, transform=None, cache_images=False)
    vocab = base_dataset.vocab
    pad_idx = vocab.stoi["<PAD>"]

    # Load cache ONCE if requested
    shared_cache = {}
    if cache_images:
        print("Pre-loading images into SHARED RAM cache...")
        # We can use the base_dataset logic to load images since it has the file list
        # Or just let the first dataset populate it.
        # Let's let the first dataset populate it, and pass the dict to the second.
        pass

    # 2. Create two separate datasets with shared vocabulary but different transforms
    # We pass cache_images here. If True, it will load images into RAM.
    train_dataset = FlickrDataset(root_folder, annotation_file, transform=train_transform, vocab=vocab, cache_images=cache_images, shared_cache=shared_cache)
    # The second dataset will see the populated shared_cache
    val_dataset = FlickrDataset(root_folder, annotation_file, transform=val_transform, vocab=vocab, cache_images=cache_images, shared_cache=shared_cache)

    dataset_size = len(base_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # 3. Generate indices for the split
    # We use random_split logic but apply it to indices so we can use Subset
    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 4. Create Subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
        persistent_workers=False, # Disable persistent workers for validation to free resources
        prefetch_factor=2 if num_workers > 0 else None
    )

    return train_loader, val_loader, base_dataset

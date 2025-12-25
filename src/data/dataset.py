import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from tqdm import tqdm
import spacy

try:
    spacy_eng = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in tqdm(sentence_list, desc="Building Vocab"):
            for word in self.tokenizer_eng(sentence):
                frequencies[word] = frequencies.get(word, 0) + 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5, cache_images=True):
        self.root_dir = root_dir
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {}  # Cache for loaded images
        
        # Robust CSV reading
        try:
            self.df = pd.read_csv(captions_file, delimiter='|')
            if len(self.df.columns) < 2:
                 self.df = pd.read_csv(captions_file, delimiter=',')
        except:
             self.df = pd.read_csv(captions_file, delimiter=',', engine='python', on_bad_lines='skip')

        self.df.columns = [c.strip() for c in self.df.columns]
        if 'image_name' not in self.df.columns:
             self.df.rename(columns={self.df.columns[0]: 'image_name', self.df.columns[1]: 'comment'}, inplace=True)

        self.df['image_name'] = self.df['image_name'].astype(str).str.strip()
        self.df['comment'] = self.df['comment'].astype(str).str.strip()
        
        self.imgs = self.df["image_name"].tolist()
        self.captions = self.df["comment"].tolist()

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_id = self.imgs[index]
        caption = self.captions[index]
        img_path = os.path.join(self.root_dir, img_id)
        
        # Use cached image if available
        if self.cache_images and img_id in self.image_cache:
            img = self.image_cache[img_id]
        else:
            try:
                img = Image.open(img_path).convert("RGB")
                if self.cache_images:
                    self.image_cache[img_id] = img
            except:
                img = Image.new('RGB', (224, 224))

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = torch.stack([item[0] for item in batch], dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return imgs, targets

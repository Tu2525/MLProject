import torch
import spacy
from collections import Counter
from torchvision import transforms
from PIL import Image

try:
    spacy_eng = spacy.load("en_core_web_sm")
except:
    print("Spacy model not found. Please run: python -m spacy download en_core_web_sm")
    pass

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        # Optimization: Use simple split if spacy is too slow, or keep spacy if accuracy is needed.
        # For speed, let's switch to a simpler regex or basic split if spacy is the bottleneck.
        # But for now, let's keep spacy but ensure we don't reload it.
        # Alternatively, we can use a faster tokenizer like 'en_core_web_sm' with disable=['parser', 'ner']
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        # Optimization: Pre-tokenize all sentences once if possible, but for large datasets do it iteratively
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

def get_transforms(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

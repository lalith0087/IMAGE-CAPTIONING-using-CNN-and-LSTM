"""Vocabulary built from Flickr8k captions, with special tokens for
sequence start/end/padding/unknown."""
import pickle
import re
from collections import Counter

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return text.split()


class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {}
        self.stoi = {}

    def __len__(self):
        return len(self.itos)

    def build(self, captions):
        counter = Counter()
        for caption in captions:
            counter.update(tokenize(caption))

        words = [PAD, START, END, UNK] + [
            w for w, c in counter.items() if c >= self.freq_threshold
        ]
        self.itos = {i: w for i, w in enumerate(words)}
        self.stoi = {w: i for i, w in enumerate(words)}

    def numericalize(self, text):
        tokens = tokenize(text)
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, indices):
        words = []
        for i in indices:
            word = self.itos.get(int(i), UNK)
            if word == END:
                break
            if word not in (PAD, START):
                words.append(word)
        return " ".join(words)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"itos": self.itos, "stoi": self.stoi,
                         "freq_threshold": self.freq_threshold}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        vocab = cls(data["freq_threshold"])
        vocab.itos = data["itos"]
        vocab.stoi = data["stoi"]
        return vocab

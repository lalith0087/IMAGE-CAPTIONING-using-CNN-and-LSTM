"""Torch Dataset over raw images + tokenized captions. Images are loaded and
transformed on the fly since the encoder now fine-tunes its last block, so
features can no longer be precomputed once and cached."""
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset

from data_loader import IMAGE_DIR, load_captions
from vocabulary import START, END, PAD

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.encoder import image_transform


class CaptionDataset(Dataset):
    """Each item is one (image, caption) pair. Since every image has
    multiple captions, one caption is sampled per image per __getitem__
    call so training sees a random caption each epoch.
    """

    def __init__(self, vocab, image_files=None, max_len=30):
        self.vocab = vocab
        self.max_len = max_len
        self.image_to_captions = load_captions()
        self.image_files = image_files or sorted(self.image_to_captions.keys())
        self.image_files = [f for f in self.image_files
                             if os.path.exists(os.path.join(IMAGE_DIR, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image = Image.open(os.path.join(IMAGE_DIR, filename)).convert("RGB")
        tensor = image_transform(image)
        caption = random.choice(self.image_to_captions[filename])

        tokens = [self.vocab.stoi[START]] + \
            self.vocab.numericalize(caption)[: self.max_len - 2] + \
            [self.vocab.stoi[END]]

        return tensor, torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch, pad_idx):
    images, captions = zip(*batch)
    images = torch.stack(images)

    lengths = [len(c) for c in captions]
    max_len = max(lengths)
    padded = torch.full((len(captions), max_len), pad_idx, dtype=torch.long)
    for i, c in enumerate(captions):
        padded[i, :len(c)] = c

    return images, padded, torch.tensor(lengths, dtype=torch.long)


def make_collate_fn(vocab):
    pad_idx = vocab.stoi[PAD]
    return lambda batch: collate_fn(batch, pad_idx)

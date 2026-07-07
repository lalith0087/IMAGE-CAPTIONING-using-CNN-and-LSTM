"""Loads the Flickr8k token file into an image -> [captions] mapping."""
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TOKEN_FILE = os.path.join(DATA_DIR, "Flickr8k.token.txt")
IMAGE_DIR = os.path.join(DATA_DIR, "Flickr8k_Dataset")
VOCAB_FILE = os.path.join(DATA_DIR, "vocab.pkl")


def load_captions(token_file=TOKEN_FILE):
    """Returns dict: image_filename -> list of raw caption strings."""
    image_to_captions = defaultdict(list)
    with open(token_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_id, caption = line.split("\t")
            image_filename = image_id.split("#")[0]
            image_to_captions[image_filename].append(caption)
    return dict(image_to_captions)

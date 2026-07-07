"""Builds and saves the vocabulary from all Flickr8k captions."""
import sys

from data_loader import load_captions, VOCAB_FILE
from vocabulary import Vocabulary


def main():
    image_to_captions = load_captions()
    all_captions = [c for caps in image_to_captions.values() for c in caps]
    print(f"Loaded {len(all_captions)} captions for {len(image_to_captions)} images")

    vocab = Vocabulary(freq_threshold=5)
    vocab.build(all_captions)
    vocab.save(VOCAB_FILE)
    print(f"Vocabulary size: {len(vocab)} -> saved to {VOCAB_FILE}")


if __name__ == "__main__":
    sys.exit(main())

"""Generates a caption for a single new image using a trained checkpoint."""
import argparse
import os
import sys

import torch
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.encoder import EncoderCNN, image_transform
from models.decoder import DecoderRNN
from data_loader import VOCAB_FILE
from vocabulary import Vocabulary, START, END


def load_models(checkpoint_path, vocab, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder = EncoderCNN(checkpoint["embed_size"]).to(device)
    decoder = DecoderRNN(checkpoint["embed_size"], checkpoint["hidden_size"],
                          len(vocab), checkpoint["num_layers"]).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def generate_caption(image_path, encoder, decoder, vocab, device, max_len=20):
    image = Image.open(image_path).convert("RGB")
    tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embed = encoder(tensor)
        token_ids = decoder.sample(image_embed, vocab.stoi[START], vocab.stoi[END], max_len)

    return vocab.decode(token_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=os.path.join(
        os.path.dirname(__file__), "..", "checkpoints", "best.pth"))
    parser.add_argument("--max-len", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary.load(VOCAB_FILE)

    encoder, decoder = load_models(args.checkpoint, vocab, device)

    caption = generate_caption(args.image, encoder, decoder, vocab, device, args.max_len)
    print(caption)


if __name__ == "__main__":
    main()

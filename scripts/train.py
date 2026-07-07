"""Trains the EncoderCNN (fine-tuned last Inception block + projection) and
DecoderRNN (LSTM) end-to-end on Flickr8k images and captions."""
import argparse
import os
import random
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from data_loader import VOCAB_FILE
from vocabulary import Vocabulary, PAD
from dataset import CaptionDataset, make_collate_fn

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cnn-lr", type=float, default=1e-5,
                         help="Lower LR for the fine-tuned Inception layers, "
                              "which start from ImageNet-pretrained weights.")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab = Vocabulary.load(VOCAB_FILE)
    print(f"Vocabulary size: {len(vocab)}")

    full_dataset = CaptionDataset(vocab)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed))
    print(f"Train images: {train_size}, Val images: {val_size}")

    collate = make_collate_fn(vocab)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate)

    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    pad_idx = vocab.stoi[PAD]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    cnn_params = [p for p in encoder.inception.parameters() if p.requires_grad]
    other_params = list(encoder.linear.parameters()) + list(encoder.bn.parameters()) + \
        list(decoder.parameters())
    optimizer = torch.optim.Adam([
        {"params": cnn_params, "lr": args.cnn_lr},
        {"params": other_params, "lr": args.lr},
    ])

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        train_loss = run_epoch(encoder, decoder, train_loader, criterion, optimizer, device)

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            val_loss = run_epoch(encoder, decoder, val_loader, criterion, None, device)

        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        checkpoint = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "embed_size": args.embed_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "epoch": epoch,
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "last.pth"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best.pth"))
            print(f"  -> new best (val_loss={val_loss:.4f}), saved checkpoints/best.pth")


def run_epoch(encoder, decoder, loader, criterion, optimizer, device):
    total_loss = 0.0
    total_tokens = 0
    training = optimizer is not None

    for images, captions, lengths in tqdm(loader, leave=False):
        images = images.to(device)
        captions = captions.to(device)

        image_embeds = encoder(images)
        outputs = decoder(image_embeds, captions)  # (batch, seq_len, vocab)

        loss = criterion(outputs.reshape(-1, outputs.size(-1)), captions.reshape(-1))

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * captions.size(0)
        total_tokens += captions.size(0)

    return total_loss / max(total_tokens, 1)


if __name__ == "__main__":
    main()
